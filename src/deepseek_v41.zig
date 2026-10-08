//! DeepSeek-V4.1 (deepseek_v41), served from MLX ops. Its own module beside
//! deepseek_v4: V4.1 shares compressed KV and index keys across layers,
//! prefilters candidate blocks, staggers the hyper-connections (each sublayer
//! collapses with the mix the previous one computed), adds Engram, and has no
//! hash-routed layers. References: DeepSeek's inference/model.py (rev
//! dba1be0a) and mlx-lm's deepseek_v41.py; fixtures from
//! tests/dump_dsv41_fixtures.py.
//!
//! The request's state lives on the model, so one request runs at a time.
//! A chunk runs one layer at a time over all of its spans: each layer's
//! experts are read once per chunk.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const transformer = @import("transformer.zig");
const log = @import("log.zig");
const engram = @import("dsv41_engram.zig");
const glm5 = @import("glm5_next.zig");
const dsv4 = @import("deepseek_v4.zig");

const A = mlx.mlx_array;
const S = mlx.mlx_stream;
const ModelConfig = model.ModelConfig;
const QuantParams = transformer.QuantParams;
const none: A = .{ .ctx = null };

/// Rows one attention span handles: the compressed rows each query gathers
/// are its largest transient ([span, index_topk, head_dim] f32).
pub const SPAN: usize = 256;
/// Window rows kept past the window, so a verify chunk can write its rows
/// and still roll back to any prefix.
const RING_SLACK: usize = 16;
/// The largest per-head indexer score block one span materializes.
pub const INDEXER_BLOCK_BYTES: usize = 256 << 20;

// ── op scope: every intermediate a forward makes, freed together ──────────

const Scope = struct {
    a: std.mem.Allocator,
    s: S,
    items: std.ArrayList(A) = .empty,

    fn init(a: std.mem.Allocator, s: S) Scope {
        return .{ .a = a, .s = s };
    }
    fn deinit(self: *Scope) void {
        for (self.items.items) |x| _ = mlx.mlx_array_free(x);
        self.items.deinit(self.a);
    }
    fn keep(self: *Scope, x: A) !A {
        self.items.append(self.a, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }
    fn res(self: *Scope, rc: c_int, o: *const A) !A {
        mlx.check(rc) catch |e| {
            _ = mlx.mlx_array_free(o.*);
            return e;
        };
        return self.keep(o.*);
    }
    /// A handle on `x` that outlives the scope (caller frees).
    fn out(_: *Scope, x: A) A {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        return o;
    }
    fn op1(sc: *Scope, comptime func: anytype, x: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(func(&o, x, sc.s), &o);
    }
    fn op2(sc: *Scope, comptime func: anytype, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(func(&o, x, y, sc.s), &o);
    }
    fn add(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_add, x, y);
    }
    fn sub(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_subtract, x, y);
    }
    fn mul(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_multiply, x, y);
    }
    fn div(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_divide, x, y);
    }
    fn matmul(sc: *Scope, x: A, y: A) !A {
        return sc.op2(mlx.mlx_matmul, x, y);
    }
    fn f(sc: *Scope, v: f32) !A {
        return sc.keep(mlx.mlx_array_new_float(v));
    }
    fn u(sc: *Scope, v: u32) !A {
        return sc.keep(mlx.mlx_array_new_data(&v, &[_]c_int{}, 0, .uint32));
    }
    fn i(sc: *Scope, v: i32) !A {
        return sc.keep(mlx.mlx_array_new_int(v));
    }
    fn ints(sc: *Scope, data: []const i32, shape: []const c_int) !A {
        return sc.keep(mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .int32));
    }
    fn astype(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        if (mlx.mlx_array_dtype(x) == dt) return x;
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_astype(&o, x, dt, sc.s), &o);
    }
    fn view(sc: *Scope, x: A, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_view(&o, x, dt, sc.s), &o);
    }
    fn reshape(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    fn transpose(sc: *Scope, x: A, axes: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, sc.s), &o);
    }
    fn t2(sc: *Scope, x: A) !A {
        return sc.transpose(x, &.{ 1, 0 });
    }
    fn broadcast(sc: *Scope, x: A, shape: []const c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_broadcast_to(&o, x, shape.ptr, shape.len, sc.s), &o);
    }
    fn expand(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_expand_dims(&o, x, axis, sc.s), &o);
    }
    /// x[..., lo:hi, ...] on one axis (negative axis counts from the end).
    fn slice(sc: *Scope, x: A, axis_in: isize, lo: usize, hi: usize) !A {
        const sh = mlx.getShape(x);
        const axis: usize = if (axis_in < 0) @intCast(@as(isize, @intCast(sh.len)) + axis_in) else @intCast(axis_in);
        var start: [8]c_int = @splat(0);
        var stop: [8]c_int = undefined;
        const st: [8]c_int = @splat(1);
        @memcpy(stop[0..sh.len], sh);
        start[axis] = @intCast(lo);
        stop[axis] = @intCast(hi);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_slice(&o, x, &start, sh.len, &stop, sh.len, &st, sh.len, sc.s), &o);
    }
    fn concat(sc: *Scope, xs: []const A, axis: c_int) !A {
        if (xs.len == 1) return xs[0];
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_concatenate_axis(&o, vec, axis, sc.s), &o);
    }
    fn stack(sc: *Scope, xs: []const A, axis: c_int) !A {
        const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_stack_axis(&o, vec, axis, sc.s), &o);
    }
    fn take(sc: *Scope, x: A, idx: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_axis(&o, x, idx, axis, sc.s), &o);
    }
    fn takeAlong(sc: *Scope, x: A, idx: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_take_along_axis(&o, x, idx, axis, sc.s), &o);
    }
    fn reduce(sc: *Scope, comptime f_: anytype, x: A, axis: c_int, keep_: bool) !A {
        var o = mlx.mlx_array_new();
        return sc.res(f_(&o, x, axis, keep_, sc.s), &o);
    }
    fn sum(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_sum_axis, x, axis, keep_);
    }
    fn mean(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_mean_axis, x, axis, keep_);
    }
    fn max(sc: *Scope, x: A, axis: c_int, keep_: bool) !A {
        return sc.reduce(mlx.mlx_max_axis, x, axis, keep_);
    }
    fn argmax(sc: *Scope, x: A, axis: c_int) !A {
        return sc.reduce(mlx.mlx_argmax_axis, x, axis, false);
    }
    fn softmax(sc: *Scope, x: A, axis: c_int) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_softmax_axis(&o, x, axis, true, sc.s), &o);
    }
    /// The first `k` positions of `x` along the last axis, largest first (unordered).
    fn topk(sc: *Scope, x: A, k: usize) !A {
        const neg = try sc.op1(mlx.mlx_negative, x);
        var o = mlx.mlx_array_new();
        const part = try sc.res(mlx.mlx_argpartition_axis(&o, neg, @intCast(k - 1), -1, sc.s), &o);
        return sc.slice(part, -1, 0, k);
    }
    fn where(sc: *Scope, c: A, x: A, y: A) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_where(&o, c, x, y, sc.s), &o);
    }
    fn rms(sc: *Scope, x: A, w: A, eps: f32) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_fast_rms_norm(&o, x, w, eps, sc.s), &o);
    }
    fn arange(sc: *Scope, lo: usize, hi: usize) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_arange(&o, @floatFromInt(lo), @floatFromInt(hi), 1, .int32, sc.s), &o);
    }
    fn zeros(sc: *Scope, shape: []const c_int, dt: mlx.mlx_dtype) !A {
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_zeros(&o, shape.ptr, shape.len, dt, sc.s), &o);
    }
    fn clip(sc: *Scope, x: A, lo: f32, hi: f32) !A {
        return sc.op2(mlx.mlx_minimum, try sc.op2(mlx.mlx_maximum, x, try sc.f(lo)), try sc.f(hi));
    }
    fn sigmoid(sc: *Scope, x: A) !A {
        return sc.op1(mlx.mlx_sigmoid, x);
    }
};

fn dim(x: A, axis: isize) usize {
    const sh = mlx.getShape(x);
    const ax: usize = if (axis < 0) @intCast(@as(isize, @intCast(sh.len)) + axis) else @intCast(axis);
    return @intCast(sh[ax]);
}

fn ci(v: usize) c_int {
    return @intCast(v);
}

// ── weights ──────────────────────────────────────────────────────────────

/// One linear: a plain weight `[out, in]`, or a quantized one whose mode,
/// bits and group size are read off its own geometry.
pub const Lin = struct {
    w: A,
    s: A = none,
    b: A = none,
    qp: QuantParams = .{ .bits = 0, .group_size = 0 },

    fn apply(self: *const Lin, sc: *Scope, x: A) !A {
        if (self.s.ctx == null) return sc.matmul(x, try sc.t2(self.w));
        var o = mlx.mlx_array_new();
        return sc.res(mlx.mlx_quantized_matmul(&o, x, self.w, self.s, self.b, true, mlx.mlx_optional_int.some(@intCast(self.qp.group_size)), mlx.mlx_optional_int.some(@intCast(self.qp.bits)), self.qp.mode.cstr(), sc.s), &o);
    }

    /// Rows `ids` of the table, dequantized to `dt`.
    fn rows(self: *const Lin, sc: *Scope, ids: A, dt: mlx.mlx_dtype) !A {
        if (self.s.ctx == null) return sc.astype(try sc.take(self.w, ids, 0), dt);
        return sc.keep(try transformer.gatherQuantizedRows(sc.s, self.w, self.s, self.b, ids, self.qp.group_size, self.qp.bits, self.qp.mode, none, .{ .value = dt, .has_value = true }));
    }
};

const Hc = struct {
    fn_w: A, // f32 [mix, hc * dim]
    scale: A, // f32 [mix]: scale[0] for pre, [1] post, [2] comb
    base: A, // f32 [mix]
};

const Comp = struct { wkv: Lin, wgate: ?Lin, norm: A };
const Idx = struct { wq_b: Lin, weights_proj: Lin, wk: ?Lin, k_norm: A };

/// Stacked routed experts `[E, out, in]`; `gather_qmm` reads them by index.
const Bank = struct {
    w: A,
    s: A = none,
    b: A = none,
    qp: QuantParams = .{ .bits = 0, .group_size = 0 },
};

const Experts = struct { gate: Bank, up: Bank, down: Bank };

const Layer = struct {
    attn_norm: A,
    ffn_norm: A,
    q_norm: A,
    kv_norm: A,
    hc_attn: Hc,
    hc_ffn: Hc,
    wq_a: Lin,
    wq_b: Lin,
    wkv: Lin,
    /// Grouped output LoRA as `[o_groups, o_lora_rank, in]` views.
    wo_a: Lin,
    wo_b: Lin,
    sink: A, // f32 [1, heads, 1]
    ratio: u8,
    comp: ?Comp,
    idx: ?Idx,
    gate_w: A, // f32 [E, dim]
    gate_bias: A, // f32 [E]
    top_k: usize,
    shared: [3]Lin, // w1 (gate), w3 (up), w2 (down)
    experts: Experts,
};

const EngramLayer = struct {
    layer: usize,
    wkv: Lin,
    qk: A, // f32 [hc, dim]: q_weight * k_weight
    table: engram.Table,
};

const DsparkW = struct {
    main_proj: Lin,
    main_norm: A,
    norm: A, // the last stage's
    markov_embed: A, // [vocab, rank]
    markov_head: A, // f32 [vocab, rank]
    conf: A, // f32 [1, dim + rank]
};

/// Rope as MLX `fast_rope` periods (`1 / inv_freq`), and their negation
/// for the inverse rotation.
const Rope = struct { fwd: A, inv: A };

pub const Dsv41Model = struct {
    gpa: std.mem.Allocator,
    s: S,
    cfg: ModelConfig,
    act: mlx.mlx_dtype,
    embed: Lin,
    head: Lin,
    norm: A,
    /// The trunk, then the DSpark stages.
    layers: []Layer,
    n_layers: usize,
    vocab: usize,
    n_mtp: usize = 0,
    ds_block: usize = 0,
    /// Rows per attention span (tests narrow it to drive several spans).
    span: usize = SPAN,
    /// Bytes per indexer score block (tests narrow it to drive several).
    idx_block_bytes: usize = INDEXER_BLOCK_BYTES,
    /// Test-only: named intermediates of a one-span forward.
    trace: ?*Trace = null,
    /// Test-only: skip the QAT round-trips (the reference's plain-math mode).
    qat_off: bool = false,
    dspark: ?DsparkW = null,
    hash: ?engram.Hash = null,
    token_map: []u32 = &.{},
    engrams: []EngramLayer = &.{},
    rope_plain: Rope,
    rope_yarn: Rope,
    split_kernel: mlx.mlx_fast_metal_kernel,
    /// `qat` per kind, compiled at init (null: the op chain).
    qat_fns: [3]mlx.mlx_closure = @splat(.{}),
    /// Per trunk layer: its decode-width stretches compiled (`Region`).
    regions: []LayerRegions = &.{},
    /// Arrays made at load (casts, reshapes, stacks); freed in `deinit`.
    owned: std.ArrayList(A) = .empty,
    dec_state: ?State = null,
    /// Bumped whenever `dec_state` is replaced: a snapshot names the state it was taken of.
    state_gen: u64 = 0,
    /// The last prompt's state one token short of its end (`resumePrompt`).
    snap: ?Snap = null,
    /// Where this prompt's snapshot is taken (`armSnapshot`), and its tokens.
    snap_at: ?usize = null,
    snap_ids: []u32 = &.{},
    ds_conf_thr: f32 = -std.math.inf(f32),
    ds_prof: ?dsv4.DsparkProfile = null,
    ds_prof_head_ns: u64 = 0,
    ds_prof_comp_sync_ns: u64 = 0,

    pub fn deinit(self: *Dsv41Model) void {
        if (self.dec_state) |*st| st.deinit();
        if (self.snap) |*sn| sn.deinit(self.gpa);
        self.gpa.free(self.snap_ids);
        for (self.engrams) |*e| e.table.close();
        self.gpa.free(self.engrams);
        self.gpa.free(self.token_map);
        for (self.owned.items) |x| _ = mlx.mlx_array_free(x);
        self.owned.deinit(self.gpa);
        self.gpa.free(self.layers);
        _ = mlx.mlx_fast_metal_kernel_free(self.split_kernel);
        for (self.qat_fns) |f| if (f.ctx != null) {
            _ = mlx.mlx_closure_free(f);
        };
        freeRegions(self);
        self.gpa.destroy(self);
    }

    fn own(self: *Dsv41Model, x: A) !A {
        self.owned.append(self.gpa, x) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        return x;
    }

    fn isKvSource(self: *const Dsv41Model, l: usize) bool {
        return l < self.n_layers and (self.cfg.dsv41_kv_sources >> @intCast(l)) & 1 != 0;
    }
    fn isIndexSource(self: *const Dsv41Model, l: usize) bool {
        return l < self.n_layers and (self.cfg.dsv41_index_sources >> @intCast(l)) & 1 != 0;
    }
    fn ratio(self: *const Dsv41Model, l: usize) usize {
        return if (l < self.n_layers) self.cfg.dsv4_compress_ratios[l] else 0;
    }
    /// The latest layer at or above `l` that matches `pred` (config parse
    /// proved one exists wherever a compressed layer asks).
    fn latest(self: *const Dsv41Model, l: usize, comptime pred: fn (*const Dsv41Model, usize) bool) usize {
        var j = l + 1;
        while (j > 0) : (j -= 1) {
            if (pred(self, j - 1)) return j - 1;
        }
        unreachable;
    }
    fn isOwner(self: *const Dsv41Model, l: usize) bool {
        return self.isKvSource(l) and self.isIndexSource(l);
    }
    fn dsparkTarget(self: *const Dsv41Model, l: usize) ?usize {
        for (self.cfg.dsv4_dspark_target_layers[0..self.cfg.dsv4_n_dspark_target_layers], 0..) |t, k| {
            if (t == l) return k;
        }
        return null;
    }
};

const NameBuf = [192]u8;

const Trace = struct {
    gpa: std.mem.Allocator,
    items: std.ArrayList(struct { name: []u8, arr: A }) = .empty,

    fn put(self: *Trace, comptime fmt: []const u8, args: anytype, x: A) !void {
        var o = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&o, x);
        try self.items.append(self.gpa, .{ .name = try std.fmt.allocPrint(self.gpa, fmt, args), .arr = o });
    }

    fn deinit(self: *Trace) void {
        for (self.items.items) |it| {
            self.gpa.free(it.name);
            _ = mlx.mlx_array_free(it.arr);
        }
        self.items.deinit(self.gpa);
    }
};

fn getW(w: *const model.Weights, comptime fmt: []const u8, args: anytype) !A {
    var buf: NameBuf = undefined;
    const name = std.fmt.bufPrint(&buf, fmt, args) catch return error.NameTooLong;
    return w.get(name) orelse {
        log.err("deepseek_v41: missing weight {s}\n", .{name});
        return error.MissingWeight;
    };
}

fn getOpt(w: *const model.Weights, comptime fmt: []const u8, args: anytype) ?A {
    var buf: NameBuf = undefined;
    const name = std.fmt.bufPrint(&buf, fmt, args) catch return null;
    return w.get(name);
}

fn getLin(w: *const model.Weights, in_dim: usize, comptime base: []const u8, args: anytype) !Lin {
    const wt = try getW(w, base ++ ".weight", args);
    const sc = getOpt(w, base ++ ".scales", args) orelse return .{ .w = wt };
    const bi = getOpt(w, base ++ ".biases", args) orelse none;
    const qp = transformer.quantParamsFromGeometry(wt, sc, bi.ctx != null, @intCast(in_dim)) orelse {
        var buf: NameBuf = undefined;
        log.err("deepseek_v41: {s} has a quantized layout this build does not read\n", .{std.fmt.bufPrint(&buf, base, args) catch base});
        return error.Dsv41QuantGeometry;
    };
    return .{ .w = wt, .s = sc, .b = bi, .qp = qp };
}

fn bankOf(l: Lin) Bank {
    return .{ .w = l.w, .s = l.s, .b = l.b, .qp = l.qp };
}

/// `x` cast to `dt` and evaluated, owned by the model.
fn ownCast(m: *Dsv41Model, x: A, dt: mlx.mlx_dtype) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, m.s));
    _ = try m.own(o);
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

fn ownReshape(m: *Dsv41Model, x: A, shape: []const c_int) !A {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, m.s));
    _ = try m.own(o);
    try mlx.check(mlx.mlx_array_eval(o));
    return o;
}

fn loadHc(m: *Dsv41Model, w: *const model.Weights, comptime pfx: []const u8, li: usize, comptime which: []const u8) !Hc {
    const hc = m.cfg.dsv4_hc_mult;
    const scale = try ownCast(m, try getW(w, pfx ++ ".{d}.hc_" ++ which ++ "_scale", .{li}), .float32);
    var ids: [64]i32 = undefined;
    const mix = (2 + hc) * hc;
    for (0..mix) |k| ids[k] = if (k < hc) 0 else if (k < 2 * hc) 1 else 2;
    const idx = mlx.mlx_array_new_data(&ids, &[_]c_int{ci(mix)}, 1, .int32);
    defer _ = mlx.mlx_array_free(idx);
    var sv = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_take_axis(&sv, scale, idx, 0, m.s));
    _ = try m.own(sv);
    try mlx.check(mlx.mlx_array_eval(sv));
    return .{
        .fn_w = try ownCast(m, try getW(w, pfx ++ ".{d}.hc_" ++ which ++ "_fn", .{li}), .float32),
        .scale = sv,
        .base = try ownCast(m, try getW(w, pfx ++ ".{d}.hc_" ++ which ++ "_base", .{li}), .float32),
    };
}

/// Routed experts: stacked banks under the pack's names, or per-expert
/// tensors stacked here.
fn loadExperts(m: *Dsv41Model, w: *model.Weights, comptime pfx: []const u8, li: usize, n_experts: usize) !Experts {
    const d = m.cfg.hidden_size;
    const inter = m.cfg.moe_intermediate_size;
    if (getOpt(w, pfx ++ ".{d}.ffn.experts.gate_proj.weight", .{li}) != null) return .{
        .gate = bankOf(try getLin(w, d, pfx ++ ".{d}.ffn.experts.gate_proj", .{li})),
        .up = bankOf(try getLin(w, d, pfx ++ ".{d}.ffn.experts.up_proj", .{li})),
        .down = bankOf(try getLin(w, inter, pfx ++ ".{d}.ffn.experts.down_proj", .{li})),
    };
    if (getOpt(w, pfx ++ ".{d}.ffn.experts.w1.weight", .{li}) != null) return .{
        .gate = bankOf(try getLin(w, d, pfx ++ ".{d}.ffn.experts.w1", .{li})),
        .up = bankOf(try getLin(w, d, pfx ++ ".{d}.ffn.experts.w3", .{li})),
        .down = bankOf(try getLin(w, inter, pfx ++ ".{d}.ffn.experts.w2", .{li})),
    };
    if (getOpt(w, pfx ++ ".{d}.ffn.experts.0.w1.weight", .{li}) != null) {
        var banks: [3]Bank = undefined;
        inline for (.{ "w1", "w3", "w2" }, 0..) |proj, k| {
            const in_dim = if (k == 2) inter else d;
            var parts: [3]std.ArrayList(A) = .{ .empty, .empty, .empty };
            defer for (&parts) |*p| p.deinit(m.gpa);
            var first: Lin = undefined;
            for (0..n_experts) |e| {
                const l = try getLin(w, in_dim, pfx ++ ".{d}.ffn.experts.{d}." ++ proj, .{ li, e });
                if (e == 0) first = l;
                if (l.qp.bits != first.qp.bits or l.qp.group_size != first.qp.group_size or l.qp.mode != first.qp.mode or (l.b.ctx == null) != (first.b.ctx == null)) return error.Dsv41QuantGeometry;
                try parts[0].append(m.gpa, l.w);
                if (l.s.ctx != null) try parts[1].append(m.gpa, l.s);
                if (l.b.ctx != null) try parts[2].append(m.gpa, l.b);
            }
            var bank: Bank = .{ .w = undefined, .qp = first.qp };
            inline for (.{ .{ "w", "weight" }, .{ "s", "scales" }, .{ "b", "biases" } }, 0..) |f, p| {
                if (parts[p].items.len > 0) {
                    // The loads first: a GPU stack waiting on hundreds of queued disk reads times out.
                    try evalAll(parts[p].items);
                    const vec = mlx.mlx_vector_array_new_data(parts[p].items.ptr, parts[p].items.len);
                    defer _ = mlx.mlx_vector_array_free(vec);
                    var o = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_stack_axis(&o, vec, 0, m.s));
                    @field(bank, f[0]) = try m.own(o);
                    try mlx.check(mlx.mlx_array_eval(o));
                    // The stack holds the experts now: the map's copies go, or the bank is resident twice.
                    for (0..n_experts) |e| {
                        var buf: NameBuf = undefined;
                        w.remove(std.fmt.bufPrint(&buf, pfx ++ ".{d}.ffn.experts.{d}." ++ proj ++ "." ++ f[1], .{ li, e }) catch return error.NameTooLong);
                    }
                }
            }
            banks[k] = bank;
        }
        return .{ .gate = banks[0], .up = banks[1], .down = banks[2] };
    }
    log.err("deepseek_v41: {s}.{d} has no routed experts\n", .{ pfx, li });
    return error.MissingWeight;
}

fn loadLayer(m: *Dsv41Model, w: *model.Weights, comptime pfx: []const u8, li: usize, trunk_li: ?usize) !Layer {
    const c = &m.cfg;
    const d = c.hidden_size;
    const heads = c.num_attention_heads;
    const hd = c.head_dim;
    const og = c.dsv4_o_groups;
    const ol = c.dsv4_o_lora_rank;
    const ratio: u8 = if (trunk_li) |t| c.dsv4_compress_ratios[t] else 0;
    const is_kv = if (trunk_li) |t| m.isKvSource(t) else false;
    const is_ix = if (trunk_li) |t| m.isIndexSource(t) else false;
    var wo_a = try getLin(w, heads * hd / og, pfx ++ ".{d}.attn.wo_a", .{li});
    inline for (.{ "w", "s", "b" }) |field| {
        const x = @field(wo_a, field);
        if (x.ctx != null) @field(wo_a, field) = try ownReshape(m, x, &.{ ci(og), ci(ol), @intCast(dim(x, -1)) });
    }
    const sink = blk: {
        const raw = try getW(w, pfx ++ ".{d}.attn.attn_sink", .{li});
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&r, raw, &[_]c_int{ 1, ci(heads), 1 }, 3, m.s));
        defer _ = mlx.mlx_array_free(r);
        break :blk try ownCast(m, r, .float32);
    };
    const n_experts: usize = if (trunk_li == null) c.dsv41_dspark_experts else c.num_experts;
    return .{
        .attn_norm = try ownCast(m, try getW(w, pfx ++ ".{d}.attn_norm.weight", .{li}), m.act),
        .ffn_norm = try ownCast(m, try getW(w, pfx ++ ".{d}.ffn_norm.weight", .{li}), m.act),
        .q_norm = try ownCast(m, try getW(w, pfx ++ ".{d}.attn.q_norm.weight", .{li}), m.act),
        .kv_norm = try ownCast(m, try getW(w, pfx ++ ".{d}.attn.kv_norm.weight", .{li}), m.act),
        .hc_attn = try loadHc(m, w, pfx, li, "attn"),
        .hc_ffn = try loadHc(m, w, pfx, li, "ffn"),
        .wq_a = try getLin(w, d, pfx ++ ".{d}.attn.wq_a", .{li}),
        .wq_b = try getLin(w, c.dsv4_q_lora_rank, pfx ++ ".{d}.attn.wq_b", .{li}),
        .wkv = try getLin(w, d, pfx ++ ".{d}.attn.wkv", .{li}),
        .wo_a = wo_a,
        .wo_b = try getLin(w, og * ol, pfx ++ ".{d}.attn.wo_b", .{li}),
        .sink = sink,
        .ratio = ratio,
        .comp = if (is_kv) .{
            .wkv = try getLin(w, d, pfx ++ ".{d}.attn.compressor.wkv", .{li}),
            .wgate = if (ratio > 1) try getLin(w, d, pfx ++ ".{d}.attn.compressor.wgate", .{li}) else null,
            .norm = try ownCast(m, try getW(w, pfx ++ ".{d}.attn.compressor.norm.weight", .{li}), m.act),
        } else null,
        .idx = if (is_ix) .{
            .wq_b = try getLin(w, c.dsv4_q_lora_rank, pfx ++ ".{d}.attn.indexer.wq_b", .{li}),
            .weights_proj = try getLin(w, d, pfx ++ ".{d}.attn.indexer.weights_proj", .{li}),
            .wk = if (is_kv) try getLin(w, hd, pfx ++ ".{d}.attn.indexer.wk", .{li}) else null,
            .k_norm = if (is_kv) try ownCast(m, try getW(w, pfx ++ ".{d}.attn.indexer.k_norm.weight", .{li}), m.act) else none,
        } else null,
        .gate_w = try ownCast(m, try getW(w, pfx ++ ".{d}.ffn.gate.weight", .{li}), .float32),
        .gate_bias = try ownCast(m, try getW(w, pfx ++ ".{d}.ffn.gate.bias", .{li}), .float32),
        .top_k = if (trunk_li == null) c.dsv41_dspark_top_k else c.num_experts_per_tok,
        .shared = .{
            try getLin(w, d, pfx ++ ".{d}.ffn.shared_experts.w1", .{li}),
            try getLin(w, d, pfx ++ ".{d}.ffn.shared_experts.w3", .{li}),
            try getLin(w, c.moe_intermediate_size, pfx ++ ".{d}.ffn.shared_experts.w2", .{li}),
        },
        .experts = try loadExperts(m, w, pfx, li, n_experts),
    };
}

/// `inference/model.py` `precompute_freqs_cis` as fast_rope periods: YaRN
/// fades the dims past the training context by `factor` when `orig > 0`.
fn buildRope(m: *Dsv41Model, base: f64, orig: u32, factor: f64, beta_fast: f64, beta_slow: f64) !Rope {
    const rd = m.cfg.dsv4_rope_head_dim;
    var fwd: [256]f32 = undefined;
    var inv: [256]f32 = undefined;
    const dimf: f64 = @floatFromInt(rd);
    var low: f64 = 0;
    var high: f64 = 0;
    if (orig > 0) {
        const corr = struct {
            fn f(rot: f64, dd: f64, b: f64, o: f64) f64 {
                return dd * @log(o / (rot * 2.0 * std.math.pi)) / (2.0 * @log(b));
            }
        }.f;
        const o: f64 = @floatFromInt(orig);
        low = @max(@floor(corr(beta_fast, dimf, base, o)), 0);
        high = @min(@ceil(corr(beta_slow, dimf, base, o)), dimf - 1);
    }
    for (0..rd / 2) |k| {
        const e: f32 = @as(f32, @floatFromInt(2 * k)) / @as(f32, @floatFromInt(rd));
        var fr: f32 = 1.0 / std.math.pow(f32, @floatCast(base), e);
        if (orig > 0) {
            const ramp: f32 = @floatCast(std.math.clamp((@as(f64, @floatFromInt(k)) - low) / @max(high - low, 1e-3), 0, 1));
            const smooth = 1 - ramp;
            fr = fr / @as(f32, @floatCast(factor)) * (1 - smooth) + fr * smooth;
        }
        fwd[k] = 1.0 / fr;
        inv[k] = -fwd[k];
    }
    const shape = [_]c_int{ci(rd / 2)};
    return .{
        .fwd = try m.own(mlx.mlx_array_new_data(&fwd, &shape, 1, .float32)),
        .inv = try m.own(mlx.mlx_array_new_data(&inv, &shape, 1, .float32)),
    };
}

/// The hyper-connection split: pre = sigmoid + eps, post = 2 sigmoid, comb =
/// row softmax + eps then Sinkhorn (a column pass, then iters-1 row/column
/// pairs, eps inside every division). One thread per row, in `hc_split_sinkhorn`'s
/// order; the row count comes from the input shape, never a template.
const SPLIT_SOURCE =
    \\uint n = thread_position_in_grid.x;
    \\if (n >= z_shape[0]) return;
    \\constexpr int MIX = (2 + HC) * HC;
    \\const float e = eps[0];
    \\const device float* zr = z + n * MIX;
    \\for (int j = 0; j < HC; ++j) {
    \\  pre[n * HC + j] = 1.0f / (1.0f + metal::exp(-zr[j])) + e;
    \\  post[n * HC + j] = 2.0f / (1.0f + metal::exp(-zr[HC + j]));
    \\}
    \\float m[HC * HC];
    \\for (int a = 0; a < HC; ++a) {
    \\  float mx = zr[2 * HC + a * HC];
    \\  for (int b = 1; b < HC; ++b) mx = metal::max(mx, zr[2 * HC + a * HC + b]);
    \\  float s = 0.0f;
    \\  for (int b = 0; b < HC; ++b) { m[a * HC + b] = metal::exp(zr[2 * HC + a * HC + b] - mx); s += m[a * HC + b]; }
    \\  for (int b = 0; b < HC; ++b) m[a * HC + b] = m[a * HC + b] / s + e;
    \\}
    \\for (int t = 0; t < 2 * ITERS - 1; ++t) {
    \\  const bool rows = t % 2 == 1;
    \\  for (int a = 0; a < HC; ++a) {
    \\    float s = 0.0f;
    \\    for (int b = 0; b < HC; ++b) s += rows ? m[a * HC + b] : m[b * HC + a];
    \\    for (int b = 0; b < HC; ++b) { const int k = rows ? a * HC + b : b * HC + a; m[k] = m[k] / (s + e); }
    \\  }
    \\}
    \\for (int k = 0; k < HC * HC; ++k) comb[n * HC * HC + k] = m[k];
;

fn buildSplitKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "z", "eps" };
    const outs = [_][*:0]const u8{ "pre", "post", "comb" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("dsv41_hc_split", iv, ov, SPLIT_SOURCE, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    return k;
}

pub const InitOpts = struct {
    /// The activation dtype; tests run f32 against the f32 reference.
    act: mlx.mlx_dtype = .bfloat16,
    /// Load the DSpark stages when the checkpoint ships them.
    dspark: bool = false,
};

pub fn init(gpa: std.mem.Allocator, cfg: *const ModelConfig, w: *model.Weights, s: S, opts: InitOpts) !*Dsv41Model {
    const m = try gpa.create(Dsv41Model);
    errdefer gpa.destroy(m);
    const n_layers = cfg.num_hidden_layers;
    const n_stages: usize = if (opts.dspark and cfg.dsv4_dspark_block_size > 0 and getOpt(w, "mtp.0.main_proj.weight", .{}) != null)
        cfg.dsv4_n_compress_ratios - n_layers
    else
        0;
    m.* = .{
        .gpa = gpa,
        .s = s,
        .cfg = cfg.*,
        .act = opts.act,
        .embed = undefined,
        .head = undefined,
        .norm = undefined,
        .layers = try gpa.alloc(Layer, n_layers + n_stages),
        .n_layers = n_layers,
        .vocab = cfg.vocab_size,
        .rope_plain = undefined,
        .rope_yarn = undefined,
        .split_kernel = try buildSplitKernel(),
    };
    m.cfg.dsv41_dir = null; // borrowed; the caller's config owns it
    errdefer {
        for (m.qat_fns) |f| if (f.ctx != null) {
            _ = mlx.mlx_closure_free(f);
        };
        freeRegions(m);
        for (m.engrams) |*e| e.table.close();
        gpa.free(m.engrams);
        gpa.free(m.token_map);
        for (m.owned.items) |x| _ = mlx.mlx_array_free(x);
        m.owned.deinit(gpa);
        gpa.free(m.layers);
        _ = mlx.mlx_fast_metal_kernel_free(m.split_kernel);
    }
    try compileQat(m);
    m.embed = try getLin(w, cfg.hidden_size, "embed", .{});
    m.head = try getLin(w, cfg.hidden_size, "head", .{});
    m.norm = try ownCast(m, try getW(w, "norm.weight", .{}), m.act);
    m.rope_plain = try buildRope(m, cfg.rope_theta, 0, 1, 0, 0);
    m.rope_yarn = try buildRope(m, cfg.dsv4_compress_rope_theta, if (cfg.rope_yarn) cfg.yarn_orig_max_pos else 0, cfg.yarn_factor, cfg.yarn_beta_fast, cfg.yarn_beta_slow);
    for (0..n_layers) |li| m.layers[li] = try loadLayer(m, w, "layers", li, li);
    try compileRegions(m);
    if (n_stages > 0) {
        for (0..n_stages) |si| m.layers[n_layers + si] = try loadLayer(m, w, "mtp", si, null);
        const last = n_stages - 1;
        const targets = cfg.dsv4_n_dspark_target_layers;
        m.dspark = .{
            .main_proj = try getLin(w, cfg.hidden_size * targets, "mtp.0.main_proj", .{}),
            .main_norm = try ownCast(m, try getW(w, "mtp.0.main_norm.weight", .{}), m.act),
            .norm = try ownCast(m, try getW(w, "mtp.{d}.norm.weight", .{last}), m.act),
            .markov_embed = try getW(w, "mtp.{d}.markov_head.embed.weight", .{last}),
            .markov_head = try ownCast(m, try getW(w, "mtp.{d}.markov_head.head.weight", .{last}), .float32),
            .conf = try ownCast(m, try getW(w, "mtp.{d}.confidence_head.proj.weight", .{last}), .float32),
        };
        m.n_mtp = n_stages;
        m.ds_block = cfg.dsv4_dspark_block_size;
        m.ds_conf_thr = dsv4.dsparkConfThreshold();
    }
    if (cfg.dsv41_dir) |dir| try initEngram(m, w, dir);
    try evalWeights(m);
    log.info("deepseek_v41: {d} layers, engram {d} layers, DSpark {d} stages\n", .{ n_layers, m.engrams.len, n_stages });
    return m;
}

/// Every array the model reads, evaluated one layer at a time. A compiled region captures its layer's weights,
/// and a lazy load in its trace would read the weight from disk into a fresh buffer on every call.
fn evalWeights(m: *Dsv41Model) !void {
    var xs: std.ArrayList(A) = .empty;
    defer xs.deinit(m.gpa);
    for (m.layers) |*ly| {
        xs.clearRetainingCapacity();
        try collectArrays(m.gpa, &xs, ly.*);
        try evalAll(xs.items);
        try wireLoaded(m);
    }
    xs.clearRetainingCapacity();
    try collectArrays(m.gpa, &xs, .{ m.embed, m.head, m.norm, m.dspark });
    for (m.engrams) |e| try collectArrays(m.gpa, &xs, .{ e.wkv, e.qk });
    try evalAll(xs.items);
}

/// Wires what has loaded so far: the residency policy takes the new buffers into the set (the scheduler applies
/// it only after init), and a GPU command buffer makes the set resident. Until then the weights are ordinary
/// pages, and a pack near the RAM size had them compressed until the kernel killed the process.
fn wireLoaded(m: *const Dsv41Model) !void {
    _ = mlx.applyWiredPolicy();
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    var o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(o);
    try mlx.check(mlx.mlx_add(&o, one, one, m.s));
    try mlx.check(mlx.mlx_array_eval(o));
}

/// The arrays inside `v` (structs, optionals and arrays of them).
fn collectArrays(gpa: std.mem.Allocator, xs: *std.ArrayList(A), v: anytype) !void {
    const T = @TypeOf(v);
    if (T == A) {
        if (v.ctx != null) try xs.append(gpa, v);
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |s| inline for (s.field_names) |name| try collectArrays(gpa, xs, @field(v, name)),
        .optional => if (v) |x| try collectArrays(gpa, xs, x),
        .array => for (v) |x| try collectArrays(gpa, xs, x),
        else => {},
    }
}

/// The prompt pass's working set over `floor`: a chunk runs one layer at a time over spans; the whole
/// chunk's stream (in and out), the picks layers hand down and the DSpark rows
/// stay live across layers, and one span at a time adds its gathered
/// compressed rows, attention scores, an indexer score block and its MoE.
/// State is the compressed caches per token plus the windows per layer.
pub fn prefillBytes(config: *const ModelConfig, seq: u64, chunk: u64, floor: u64) u64 {
    const c: u64 = @min(chunk, @max(seq, 1));
    const span: u64 = @min(SPAN, c);
    const d: u64 = config.hidden_size;
    const hd: u64 = config.head_dim;
    const topk: u64 = config.dsv4_index_topk;
    const window: u64 = config.sliding_window;
    const cand_block: u64 = @max(config.dsv41_candidate_block_size, 1);
    const state = seq * config.kvBytesPerToken() + config.num_hidden_layers * (window + 16) * hd * 2;
    const held = c * (2 * config.dsv4_hc_mult * d * 2 + topk * 4 + seq / cand_block + config.dsv4_n_dspark_target_layers * d * 2);
    const attn = span * topk * hd * 4 * 2 + span * config.num_attention_heads * ((window + topk) * 4 * 2 + hd * 4 * 2);
    const idx_bytes = @min(INDEXER_BLOCK_BYTES, span * config.dsv4_index_n_heads * seq * 4) + span * seq * 4 * 3;
    const moe_bytes = span * config.num_experts_per_tok * (2 * config.moe_intermediate_size + d) * 4;
    return (state + held + attn + idx_bytes + moe_bytes + floor) * 5 / 4;
}

/// The DSpark stages load when the pack ships them, unless `--no-mtp` or the model's `mtp` setting turn them off
/// (`applyModelSettings` folds both into `mtp_override`); V4's `MLX_SERVE_DSV4_DSPARK` turns them on either way.
pub fn dsparkWanted(cfg: *const ModelConfig) bool {
    const env = std.c.getenv("MLX_SERVE_DSV4_DSPARK");
    return (cfg.mtp_override orelse true) or (env != null and env.?[0] != '0');
}

/// What a load holds of the pack: its safetensors bytes less the Engram tables it preads and the tensors it
/// drops (the vision tower; the DSpark stages when they do not load).
pub fn residentDiskBytes(gpa: std.mem.Allocator, dir: []const u8, cfg: *const ModelConfig, shards: u64) !u64 {
    var total = shards -| try droppedBytes(gpa, dir, dsparkWanted(cfg));
    for (cfg.dsv41_engram_layers[0..cfg.dsv41_n_engram_layers], 0..) |l, k| {
        var t = try engram.Table.open(gpa, dir, k, l, cfg.dsv41_engram_rows[k], cfg.dsv41_engram_head_dim);
        defer t.close();
        total -|= t.shardBytes();
    }
    return total;
}

fn droppedBytes(gpa: std.mem.Allocator, dir_path: []const u8, dspark: bool) !u64 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = try std.Io.Dir.cwd().openDir(io, dir_path, .{});
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", a, .limited(64 << 20)) catch return 0;
    const idx = std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{}) catch return 0;
    const wm = (if (idx == .object) idx.object.get("weight_map") else null) orelse return 0;
    if (wm != .object) return 0;
    var files: std.StringArrayHashMapUnmanaged(void) = .empty;
    for (wm.object.values()) |v| if (v == .string) try files.put(a, v.string, {});
    var total: u64 = 0;
    for (files.keys()) |name| {
        const f = dir.openFile(io, name, .{}) catch continue;
        defer f.close(io);
        var rb: [8192]u8 = undefined;
        var rs = f.reader(io, &rb);
        const len = rs.interface.takeInt(u64, .little) catch continue;
        if (len > 64 << 20) continue;
        const hdr = try a.alloc(u8, @intCast(len));
        rs.interface.readSliceAll(hdr) catch continue;
        const h = std.json.parseFromSliceLeaky(std.json.Value, a, hdr, .{}) catch continue;
        if (h != .object) continue;
        var it = h.object.iterator();
        while (it.next()) |e| {
            const key = e.key_ptr.*;
            const bare = if (std.mem.startsWith(u8, key, model.dsv41_text_prefix)) key[model.dsv41_text_prefix.len..] else key;
            if (!model.dsv41VisionKey(bare) and (dspark or !std.mem.startsWith(u8, bare, "mtp."))) continue;
            const off = (if (e.value_ptr.* == .object) e.value_ptr.object.get("data_offsets") else null) orelse continue;
            if (off != .array or off.array.items.len != 2 or off.array.items[0] != .integer or off.array.items[1] != .integer) continue;
            total += @intCast(@max(off.array.items[1].integer - off.array.items[0].integer, 0));
        }
    }
    return total;
}

/// Engram for the pack in `dir`: the token map, the hash, and each layer's
/// table and projections. Without Engram layers in the config this is a no-op.
fn initEngram(m: *Dsv41Model, w: *const model.Weights, dir: []const u8) !void {
    const c = &m.cfg;
    if (c.dsv41_n_engram_layers == 0) return;
    m.token_map = try engram.loadTokenMap(m.gpa, dir, c.vocab_size, c.dsv41_engram_compressed_vocab);
    if (c.dsv41_engram_pad_id >= m.token_map.len) return error.EngramTokenMap;
    m.hash = try engram.Hash.init(c, m.token_map[c.dsv41_engram_pad_id]);
    const n = c.dsv41_n_engram_layers;
    m.engrams = try m.gpa.alloc(EngramLayer, n);
    var opened: usize = 0;
    errdefer {
        for (m.engrams[0..opened]) |*e| e.table.close();
        m.gpa.free(m.engrams);
        m.engrams = &.{};
    }
    const cols = m.hash.?.cols();
    for (0..n) |k| {
        const l: usize = c.dsv41_engram_layers[k];
        const qw = try getW(w, "layers.{d}.engram.q_weight", .{l});
        const kw = try getW(w, "layers.{d}.engram.k_weight", .{l});
        var prod = mlx.mlx_array_new();
        {
            var q32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(q32);
            var k32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(k32);
            try mlx.check(mlx.mlx_astype(&q32, qw, .float32, m.s));
            try mlx.check(mlx.mlx_astype(&k32, kw, .float32, m.s));
            try mlx.check(mlx.mlx_multiply(&prod, q32, k32, m.s));
        }
        _ = try m.own(prod);
        try mlx.check(mlx.mlx_array_eval(prod));
        m.engrams[k] = .{
            .layer = l,
            .wkv = try getLin(w, cols * c.dsv41_engram_head_dim, "layers.{d}.engram.wkv", .{l}),
            .qk = prod,
            .table = try engram.Table.open(m.gpa, dir, k, l, c.dsv41_engram_rows[k], c.dsv41_engram_head_dim),
        };
        opened += 1;
    }
}

// ── per-request state ────────────────────────────────────────────────────

/// Append-only rows on the GPU, growing by a quarter; a rollback is offset-only.
const Rows = struct {
    buf: A = none,
    used: usize = 0,
    cap: usize = 0,
    width: usize,
    dtype: mlx.mlx_dtype,

    fn deinit(self: *Rows) void {
        if (self.buf.ctx != null) _ = mlx.mlx_array_free(self.buf);
    }

    fn append(self: *Rows, x: A, s: S) !void {
        const n = dim(x, 0);
        if (n == 0) return;
        if (self.used + n > self.cap) {
            const cap = @max(self.used + n, @max(64, self.cap + self.cap / 4));
            var z = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ ci(cap), ci(self.width) }, 2, self.dtype, s));
            if (self.used > 0) {
                const old = try self.view(self.used, s);
                defer _ = mlx.mlx_array_free(old);
                const zeros = z;
                defer _ = mlx.mlx_array_free(zeros);
                var up = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_slice_update(&up, zeros, old, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(self.used), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
                z = up;
            }
            if (self.buf.ctx != null) _ = mlx.mlx_array_free(self.buf);
            self.buf = z;
            self.cap = cap;
        }
        var xc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xc);
        try mlx.check(mlx.mlx_astype(&xc, x, self.dtype, s));
        var up = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice_update(&up, self.buf, xc, &[_]c_int{ ci(self.used), 0 }, 2, &[_]c_int{ ci(self.used + n), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
        _ = mlx.mlx_array_free(self.buf);
        self.buf = up;
        self.used += n;
    }

    /// Rows [0, n) (caller frees).
    fn view(self: *const Rows, n: usize, s: S) !A {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&o, self.buf, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ ci(n), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
        return o;
    }
};

/// The window's rows by absolute position in a ring of window + slack rows.
const Ring = struct {
    buf: A,
    cap: usize,
    width: usize,

    fn init(cap: usize, width: usize, dt: mlx.mlx_dtype, s: S) !Ring {
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{ ci(cap), ci(width) }, 2, dt, s));
        return .{ .buf = z, .cap = cap, .width = width };
    }

    fn deinit(self: *Ring) void {
        _ = mlx.mlx_array_free(self.buf);
    }

    /// Rows for positions [p0, p0 + n); only the last `cap` can survive.
    fn write(self: *Ring, x: A, p0: usize, s: S) !void {
        const n = dim(x, 0);
        const keep_n = @min(n, self.cap);
        const first = p0 + n - keep_n;
        var done: usize = 0;
        while (done < keep_n) {
            const slot = (first + done) % self.cap;
            const run = @min(keep_n - done, self.cap - slot);
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, x, &[_]c_int{ ci(n - keep_n + done), 0 }, 2, &[_]c_int{ ci(n - keep_n + done + run), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            var cast = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cast);
            try mlx.check(mlx.mlx_astype(&cast, part, mlx.mlx_array_dtype(self.buf), s));
            var up = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice_update(&up, self.buf, cast, &[_]c_int{ ci(slot), 0 }, 2, &[_]c_int{ ci(slot + run), ci(self.width) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            _ = mlx.mlx_array_free(self.buf);
            self.buf = up;
            done += run;
        }
    }

    /// Rows for positions [lo, hi) in order; `hi - lo <= cap`.
    fn read(self: *const Ring, sc: *Scope, lo: usize, hi: usize) !A {
        std.debug.assert(hi - lo <= self.cap);
        var idx: [512]i32 = undefined;
        for (lo..hi, 0..) |p, k| idx[k] = @intCast(p % self.cap);
        return sc.take(self.buf, try sc.ints(idx[0 .. hi - lo], &.{ci(hi - lo)}), 0);
    }
};

const LState = struct {
    win: Ring,
    comp: ?Rows = null,
    idx_k: ?Rows = null,
    /// Ratio > 1 compressors: the open group's kv and gate rows, f32.
    pend_kv: A = none,
    pend_gate: A = none,

    fn deinit(self: *LState) void {
        self.win.deinit();
        if (self.comp) |*r| r.deinit();
        if (self.idx_k) |*r| r.deinit();
        if (self.pend_kv.ctx != null) _ = mlx.mlx_array_free(self.pend_kv);
        if (self.pend_gate.ctx != null) _ = mlx.mlx_array_free(self.pend_gate);
    }
};

/// What a verify chunk needs to keep only a prefix: each pooling
/// compressor's rows from its open group's first token on, and the chunk's
/// DSpark main-hidden rows (committed only once the accepted count is known).
const Verify = struct {
    n0: usize,
    comp_used: []usize,
    idx_used: []usize,
    pend_kv: []A,
    pend_gate: []A,
    mh: A = none,
};

pub const State = struct {
    gpa: std.mem.Allocator,
    n: usize = 0,
    layers: []LState,
    /// The request's compressed token ids (Engram look-back).
    hist: std.ArrayList(u32) = .empty,
    /// Per DSpark stage: the main-stream KV rows of committed positions.
    ds: []Ring = &.{},
    /// The last committed position's main hidden `[1, targets * dim]`.
    mh: A = none,
    verify: ?Verify = null,

    pub fn deinit(self: *State) void {
        for (self.layers) |*l| l.deinit();
        self.gpa.free(self.layers);
        for (self.ds) |*r| r.deinit();
        self.gpa.free(self.ds);
        self.hist.deinit(self.gpa);
        if (self.mh.ctx != null) _ = mlx.mlx_array_free(self.mh);
        self.dropVerify();
    }

    fn dropVerify(self: *State) void {
        const v = self.verify orelse return;
        for (v.pend_kv) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        for (v.pend_gate) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        if (v.mh.ctx != null) _ = mlx.mlx_array_free(v.mh);
        self.gpa.free(v.comp_used);
        self.gpa.free(v.idx_used);
        self.gpa.free(v.pend_kv);
        self.gpa.free(v.pend_gate);
        self.verify = null;
    }
};

pub fn initState(m: *const Dsv41Model, gpa: std.mem.Allocator) !State {
    const layers = try gpa.alloc(LState, m.n_layers);
    var made: usize = 0;
    errdefer {
        for (layers[0..made]) |*l| l.deinit();
        gpa.free(layers);
    }
    const hd = m.cfg.head_dim;
    const cap = m.cfg.sliding_window + RING_SLACK;
    for (layers, 0..) |*l, li| {
        l.* = .{ .win = try Ring.init(cap, hd, m.act, m.s) };
        made += 1;
        if (m.isKvSource(li)) l.comp = .{ .width = hd, .dtype = m.act };
        if (m.isOwner(li)) l.idx_k = .{ .width = m.cfg.dsv4_index_head_dim, .dtype = m.act };
    }
    var st: State = .{ .gpa = gpa, .layers = layers };
    if (m.n_mtp > 0) {
        st.ds = try gpa.alloc(Ring, m.n_mtp);
        for (st.ds, 0..) |*r, k| r.* = Ring.init(cap, hd, m.act, m.s) catch |e| {
            for (st.ds[0..k]) |*q| q.deinit();
            gpa.free(st.ds);
            return e;
        };
    }
    return st;
}

// ── numerics ──────────────────────────────────────────────────────────────

/// 2^ceil(log2(x)) for positive normal f32 `x`, from its bits (the kernel's
/// `fast_round_scale`); a log2/ceil pair lands a power below just above it.
fn pow2Ceil(sc: *Scope, x: A) !A {
    const bits = try sc.view(x, .uint32);
    const e = try sc.op2(mlx.mlx_bitwise_and, try sc.op2(mlx.mlx_right_shift, bits, try sc.u(23)), try sc.u(0xFF));
    const mant = try sc.op2(mlx.mlx_greater, try sc.op2(mlx.mlx_bitwise_and, bits, try sc.u(0x7FFFFF)), try sc.u(0));
    const up = try sc.add(e, try sc.astype(mant, .uint32));
    return sc.view(try sc.op2(mlx.mlx_left_shift, up, try sc.u(23)), .float32);
}

/// Round |v| <= 6 to e2m1 {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to even.
fn e2m1Round(sc: *Scope, v: A) !A {
    const mag = try sc.op1(mlx.mlx_abs, v);
    var idx = try sc.zeros(&.{}, .int32);
    for ([_]f32{ 0.25, 1.25, 2.5, 5.0 }) |t| idx = try sc.add(idx, try sc.astype(try sc.op2(mlx.mlx_greater, mag, try sc.f(t)), .int32));
    for ([_]f32{ 0.75, 1.75, 3.5 }) |t| idx = try sc.add(idx, try sc.astype(try sc.op2(mlx.mlx_greater_equal, mag, try sc.f(t)), .int32));
    const lut_v = [_]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    const lut = try sc.keep(mlx.mlx_array_new_data(&lut_v, &[_]c_int{8}, 1, .float32));
    return sc.mul(try sc.op1(mlx.mlx_sign, v), try sc.take(lut, idx, 0));
}

const Qat = enum { fp8_ue8m0, fp4_ue8m0, fp4_e4m3 };

/// The reference's in-place activation quant round-trips (`act_quant`,
/// `fp4_act_quant`) over blocks of the last axis, back in `x`'s dtype.
fn qat(sc: *Scope, m: *const Dsv41Model, x: A, comptime kind: Qat, blk_n: usize) !A {
    if (m.qat_off) return x;
    std.debug.assert(blk_n == qatBlock(kind));
    const f = m.qat_fns[@intFromEnum(kind)];
    if (f.ctx == null) return qatChain(sc, x, kind);
    const in = mlx.mlx_vector_array_new_data(&.{x}, 1);
    defer _ = mlx.mlx_vector_array_free(in);
    var out = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(out);
    try mlx.check(mlx.mlx_closure_apply(&out, f, in));
    var o = mlx.mlx_array_new();
    return sc.res(mlx.mlx_vector_array_get(&o, out, 0), &o);
}

fn qatBlock(kind: Qat) usize {
    return if (kind == .fp4_e4m3) 16 else 32;
}

/// `qat` as one compiled graph per kind: ~20 ops become a few fused kernels.
fn compileQat(m: *Dsv41Model) !void {
    inline for (comptime std.enums.values(Qat)) |kind| {
        const Body = struct {
            fn call(res: *mlx.mlx_vector_array, in: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
                const s: *const S = @ptrCast(@alignCast(payload.?));
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                mlx.check(mlx.mlx_vector_array_get(&x, in, 0)) catch return -1;
                var sc = Scope.init(std.heap.page_allocator, s.*);
                defer sc.deinit();
                const q = qatChain(&sc, x, kind) catch return -1;
                res.* = mlx.mlx_vector_array_new_data(&.{q}, 1);
                return 0;
            }
        };
        const raw = mlx.mlx_closure_new_func_payload(&Body.call, @constCast(@ptrCast(&m.s)), null);
        defer _ = mlx.mlx_closure_free(raw);
        try mlx.check(mlx.mlx_compile(&m.qat_fns[@intFromEnum(kind)], raw, false));
    }
}

fn qatChain(sc: *Scope, x: A, comptime kind: Qat) !A {
    const blk_n = comptime qatBlock(kind);
    const sh = mlx.getShape(x);
    var gshape: [8]c_int = undefined;
    @memcpy(gshape[0..sh.len], sh);
    gshape[sh.len - 1] = @divExact(sh[sh.len - 1], ci(blk_n));
    gshape[sh.len] = ci(blk_n);
    const xb = try sc.reshape(try sc.astype(x, .float32), gshape[0 .. sh.len + 1]);
    const amax_raw = try sc.max(try sc.op1(mlx.mlx_abs, xb), -1, true);
    const q = switch (kind) {
        .fp8_ue8m0 => blk: {
            const amax = try sc.op2(mlx.mlx_maximum, amax_raw, try sc.f(1e-4));
            const scale = try pow2Ceil(sc, try sc.mul(amax, try sc.f(1.0 / 448.0)));
            const v = try sc.clip(try sc.div(xb, scale), -448, 448);
            var o8 = mlx.mlx_array_new();
            const c8 = try sc.res(mlx.mlx_to_fp8(&o8, v, sc.s), &o8);
            var o32 = mlx.mlx_array_new();
            break :blk try sc.mul(try sc.res(mlx.mlx_from_fp8(&o32, c8, .float32, sc.s), &o32), scale);
        },
        .fp4_ue8m0 => blk: {
            const amax = try sc.op2(mlx.mlx_maximum, amax_raw, try sc.f(6.0 * std.math.pow(f32, 2, -126)));
            const scale = try pow2Ceil(sc, try sc.mul(amax, try sc.f(1.0 / 6.0)));
            break :blk try sc.mul(try e2m1Round(sc, try sc.clip(try sc.div(xb, scale), -6, 6)), scale);
        },
        .fp4_e4m3 => blk: {
            const amax = try sc.op2(mlx.mlx_maximum, amax_raw, try sc.f(6.0 * std.math.pow(f32, 2, -9)));
            var o8 = mlx.mlx_array_new();
            const c8 = try sc.res(mlx.mlx_to_fp8(&o8, try sc.div(amax, try sc.f(6)), sc.s), &o8);
            var o32 = mlx.mlx_array_new();
            const scale = try sc.res(mlx.mlx_from_fp8(&o32, c8, .float32, sc.s), &o32);
            break :blk try sc.mul(try e2m1Round(sc, try sc.clip(try sc.div(xb, scale), -6, 6)), scale);
        },
    };
    return sc.astype(try sc.reshape(q, sh), mlx.mlx_array_dtype(x));
}

/// Rope on the last `rd` channels of `x` `[..., L, D]`, positions
/// `offset + k * scale` along the second-to-last axis, adjacent pairs.
/// `offset`: the first row's position, an int32 scalar (an input of a compiled stretch).
fn ropeTail(sc: *Scope, m: *const Dsv41Model, x: A, periods: A, offset: A, scale: f32) !A {
    const rd = m.cfg.dsv4_rope_head_dim;
    const d = dim(x, -1);
    const head = try sc.slice(x, -1, 0, d - rd);
    const tail = try sc.astype(try sc.slice(x, -1, d - rd, d), .float32);
    // fast_rope wants a batch axis in front of [L, D].
    const flat = mlx.getShape(tail).len < 3;
    var o = mlx.mlx_array_new();
    var r = try sc.res(mlx.mlx_fast_rope_dynamic(&o, if (flat) try sc.expand(tail, 0) else tail, ci(rd), true, mlx.mlx_optional_float.none(), scale, offset, periods, sc.s), &o);
    if (flat) r = try sc.reshape(r, mlx.getShape(tail));
    return sc.concat(&.{ head, try sc.astype(r, mlx.mlx_array_dtype(x)) }, -1);
}

/// Per-head rope for `[L, H, D]` (positions on axis 0).
fn ropeHeads(sc: *Scope, m: *const Dsv41Model, x: A, periods: A, offset: A) !A {
    const t = try sc.transpose(x, &.{ 1, 0, 2 });
    return sc.transpose(try ropeTail(sc, m, t, periods, offset, 1.0), &.{ 1, 0, 2 });
}

fn ropeFor(m: *const Dsv41Model, li: usize) *const Rope {
    return if (m.ratio(li) > 0) &m.rope_yarn else &m.rope_plain;
}

const Split = struct { pre: A, post: A, comb: A };

/// `hc_mixes`: rms over the flattened stream, after the projection.
fn hcSplit(sc: *Scope, m: *const Dsv41Model, x: A, hc: *const Hc) !Split {
    const L = dim(x, 0);
    const flat = try sc.reshape(try sc.astype(x, .float32), &.{ ci(L), -1 });
    const ms = try sc.mean(try sc.op1(mlx.mlx_square, flat), -1, true);
    const rs = try sc.op1(mlx.mlx_rsqrt, try sc.add(ms, try sc.f(m.cfg.rms_norm_eps)));
    const mixes = try sc.mul(try sc.matmul(flat, try sc.t2(hc.fn_w)), rs);
    const z = try sc.add(try sc.mul(mixes, hc.scale), hc.base);
    const nh: c_int = @intCast(m.cfg.dsv4_hc_mult);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), nh }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), nh }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ ci(L), nh, nh }, 3, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, ci(L), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(ci(L), 32), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", nh));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ITERS", @intCast(m.cfg.dsv4_hc_sinkhorn_iters)));
    const eps_v = [1]f32{m.cfg.dsv4_hc_eps};
    const eps = try sc.keep(mlx.mlx_array_new_data(&eps_v, &[_]c_int{1}, 1, .float32));
    const inputs = [_]A{ z, eps };
    const iv = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, m.split_kernel, iv, cfg, sc.s));
    var outs: [3]A = undefined;
    for (&outs, 0..) |*o, k| {
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(o, ov, k));
        _ = try sc.keep(o.*);
    }
    return .{ .pre = outs[0], .post = outs[1], .comb = outs[2] };
}

/// Collapse the hc copies `[L, hc, D]` with `pre` `[L, hc]`, in f32.
fn hcCollapse(sc: *Scope, x: A, pre: A) !A {
    const w = try sc.expand(pre, 1); // [L, 1, hc]
    const y = try sc.matmul(w, try sc.astype(x, .float32)); // [L, 1, D]
    return sc.astype(try sc.reshape(y, &.{ ci(dim(x, 0)), ci(dim(x, 2)) }), mlx.mlx_array_dtype(x));
}

/// `out[k] = post[k] * y + sum_j comb[j, k] * x[j]` — the new stream.
fn hcExpand(sc: *Scope, x: A, y: A, sp: *const Split) !A {
    const L = dim(x, 0);
    const x4 = try sc.reshape(x, &.{ 1, ci(L), ci(dim(x, 1)), ci(dim(x, 2)) });
    const c: glm5.Collapsed = .{ .x = none, .post = sp.post, .comb = sp.comb };
    const o = try sc.keep(try glm5.expand(sc.s, try sc.expand(y, 0), x4, &c));
    return sc.reshape(o, mlx.getShape(x));
}

fn clampedSwiglu(sc: *Scope, gate: A, up: A, limit: f32) !A {
    var g = try sc.astype(gate, .float32);
    var u_ = try sc.astype(up, .float32);
    if (limit > 0) {
        g = try sc.op2(mlx.mlx_minimum, g, try sc.f(limit));
        u_ = try sc.clip(u_, -limit, limit);
    }
    return sc.mul(try sc.mul(g, try sc.sigmoid(g)), u_);
}

// ── attention ──────────────────────────────────────────────────────────────

/// What attention layers hand down within one span of one chunk.
const Shared = struct {
    topk: A = none, // [L, k] compressed indices, -1 = none
    cand: A = none, // [L, n_blocks] candidate blocks
};

/// `x` normed attention input `[L, dim]` at positions [p0, p0 + L).
fn attention(sc: *Scope, m: *Dsv41Model, st: *State, li: usize, x: A, p0: usize, sh: *Shared) !A {
    const c = &m.cfg;
    const ly = &m.layers[li];
    const L = dim(x, 0);
    const hd = c.head_dim;
    const ls = &st.layers[li];
    const pos = try sc.i(ci(p0));
    const compiled = compiledAt(m, L);

    var pj: Proj = undefined;
    if (compiled) {
        var a: [3]A = undefined;
        try runRegion(sc, m, li, .q_kv, &.{ x, pos }, &a);
        pj = .{ .q = a[0], .kv = a[1], .qr = a[2] };
    } else pj = try qkvProj(sc, m, ly, ropeFor(m, li).fwd, x, pos);
    const q = pj.q;
    const qr = pj.qr;
    const kv = pj.kv;

    const window = c.sliding_window;
    const prev_lo = if (p0 + 1 > window) p0 + 1 - window else 0;
    const prev = if (p0 > prev_lo) try ls.win.read(sc, prev_lo, p0) else none;
    const wp = p0 - prev_lo;
    const keys_w = if (prev.ctx != null) try sc.concat(&.{ prev, kv }, 0) else kv;
    try ls.win.write(kv, p0, sc.s);
    // Query i sees the window rows (wp + i - window, wp + i].
    const qpos = try sc.reshape(try sc.arange(wp, wp + L), &.{ ci(L), 1 });
    const kpos = try sc.reshape(try sc.arange(0, wp + L), &.{ 1, ci(wp + L) });
    const band = try sc.op2(mlx.mlx_logical_and, try sc.op2(mlx.mlx_less_equal, kpos, qpos), try sc.op2(mlx.mlx_greater, kpos, try sc.sub(qpos, try sc.i(@intCast(window)))));

    var comp_rows = none;
    var topk = none;
    if (ly.ratio > 0) {
        const r: usize = ly.ratio;
        const clen = (p0 + L) / r;
        if (ly.comp) |*cw| try compress(sc, m, st, li, cw, x, p0);
        if (ly.idx) |*iw| {
            sh.topk = if (clen == 0) none else try indexer(sc, m, st, li, iw, x, qr, p0, sh);
        }
        topk = sh.topk;
        if (clen > 0 and topk.ctx != null) {
            const src = m.latest(li, Dsv41Model.isKvSource);
            comp_rows = try sc.keep(try st.layers[src].comp.?.view(clen, sc.s));
        }
    }

    // Scores over the window band, then the selected compressed rows; the
    // per-head sink joins the softmax denominator only.
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));
    const qf = try sc.astype(q, .float32);
    const kw = try sc.astype(keys_w, .float32);
    var logits = try sc.matmul(qf, try sc.t2(kw)); // [L, H, Wk]
    var mask = try sc.expand(band, 1);
    var g = none;
    if (comp_rows.ctx != null) {
        const safe = try sc.op2(mlx.mlx_maximum, topk, try sc.i(0));
        g = try sc.astype(try sc.take(comp_rows, safe, 0), .float32); // [L, k, hd]
        const lc = try sc.matmul(qf, try sc.transpose(g, &.{ 0, 2, 1 })); // [L, H, k]
        logits = try sc.concat(&.{ logits, lc }, -1);
        mask = try sc.concat(&.{ mask, try sc.expand(try sc.op2(mlx.mlx_greater_equal, topk, try sc.i(0)), 1) }, -1);
    }
    logits = try sc.where(mask, try sc.mul(logits, try sc.f(scale)), try sc.f(-std.math.inf(f32)));
    const mx = try sc.op2(mlx.mlx_maximum, try sc.max(logits, -1, true), ly.sink);
    const p = try sc.op1(mlx.mlx_exp, try sc.sub(logits, mx));
    const denom = try sc.add(try sc.sum(p, -1, true), try sc.op1(mlx.mlx_exp, try sc.sub(ly.sink, mx)));
    const wk = dim(keys_w, 0);
    var o = try sc.matmul(try sc.slice(p, -1, 0, wk), kw);
    if (g.ctx != null) o = try sc.add(o, try sc.matmul(try sc.slice(p, -1, wk, dim(p, -1)), g));
    o = try sc.astype(try sc.div(o, denom), m.act); // [L, H, hd]
    if (m.trace) |t| {
        try t.put("sa_q_{d}", .{li}, q);
        try t.put("sa_kv_{d}", .{li}, keys_w);
        try t.put("sa_o_{d}", .{li}, o);
    }

    if (!compiled) return attnOut(sc, m, ly, ropeFor(m, li).inv, o, pos);
    var y: [1]A = undefined;
    try runRegion(sc, m, li, .attn_out, &.{ o, pos }, &y);
    return y[0];
}

const Proj = struct { q: A, kv: A, qr: A };

/// Rows at `pos` -> the roped query heads, the window row (roped, fp8
/// round-trip) and the query latent the indexer reads.
fn qkvProj(sc: *Scope, m: *Dsv41Model, ly: *const Layer, periods: A, x: A, pos: A) !Proj {
    const c = &m.cfg;
    const qr = try sc.rms(try ly.wq_a.apply(sc, x), ly.q_norm, c.rms_norm_eps);
    const q = try ropeHeads(sc, m, try sc.reshape(try ly.wq_b.apply(sc, qr), &.{ ci(dim(x, 0)), ci(c.num_attention_heads), ci(c.head_dim) }), periods, pos);
    return .{ .q = q, .kv = try kvRow(sc, m, ly, periods, x, pos), .qr = qr };
}

fn kvRow(sc: *Scope, m: *Dsv41Model, ly: *const Layer, periods: A, x: A, pos: A) !A {
    const kv = try sc.rms(try ly.wkv.apply(sc, x), ly.kv_norm, m.cfg.rms_norm_eps);
    return qat(sc, m, try ropeTail(sc, m, kv, periods, pos, 1.0), .fp8_ue8m0, 32);
}

/// Attention heads `o` [L, H, hd] at `pos` back to the stream: the inverse
/// rope, the grouped `wo_a`, then `wo_b`.
fn attnOut(sc: *Scope, m: *Dsv41Model, ly: *const Layer, inv: A, o: A, pos: A) !A {
    const c = &m.cfg;
    const L = dim(o, 0);
    const og = c.dsv4_o_groups;
    const r = try ropeHeads(sc, m, o, inv, pos);
    const ob = try sc.transpose(try sc.reshape(r, &.{ ci(L), ci(og), ci(c.num_attention_heads * c.head_dim / og) }), &.{ 1, 0, 2 }); // [og, L, in]
    const oa = if (ly.wo_a.s.ctx == null)
        try sc.matmul(ob, try sc.transpose(ly.wo_a.w, &.{ 0, 2, 1 }))
    else blk: {
        var ro = mlx.mlx_array_new();
        break :blk try sc.res(mlx.mlx_quantized_matmul(&ro, ob, ly.wo_a.w, ly.wo_a.s, ly.wo_a.b, true, mlx.mlx_optional_int.some(@intCast(ly.wo_a.qp.group_size)), mlx.mlx_optional_int.some(@intCast(ly.wo_a.qp.bits)), ly.wo_a.qp.mode.cstr(), sc.s), &ro);
    };
    return ly.wo_b.apply(sc, try sc.astype(try sc.reshape(try sc.transpose(oa, &.{ 1, 0, 2 }), &.{ ci(L), -1 }), m.act));
}

/// A KV source's compressor: pools each completed group of `ratio` tokens
/// into one latent (softmax gate in f32 above ratio 1), publishes index keys
/// from the latent before rope when this layer owns them, then ropes the
/// latent at its group's first position and stores it fp4-simulated.
fn compress(sc: *Scope, m: *Dsv41Model, st: *State, li: usize, cw: *const Comp, x: A, p0: usize) !void {
    const c = &m.cfg;
    const ly = &m.layers[li];
    const ls = &st.layers[li];
    const r: usize = ly.ratio;
    var latent = none;
    if (r == 1) {
        latent = try sc.rms(try cw.wkv.apply(sc, x), cw.norm, c.rms_norm_eps);
    } else {
        // Pooling runs in f32, weights included (the reference promotes them).
        const xf = try sc.astype(x, .float32);
        var kv = try sc.astype(try cw.wkv.apply(sc, xf), .float32);
        var gate = try sc.astype(try cw.wgate.?.apply(sc, xf), .float32);
        if (ls.pend_kv.ctx != null) {
            kv = try sc.concat(&.{ ls.pend_kv, kv }, 0);
            gate = try sc.concat(&.{ ls.pend_gate, gate }, 0);
        }
        if (st.verify) |*v| {
            v.pend_kv[li] = sc.out(kv);
            v.pend_gate[li] = sc.out(gate);
        }
        const total = dim(kv, 0);
        const g = total / r;
        const full = g * r;
        const new_kv = if (full < total) sc.out(try sc.slice(kv, 0, full, total)) else none;
        const new_gate = if (full < total) sc.out(try sc.slice(gate, 0, full, total)) else none;
        if (ls.pend_kv.ctx != null) _ = mlx.mlx_array_free(ls.pend_kv);
        if (ls.pend_gate.ctx != null) _ = mlx.mlx_array_free(ls.pend_gate);
        ls.pend_kv = new_kv;
        ls.pend_gate = new_gate;
        if (g == 0) return;
        const hd = c.head_dim;
        const kg = try sc.reshape(try sc.slice(kv, 0, 0, full), &.{ ci(g), ci(r), ci(hd) });
        const gg = try sc.reshape(try sc.slice(gate, 0, 0, full), &.{ ci(g), ci(r), ci(hd) });
        const pooled = try sc.sum(try sc.mul(kg, try sc.softmax(gg, 1)), 1, false);
        latent = try sc.rms(try sc.astype(pooled, m.act), cw.norm, c.rms_norm_eps);
    }
    const g0 = p0 / r;
    if (ly.idx) |*iw| if (iw.wk) |*wk| {
        var k = try sc.rms(try wk.apply(sc, latent), iw.k_norm, c.rms_norm_eps);
        k = try qat(sc, m, try ropeTail(sc, m, k, m.rope_yarn.fwd, try sc.i(ci(g0)), @floatFromInt(r)), .fp4_ue8m0, 32);
        try ls.idx_k.?.append(k, sc.s);
    };
    const rot = try qat(sc, m, try ropeTail(sc, m, latent, m.rope_yarn.fwd, try sc.i(ci(g0)), @floatFromInt(r)), .fp4_e4m3, 16);
    try ls.comp.?.append(rot, sc.s);
}

/// The top-`index_topk` compressed entries each query keeps (-1 = none),
/// scored against the latest index-key owner's keys. The candidate source
/// also publishes its pinned top blocks; later index sources score only
/// inside them.
fn indexer(sc: *Scope, m: *Dsv41Model, st: *State, li: usize, iw: *const Idx, x: A, qr: A, p0: usize, sh: *Shared) !A {
    const c = &m.cfg;
    const L = dim(x, 0);
    const r: usize = m.ratio(li);
    const nb = (p0 + L) / r;
    const ih = c.dsv4_index_n_heads;
    const ihd = c.dsv4_index_head_dim;
    const owner = m.latest(li, Dsv41Model.isOwner);
    const keys = try sc.astype(try sc.keep(try st.layers[owner].idx_k.?.view(nb, sc.s)), .float32);
    var q = try ropeHeads(sc, m, try sc.reshape(try iw.wq_b.apply(sc, qr), &.{ ci(L), ci(ih), ci(ihd) }), m.rope_yarn.fwd, try sc.i(ci(p0)));
    q = try sc.astype(try qat(sc, m, q, .fp4_ue8m0, 32), .float32);
    const wscale = 1.0 / @sqrt(@as(f32, @floatFromInt(ihd))) / @sqrt(@as(f32, @floatFromInt(ih)));
    const weights = try sc.mul(try sc.astype(try iw.weights_proj.apply(sc, x), .float32), try sc.f(wscale)); // [L, ih]
    // Per-head scores are ReLU'd before the weighted head sum, so the
    // [rows, ih, nb] f32 block is materialized: bound it by rows.
    const keys_t = try sc.t2(keys);
    const rows_per = @max(1, m.idx_block_bytes / (ih * @max(nb, 1) * 4));
    var parts: std.ArrayList(A) = .empty;
    defer parts.deinit(sc.a);
    var r0: usize = 0;
    while (r0 < L) : (r0 += rows_per) {
        const r1 = @min(L, r0 + rows_per);
        const raw = try sc.op2(mlx.mlx_maximum, try sc.matmul(try sc.slice(q, 0, r0, r1), keys_t), try sc.f(0)); // [r, ih, nb]
        try parts.append(sc.a, try sc.sum(try sc.mul(raw, try sc.expand(try sc.slice(weights, 0, r0, r1), -1)), 1, false));
    }
    var scores = try sc.concat(parts.items, 0); // [L, nb]
    // A query sees the groups that end at or before it.
    const lens = try sc.op2(mlx.mlx_divide, try sc.reshape(try sc.arange(p0 + 1, p0 + L + 1), &.{ ci(L), 1 }), try sc.i(@intCast(r)));
    const lens_i = try sc.astype(try sc.op1(mlx.mlx_floor, lens), .int32);
    const cols = try sc.reshape(try sc.arange(0, nb), &.{ 1, ci(nb) });
    const ninf = try sc.f(-std.math.inf(f32));
    scores = try sc.where(try sc.op2(mlx.mlx_less, cols, lens_i), scores, ninf);

    const cand_src = c.dsv41_candidate_source;
    const bs: usize = c.dsv41_candidate_block_size;
    if (cand_src >= 0 and li == @as(usize, @intCast(cand_src))) {
        const n_blocks = (nb + bs - 1) / bs;
        var padded = scores;
        if (n_blocks * bs > nb) padded = try sc.concat(&.{ scores, try sc.broadcast(ninf, &.{ ci(L), ci(n_blocks * bs - nb) }) }, -1);
        var blocks = try sc.max(try sc.reshape(padded, &.{ ci(L), ci(n_blocks), ci(bs) }), -1, false);
        // The block holding the query's newest group is only partly full: pin it.
        const last = try sc.op1(mlx.mlx_floor, try sc.div(try sc.astype(try sc.sub(lens_i, try sc.i(1)), .float32), try sc.f(@floatFromInt(bs))));
        const bidx = try sc.reshape(try sc.arange(0, n_blocks), &.{ 1, ci(n_blocks) });
        blocks = try sc.where(try sc.op2(mlx.mlx_equal, try sc.astype(bidx, .float32), last), try sc.f(std.math.inf(f32)), blocks);
        const kb = @min(@as(usize, c.dsv41_candidate_topk_blocks), n_blocks);
        const chosen = try sc.topk(blocks, kb);
        const reach = try sc.op2(mlx.mlx_greater, try sc.takeAlong(blocks, chosen, -1), ninf);
        var o = mlx.mlx_array_new();
        sh.cand = try sc.res(mlx.mlx_put_along_axis(&o, try sc.zeros(&.{ ci(L), ci(n_blocks) }, .bool_), chosen, reach, -1, sc.s), &o);
    } else if (cand_src >= 0 and li > @as(usize, @intCast(cand_src)) and sh.cand.ctx != null) {
        var o = mlx.mlx_array_new();
        const rep = try sc.res(mlx.mlx_repeat_axis(&o, sh.cand, ci(bs), -1, sc.s), &o);
        scores = try sc.where(try sc.slice(rep, -1, 0, nb), scores, ninf);
    }
    const k = @min(@as(usize, c.dsv4_index_topk), nb);
    const idx = try sc.astype(try sc.topk(scores, k), .int32);
    const picked = try sc.where(try sc.op2(mlx.mlx_less, idx, lens_i), idx, try sc.i(-1));
    if (m.trace) |t| {
        var o = mlx.mlx_array_new();
        try t.put("topk_{d}", .{li}, try sc.res(mlx.mlx_sort_axis(&o, picked, -1, sc.s), &o));
    }
    return picked;
}

// ── MoE ──────────────────────────────────────────────────────────────────

fn gatherMm(sc: *Scope, x: A, b: *const Bank, idx: A, sorted: bool) !A {
    var o = mlx.mlx_array_new();
    if (b.s.ctx == null) return sc.res(mlx.mlx_gather_mm(&o, x, try sc.transpose(b.w, &.{ 0, 2, 1 }), none, idx, sorted, sc.s), &o);
    return sc.res(mlx.mlx_gather_qmm(&o, x, b.w, b.s, b.b, none, idx, true, mlx.mlx_optional_int.some(@intCast(b.qp.group_size)), mlx.mlx_optional_int.some(@intCast(b.qp.bits)), b.qp.mode.cstr(), sorted, sc.s), &o);
}

fn routed(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, x: A, idx: A, wts: A) !A {
    const L = dim(x, 0);
    const k = dim(idx, 1);
    const d = dim(x, 1);
    const limit = m.cfg.dsv4_swiglu_limit;
    const e = &ly.experts;
    const n = L * k;
    const uidx = try sc.astype(idx, .uint32);
    if (n >= 64) {
        // Sorted by expert so consecutive rows stream one bank.
        const flat = try sc.reshape(uidx, &.{ci(n)});
        var oo = mlx.mlx_array_new();
        const order = try sc.res(mlx.mlx_argsort_axis(&oo, flat, 0, sc.s), &oo);
        var oi = mlx.mlx_array_new();
        const inv = try sc.res(mlx.mlx_argsort_axis(&oi, order, 0, sc.s), &oi);
        const rows = try sc.op2(mlx.mlx_floor_divide, order, try sc.u(@intCast(k)));
        const xs = try sc.expand(try sc.take(x, rows, 0), 1); // [n, 1, d]
        const sidx = try sc.take(flat, order, 0);
        const gt = try gatherMm(sc, xs, &e.gate, sidx, true);
        const up = try gatherMm(sc, xs, &e.up, sidx, true);
        const w_sorted = try sc.reshape(try sc.take(try sc.reshape(wts, &.{ci(n)}), order, 0), &.{ ci(n), 1, 1 });
        const act = try sc.astype(try sc.mul(try clampedSwiglu(sc, gt, up, limit), w_sorted), m.act);
        const dn = try gatherMm(sc, act, &e.down, sidx, true); // [n, 1, d]
        const unsorted = try sc.take(try sc.reshape(dn, &.{ ci(n), ci(d) }), inv, 0);
        return sc.sum(try sc.astype(try sc.reshape(unsorted, &.{ ci(L), ci(k), ci(d) }), .float32), 1, false);
    }
    const xe = try sc.reshape(x, &.{ ci(L), 1, 1, ci(d) });
    const gt = try gatherMm(sc, xe, &e.gate, uidx, false); // [L, k, 1, inter]
    const up = try gatherMm(sc, xe, &e.up, uidx, false);
    const act = try sc.astype(try sc.mul(try clampedSwiglu(sc, gt, up, limit), try sc.reshape(wts, &.{ ci(L), ci(k), 1, 1 })), m.act);
    const dn = try gatherMm(sc, act, &e.down, uidx, false); // [L, k, 1, d]
    return sc.sum(try sc.astype(try sc.reshape(dn, &.{ ci(L), ci(k), ci(d) }), .float32), 1, false);
}

const Routes = struct { idx: A, wts: A };

/// sqrt-softplus scores; the bias picks the top-k, the unbiased scores
/// (normalized, times the routed scale) weigh them.
fn route(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, x: A) !Routes {
    const logits = try sc.matmul(try sc.astype(x, .float32), try sc.t2(ly.gate_w));
    const scores = try sc.op1(mlx.mlx_sqrt, try sc.op2(mlx.mlx_logaddexp, logits, try sc.f(0)));
    const idx = try sc.topk(try sc.add(scores, ly.gate_bias), ly.top_k);
    var wts = try sc.takeAlong(scores, idx, -1);
    if (m.cfg.moe_route_norm and ly.top_k > 1) wts = try sc.div(wts, try sc.add(try sc.sum(wts, -1, true), try sc.f(1e-20)));
    return .{ .idx = idx, .wts = try sc.mul(wts, try sc.f(m.cfg.router_scaling_factor)) };
}

/// The routed sum `y` (f32) plus the shared expert.
fn withShared(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, x: A, y: A) !A {
    return addShared(sc, m, y, try shared(sc, m, ly, x));
}

fn shared(sc: *Scope, m: *const Dsv41Model, ly: *const Layer, x: A) !A {
    const sh = try clampedSwiglu(sc, try ly.shared[0].apply(sc, x), try ly.shared[1].apply(sc, x), m.cfg.dsv4_swiglu_limit);
    return ly.shared[2].apply(sc, try sc.astype(sh, m.act));
}

fn addShared(sc: *Scope, m: *const Dsv41Model, y: A, sy: A) !A {
    return sc.astype(try sc.add(y, try sc.astype(sy, .float32)), m.act);
}

fn moe(sc: *Scope, m: *const Dsv41Model, li: usize, x: A) !A {
    const ly = &m.layers[li];
    const r = try route(sc, m, ly, x);
    return withShared(sc, m, ly, x, try routed(sc, m, ly, x, r.idx, r.wts));
}

// ── Engram ──────────────────────────────────────────────────────────────

/// The gated write of the n-gram rows into the stream `[L, hc, dim]`: `wkv`
/// turns the rows into one key per copy plus a shared value, and the gate is
/// sigmoid of the signed sqrt of a per-copy normalized stream-key dot.
fn engramApply(sc: *Scope, m: *const Dsv41Model, e: *const EngramLayer, x: A, rows: A) !A {
    const c = &m.cfg;
    const L = dim(x, 0);
    const hc = c.dsv4_hc_mult;
    const d = c.hidden_size;
    const kv = try e.wkv.apply(sc, try sc.astype(rows, m.act));
    const key = try sc.reshape(try sc.astype(try sc.slice(kv, -1, 0, hc * d), .float32), &.{ ci(L), ci(hc), ci(d) });
    const value = try sc.astype(try sc.slice(kv, -1, hc * d, (hc + 1) * d), .float32);
    const h = try sc.astype(x, .float32);
    const eps = try sc.f(c.rms_norm_eps);
    const rs_h = try sc.op1(mlx.mlx_rsqrt, try sc.add(try sc.mean(try sc.op1(mlx.mlx_square, h), -1, false), eps));
    const rs_k = try sc.op1(mlx.mlx_rsqrt, try sc.add(try sc.mean(try sc.op1(mlx.mlx_square, key), -1, false), eps));
    const dot = try sc.mul(try sc.mul(try sc.sum(try sc.mul(try sc.mul(h, e.qk), key), -1, false), try sc.mul(rs_h, rs_k)), try sc.f(1.0 / @sqrt(@as(f32, @floatFromInt(d)))));
    const mag = try sc.op1(mlx.mlx_sqrt, try sc.op2(mlx.mlx_maximum, try sc.op1(mlx.mlx_abs, dot), try sc.f(1e-6)));
    const gate = try sc.sigmoid(try sc.where(try sc.op2(mlx.mlx_less, dot, try sc.f(0)), try sc.op1(mlx.mlx_negative, mag), mag));
    const out = try sc.add(h, try sc.mul(try sc.expand(gate, -1), try sc.expand(value, 1)));
    return sc.astype(out, mlx.mlx_array_dtype(x));
}

// ── the forward ──────────────────────────────────────────────────────────

const Want = enum { last, all };

/// One block over one span: the staggered hyper-connections around
/// attention and the MoE. Returns the new stream; `pre` becomes the FFN's
/// mix, which the next layer's attention collapses with.
fn block(sc: *Scope, m: *Dsv41Model, st: *State, li: usize, x: A, pre: *A, p0: usize, sh: *Shared, ds_main: ?A) !A {
    if (ds_main == null and li < m.regions.len and compiledAt(m, dim(x, 0))) {
        var a: [4]A = undefined; // the attention's pre, post, comb; its input
        try runRegion(sc, m, li, .attn_in, &.{ x, pre.* }, &a);
        const ao = try attention(sc, m, st, li, a[3], p0, sh);
        var o: [2]A = undefined;
        try runRegion(sc, m, li, .ffn, &.{ x, ao, a[0], a[1], a[2] }, &o);
        pre.* = o[1];
        return o[0];
    }
    const mid = try blockAttn(sc, m, st, li, x, pre.*, p0, sh, ds_main);
    const y = try moe(sc, m, li, mid.h);
    if (m.trace) |t| try t.put("ffn_{d}", .{li}, y);
    pre.* = mid.sf.pre;
    return hcExpand(sc, mid.x1, y, &mid.sf);
}

/// A block up to its FFN: the stream after attention, the FFN's mix and the
/// normed FFN input.
const Mid = struct { x1: A, sf: Split, h: A };

fn blockAttn(sc: *Scope, m: *Dsv41Model, st: *State, li: usize, x: A, pre: A, p0: usize, sh: *Shared, ds_main: ?A) !Mid {
    const ly = &m.layers[li];
    const eps = m.cfg.rms_norm_eps;
    const sa = try hcSplit(sc, m, x, &ly.hc_attn);
    var h = try sc.rms(try hcCollapse(sc, x, pre), ly.attn_norm, eps);
    if (m.trace) |t| {
        try t.put("hc_pre_{d}", .{li}, sa.pre);
        try t.put("hc_post_{d}", .{li}, sa.post);
        try t.put("hc_comb_{d}", .{li}, sa.comb);
        try t.put("attn_in_{d}", .{li}, h);
    }
    h = if (ds_main) |mk| try dsparkAttention(sc, m, li, h, mk, p0) else try attention(sc, m, st, li, h, p0, sh);
    if (m.trace) |t| try t.put("attn_{d}", .{li}, h);
    const x1 = try hcExpand(sc, x, h, &sa);
    const sf = try hcSplit(sc, m, x1, &ly.hc_ffn);
    return .{ .x1 = x1, .sf = sf, .h = try sc.rms(try hcCollapse(sc, x1, sa.pre), ly.ffn_norm, eps) };
}

fn onehotPre(sc: *Scope, m: *const Dsv41Model, L: usize) !A {
    const hc = m.cfg.dsv4_hc_mult;
    const first = try sc.op2(mlx.mlx_equal, try sc.arange(0, hc), try sc.i(0));
    return sc.broadcast(try sc.astype(first, .float32), &.{ ci(L), ci(hc) });
}

fn evalAll(xs: []const A) !void {
    const vec = mlx.mlx_vector_array_new_data(xs.ptr, xs.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// Advance `st` by `ids` and return the logits (f32, `[1, V]` for the last
/// position or `[L, V]` for every one; caller frees). Chunks wider than one
/// span run a layer at a time over all spans, evaluating per layer.
pub fn extend(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, ids: []const u32, want: Want) !A {
    const c = &m.cfg;
    const L = ids.len;
    const p0 = st.n;
    const hc = c.dsv4_hc_mult;
    const d = c.hidden_size;

    // Engram row ids for the whole chunk (host hash); the rows themselves are
    // read span by span at their layer.
    var eng_ids: []u32 = &.{};
    defer gpa.free(eng_ids);
    const ncols: usize = if (m.hash) |*hs| hs.cols() else 0;
    if (m.hash) |*hs| {
        const comp_ids = try gpa.alloc(u32, L);
        defer gpa.free(comp_ids);
        for (ids, comp_ids) |t, *o| o.* = if (t < m.token_map.len) m.token_map[t] else 0;
        eng_ids = try gpa.alloc(u32, L * hs.n_layers * ncols);
        hs.rows(st.hist.items, comp_ids, eng_ids);
        try st.hist.appendSlice(st.gpa, comp_ids);
    }

    // What crosses layers: each span's stream and mix, the picks attention
    // layers hand down, and the DSpark main rows.
    const span = m.span;
    const n_spans = (L + span - 1) / span;
    const targets = c.dsv4_n_dspark_target_layers;
    const n_mh = if (m.n_mtp > 0) n_spans * targets else 0;
    const held = try gpa.alloc(A, n_spans * 4 + n_mh);
    defer {
        for (held) |x| if (x.ctx != null) {
            _ = mlx.mlx_array_free(x);
        };
        gpa.free(held);
    }
    @memset(held, none);
    const xs = held[0..n_spans];
    const pres = held[n_spans .. 2 * n_spans];
    const topks = held[2 * n_spans .. 3 * n_spans];
    const cands = held[3 * n_spans .. 4 * n_spans];
    const mh = held[4 * n_spans ..];
    {
        var sc = Scope.init(gpa, m.s);
        defer sc.deinit();
        const emb = try m.embed.rows(&sc, try sc.ints(@ptrCast(ids), &.{ci(L)}), m.act);
        for (0..n_spans) |k| {
            const lo = k * span;
            const hi = @min(L, lo + span);
            hold(&xs[k], try sc.broadcast(try sc.expand(try sc.slice(emb, 0, lo, hi), 1), &.{ ci(hi - lo), ci(hc), ci(d) }));
            hold(&pres[k], try onehotPre(&sc, m, hi - lo));
        }
    }

    const sp: Spans = .{ .xs = xs, .pres = pres, .topks = topks, .cands = cands, .mh = mh, .span = span, .L = L, .p0 = p0, .eng_ids = eng_ids, .ncols = ncols };
    for (0..m.n_layers) |li| {
        for (0..n_spans) |k| {
            // A span's intermediates are dropped as soon as it is built, so the
            // layer's evaluation frees each one when its consumers are done.
            var sc = Scope.init(gpa, m.s);
            defer sc.deinit();
            var x = try spanIn(&sc, m, gpa, &sp, li, k);
            var sh: Shared = .{ .topk = topks[k], .cand = cands[k] };
            var pre = pres[k];
            x = try block(&sc, m, st, li, x, &pre, p0 + k * span, &sh, null);
            if (m.trace) |t| try t.put("stream_{d}", .{li}, x);
            hold(&xs[k], x);
            hold(&pres[k], pre);
            hold(&topks[k], sh.topk);
            hold(&cands[k], sh.cand);
        }
        if (n_spans > 1) {
            var ev: std.ArrayList(A) = .empty;
            defer ev.deinit(gpa);
            for (held) |x| if (x.ctx != null) try ev.append(gpa, x);
            try appendLayerState(gpa, &ev, &st.layers[li]);
            try evalAll(ev.items);
        }
    }
    st.n = p0 + L;

    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    if (m.n_mtp > 0) {
        var rows: std.ArrayList(A) = .empty;
        defer rows.deinit(gpa);
        for (0..n_spans) |k| try rows.append(gpa, try sc.concat(mh[k * targets ..][0..targets], -1));
        const all = try sc.concat(rows.items, 0); // [L, targets * dim]
        if (st.verify) |*v| {
            v.mh = sc.out(all);
        } else try commitMain(&sc, m, st, all, p0);
    }

    // The final collapse uses the last layer's FFN mix.
    var h: A = undefined;
    if (want == .last) {
        const k = n_spans - 1;
        const rows = dim(xs[k], 0);
        h = try hcCollapse(&sc, try sc.slice(xs[k], 0, rows - 1, rows), try sc.slice(pres[k], 0, rows - 1, rows));
    } else {
        var parts: std.ArrayList(A) = .empty;
        defer parts.deinit(gpa);
        for (0..n_spans) |k| try parts.append(gpa, try hcCollapse(&sc, xs[k], pres[k]));
        h = try sc.concat(parts.items, 0);
    }
    const logits = try sc.astype(try m.head.apply(&sc, try sc.rms(h, m.norm, c.rms_norm_eps)), .float32);
    var ev: std.ArrayList(A) = .empty;
    defer ev.deinit(gpa);
    try ev.append(gpa, logits);
    for (st.layers) |*ls| try appendLayerState(gpa, &ev, ls);
    for (st.ds) |r| try ev.append(gpa, r.buf);
    const vec = mlx.mlx_vector_array_new_data(ev.items.ptr, ev.items.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_async_eval(vec));
    return sc.out(logits);
}

/// What crosses layers in `extend`: each span's stream and mix, the picks
/// attention layers hand down, the DSpark main rows, and the Engram row ids.
const Spans = struct {
    xs: []A,
    pres: []A,
    topks: []A,
    cands: []A,
    mh: []A,
    span: usize,
    L: usize,
    p0: usize,
    eng_ids: []const u32,
    ncols: usize,

    fn bounds(sp: *const Spans, k: usize) [2]usize {
        const lo = k * sp.span;
        return .{ lo, @min(sp.L, lo + sp.span) };
    }
};

/// Span `k`'s stream entering layer `li`: its Engram write, and its DSpark
/// main row when the layer is a target.
fn spanIn(sc: *Scope, m: *Dsv41Model, gpa: std.mem.Allocator, sp: *const Spans, li: usize, k: usize) !A {
    const lo, const hi = sp.bounds(k);
    var x = sp.xs[k];
    for (m.engrams, 0..) |*e, ek| {
        if (e.layer != li) continue;
        const nc = sp.ncols;
        const mine = try gpa.alloc(u32, (hi - lo) * nc);
        defer gpa.free(mine);
        for (lo..hi) |t| @memcpy(mine[(t - lo) * nc ..][0..nc], sp.eng_ids[(t * m.engrams.len + ek) * nc ..][0..nc]);
        const rows = try sc.keep(try e.table.gather(gpa, mine, sc.s));
        x = try engramApply(sc, m, e, x, try sc.reshape(rows, &.{ ci(hi - lo), -1 }));
        if (m.trace) |t| try t.put("engram_{d}", .{li}, x);
    }
    if (m.n_mtp > 0) if (m.dsparkTarget(li)) |t| {
        hold(&sp.mh[k * m.cfg.dsv4_n_dspark_target_layers + t], try sc.astype(try sc.mean(try sc.astype(x, .float32), 1, false), m.act));
    };
    return x;
}

// ── compiled decode stretches ────────────────────────────────────────────

/// Blocks of at most this many rows (decode and DSpark verify) run the
/// stretches around attention's state as compiled graphs, one per layer: what
/// precedes attention, its projections, and the FFN half of the block.
const region_rows = 8;

const Region = enum { attn_in, q_kv, attn_out, ffn };

const LayerRegions = struct {
    m: *Dsv41Model,
    li: usize,
    fns: [std.enums.values(Region).len]mlx.mlx_closure = @splat(.{}),
};

fn compiledAt(m: *const Dsv41Model, rows: usize) bool {
    return rows <= region_rows and m.regions.len > 0 and m.trace == null;
}

fn compileRegions(m: *Dsv41Model) !void {
    m.regions = try m.gpa.alloc(LayerRegions, m.n_layers);
    for (m.regions, 0..) |*r, li| r.* = .{ .m = m, .li = li };
    for (m.regions) |*r| inline for (comptime std.enums.values(Region)) |kind| {
        const raw = mlx.mlx_closure_new_func_payload(&RegionBody(kind).call, @ptrCast(r), null);
        defer _ = mlx.mlx_closure_free(raw);
        try mlx.check(mlx.mlx_compile(&r.fns[@intFromEnum(kind)], raw, false));
    };
}

fn freeRegions(m: *Dsv41Model) void {
    for (m.regions) |r| for (r.fns) |f| if (f.ctx != null) {
        _ = mlx.mlx_closure_free(f);
    };
    m.gpa.free(m.regions);
    m.regions = &.{};
}

fn runRegion(sc: *Scope, m: *const Dsv41Model, li: usize, comptime kind: Region, args: []const A, out: []A) !void {
    const in = mlx.mlx_vector_array_new_data(args.ptr, args.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var res = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(res);
    try mlx.check(mlx.mlx_closure_apply(&res, m.regions[li].fns[@intFromEnum(kind)], in));
    for (out, 0..) |*o, i| {
        var x = mlx.mlx_array_new();
        o.* = try sc.res(mlx.mlx_vector_array_get(&x, res, i), &x);
    }
}

fn RegionBody(comptime kind: Region) type {
    return struct {
        fn call(res: *mlx.mlx_vector_array, in: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
            const r: *const LayerRegions = @ptrCast(@alignCast(payload.?));
            run(r.m, r.li, res, in) catch return -1;
            return 0;
        }

        fn run(m: *Dsv41Model, li: usize, res: *mlx.mlx_vector_array, in: mlx.mlx_vector_array) !void {
            const ly = &m.layers[li];
            const eps = m.cfg.rms_norm_eps;
            var a: [8]A = undefined;
            const n = mlx.mlx_vector_array_size(in);
            for (a[0..n], 0..) |*x, i| {
                x.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_vector_array_get(x, in, i));
            }
            defer for (a[0..n]) |x| {
                _ = mlx.mlx_array_free(x);
            };
            var sc = Scope.init(std.heap.page_allocator, m.s);
            defer sc.deinit();
            switch (kind) {
                // x, the previous FFN mix -> the attention's mix and its normed input
                .attn_in => {
                    const sa = try hcSplit(&sc, m, a[0], &ly.hc_attn);
                    const h = try sc.rms(try hcCollapse(&sc, a[0], a[1]), ly.attn_norm, eps);
                    res.* = mlx.mlx_vector_array_new_data(&.{ sa.pre, sa.post, sa.comb, h }, 4);
                },
                // the attention's input and position -> its query, window row and query latent
                .q_kv => {
                    const pj = try qkvProj(&sc, m, ly, ropeFor(m, li).fwd, a[0], a[1]);
                    res.* = mlx.mlx_vector_array_new_data(&.{ pj.q, pj.kv, pj.qr }, 3);
                },
                // the attention heads and position -> the attention's output
                .attn_out => res.* = mlx.mlx_vector_array_new_data(&.{try attnOut(&sc, m, ly, ropeFor(m, li).inv, a[0], a[1])}, 1),
                // x, the attention's output and mix -> the new stream and the FFN's pre
                .ffn => {
                    const sa: Split = .{ .pre = a[2], .post = a[3], .comb = a[4] };
                    const x1 = try hcExpand(&sc, a[0], a[1], &sa);
                    const sf = try hcSplit(&sc, m, x1, &ly.hc_ffn);
                    const hh = try sc.rms(try hcCollapse(&sc, x1, a[2]), ly.ffn_norm, eps);
                    res.* = mlx.mlx_vector_array_new_data(&.{ try hcExpand(&sc, x1, try moe(&sc, m, li, hh), &sf), sf.pre }, 2);
                },
            }
        }
    };
}

/// Point `slot` at `x` (a new handle), releasing what it held.
fn hold(slot: *A, x: A) void {
    if (slot.ctx == x.ctx) return;
    var o = mlx.mlx_array_new();
    if (x.ctx != null) _ = mlx.mlx_array_set(&o, x) else {
        _ = mlx.mlx_array_free(o);
        o = none;
    }
    if (slot.ctx != null) _ = mlx.mlx_array_free(slot.*);
    slot.* = o;
}

fn appendLayerState(gpa: std.mem.Allocator, ev: *std.ArrayList(A), ls: *const LState) !void {
    try ev.append(gpa, ls.win.buf);
    if (ls.comp) |r| if (r.buf.ctx != null) try ev.append(gpa, r.buf);
    if (ls.idx_k) |r| if (r.buf.ctx != null) try ev.append(gpa, r.buf);
    if (ls.pend_kv.ctx != null) try ev.append(gpa, ls.pend_kv);
    if (ls.pend_gate.ctx != null) try ev.append(gpa, ls.pend_gate);
}

/// A fresh request: drop the old state and prefill `ids`.
pub fn prefill(m: *Dsv41Model, gpa: std.mem.Allocator, ids: []const u32) !A {
    if (m.dec_state) |*st| st.deinit();
    m.dec_state = null;
    m.state_gen += 1;
    m.dec_state = try initState(m, gpa);
    return extendResumable(m, gpa, ids);
}

// ── prefix resume: one conversation ──────────────────────────────────────

/// What a resume restores: every buffer a later step replaces rather than
/// appends to is held by reference (a ring's first write after copies it
/// once), and append-only rows by their count.
const Snap = struct {
    gen: u64,
    ids: []u32,
    layers: []LSnap,
    ds: []A,
    mh: A,
    hist: []u32,

    const LSnap = struct { win: A, comp_used: usize, idx_used: usize, pend_kv: A, pend_gate: A };

    fn deinit(self: *Snap, gpa: std.mem.Allocator) void {
        for (self.layers) |l| for ([_]A{ l.win, l.pend_kv, l.pend_gate }) |a| if (a.ctx != null) {
            _ = mlx.mlx_array_free(a);
        };
        for (self.ds) |a| _ = mlx.mlx_array_free(a);
        if (self.mh.ctx != null) _ = mlx.mlx_array_free(self.mh);
        gpa.free(self.layers);
        gpa.free(self.ds);
        gpa.free(self.hist);
        gpa.free(self.ids);
    }
};

fn ref(x: A) A {
    var o = mlx.mlx_array_new();
    if (x.ctx == null) {
        _ = mlx.mlx_array_free(o);
        return none;
    }
    _ = mlx.mlx_array_set(&o, x);
    return o;
}

/// The state `prompt`'s run reaches one token short of its end is kept for
/// the next request (`snapshotIfDue`); restores the last one when it is a
/// prefix of `prompt` and returns its length (0: a cold prompt). A restore
/// always leaves a token to forward.
pub fn resumePrompt(m: *Dsv41Model, prompt: []const u32) !usize {
    const n = try restore(m, prompt);
    m.gpa.free(m.snap_ids);
    m.snap_ids = &.{};
    m.snap_at = null;
    if (prompt.len >= 2) {
        m.snap_ids = try m.gpa.dupe(u32, prompt[0 .. prompt.len - 1]);
        m.snap_at = prompt.len - 1;
    }
    return n;
}

fn restore(m: *Dsv41Model, prompt: []const u32) !usize {
    const sn = &(m.snap orelse return 0);
    const st = &(m.dec_state orelse return 0);
    if (sn.gen != m.state_gen or sn.ids.len >= prompt.len or !std.mem.eql(u32, sn.ids, prompt[0..sn.ids.len])) return 0;
    st.dropVerify();
    for (st.layers, sn.layers) |*l, s| {
        hold(&l.win.buf, s.win);
        if (l.comp) |*r| r.used = s.comp_used;
        if (l.idx_k) |*r| r.used = s.idx_used;
        hold(&l.pend_kv, s.pend_kv);
        hold(&l.pend_gate, s.pend_gate);
    }
    for (st.ds, sn.ds) |*r, b| hold(&r.buf, b);
    hold(&st.mh, sn.mh);
    st.hist.clearRetainingCapacity();
    try st.hist.appendSlice(st.gpa, sn.hist);
    st.n = sn.ids.len;
    return st.n;
}

/// Takes the armed snapshot once the state stands at its position.
fn snapshotIfDue(m: *Dsv41Model) !void {
    const at = m.snap_at orelse return;
    const st = &(m.dec_state orelse return);
    if (st.n != at) return;
    m.snap_at = null;
    if (m.snap) |*old| old.deinit(m.gpa);
    m.snap = null;
    const layers = try m.gpa.alloc(Snap.LSnap, st.layers.len);
    errdefer m.gpa.free(layers);
    const ds = try m.gpa.alloc(A, st.ds.len);
    errdefer m.gpa.free(ds);
    const hist = try m.gpa.dupe(u32, st.hist.items);
    for (layers, st.layers) |*o, *l| o.* = .{
        .win = ref(l.win.buf),
        .comp_used = if (l.comp) |r| r.used else 0,
        .idx_used = if (l.idx_k) |r| r.used else 0,
        .pend_kv = ref(l.pend_kv),
        .pend_gate = ref(l.pend_gate),
    };
    for (ds, st.ds) |*o, r| o.* = ref(r.buf);
    m.snap = .{ .gen = m.state_gen, .ids = m.snap_ids, .layers = layers, .ds = ds, .mh = ref(st.mh), .hist = hist };
    m.snap_ids = &.{};
}

/// `extend` that takes the armed snapshot on the way: a chunk crossing its
/// position runs as two.
pub fn extendResumable(m: *Dsv41Model, gpa: std.mem.Allocator, ids: []const u32) !A {
    try snapshotIfDue(m);
    const st = &m.dec_state.?;
    if (m.snap_at) |at| if (at > st.n and at < st.n + ids.len) {
        const k = at - st.n;
        const head = try extend(m, gpa, st, ids[0..k], .last);
        _ = mlx.mlx_array_free(head);
        try snapshotIfDue(m);
        return extend(m, gpa, st, ids[k..], .last);
    };
    const out = try extend(m, gpa, st, ids, .last);
    try snapshotIfDue(m);
    return out;
}

// ── DSpark ──────────────────────────────────────────────────────────────

/// Each stage's window rows for committed positions [p0, p0 + n): the main
/// stream `main_norm(main_proj(hidden))` through the stage's own kv path.
fn commitMain(sc: *Scope, m: *Dsv41Model, st: *State, mh: A, p0: usize) !void {
    const ds = &(m.dspark orelse return);
    const n = dim(mh, 0);
    if (st.mh.ctx != null) _ = mlx.mlx_array_free(st.mh);
    st.mh = sc.out(try sc.slice(mh, 0, n - 1, n));
    const main_x = try sc.rms(try ds.main_proj.apply(sc, mh), ds.main_norm, m.cfg.rms_norm_eps);
    for (st.ds, 0..) |*ring, si| {
        const kv = try kvRow(sc, m, &m.layers[m.n_layers + si], m.rope_plain.fwd, main_x, try sc.i(ci(p0)));
        try ring.write(kv, p0, sc.s);
    }
}

/// A stage's attention for the draft rows at [p0, p0 + B): every row sees
/// the stage's window of committed main rows and every draft row.
fn dsparkAttention(sc: *Scope, m: *Dsv41Model, li: usize, x: A, main_keys: A, p0: usize) !A {
    const ly = &m.layers[li];
    const hd = m.cfg.head_dim;
    const pos = try sc.i(ci(p0));
    const pj = try qkvProj(sc, m, ly, m.rope_plain.fwd, x, pos);
    const q = pj.q;
    const keys = try sc.astype(try sc.concat(&.{ main_keys, pj.kv }, 0), .float32);
    const qf = try sc.astype(q, .float32);
    const logits = try sc.mul(try sc.matmul(qf, try sc.t2(keys)), try sc.f(1.0 / @sqrt(@as(f32, @floatFromInt(hd)))));
    const mx = try sc.op2(mlx.mlx_maximum, try sc.max(logits, -1, true), ly.sink);
    const p = try sc.op1(mlx.mlx_exp, try sc.sub(logits, mx));
    const denom = try sc.add(try sc.sum(p, -1, true), try sc.op1(mlx.mlx_exp, try sc.sub(ly.sink, mx)));
    const o = try sc.astype(try sc.div(try sc.matmul(p, keys), denom), m.act);
    return attnOut(sc, m, ly, m.rope_plain.inv, o, pos);
}

pub const DsparkDraft = dsv4.DsparkDraft;
pub const DsparkRound = dsv4.DsparkRound;
pub const DsparkPhases = dsv4.DsparkPhases;

/// The draft for the block at [n, n + B): `[t1, noise…]` through the stages,
/// the last stage's collapse, the trunk head over its norm, then the Markov
/// bigram bias chained position by position on the GPU (greedy). The
/// confidence head truncates the block only when a threshold is set.
pub fn dsparkDraft(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32) !DsparkDraft {
    return draftRows(m, gpa, st, t1, false);
}

/// `keep_logits` also copies the Markov-biased rows to the host.
fn draftRows(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32, keep_logits: bool) !DsparkDraft {
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    const c = &m.cfg;
    const B = m.ds_block;
    const n = st.n;
    const hc = c.dsv4_hc_mult;
    const d = c.hidden_size;
    const ds = &m.dspark.?;
    var ids: [16]i32 = undefined;
    for (0..B) |k| ids[k] = @intCast(if (k == 0) t1 else c.dsv4_dspark_noise_token_id);
    const emb = try m.embed.rows(&sc, try sc.ints(ids[0..B], &.{ci(B)}), m.act);
    var x = try sc.broadcast(try sc.expand(emb, 1), &.{ ci(B), ci(hc), ci(d) });
    var pre = try onehotPre(&sc, m, B);
    const window = @min(c.sliding_window, n);
    var dummy: Shared = .{};
    for (0..m.n_mtp) |si| {
        const keys = try st.ds[si].read(&sc, n - window, n);
        x = try block(&sc, m, st, m.n_layers + si, x, &pre, n, &dummy, keys);
    }
    const hout = try hcCollapse(&sc, x, pre);
    const logits = try sc.astype(try m.head.apply(&sc, try sc.rms(hout, ds.norm, c.rms_norm_eps)), .float32);
    var cur = try sc.ints(ids[0..1], &.{1});
    var toks: [16]A = undefined;
    var confs: [16]A = undefined;
    var rows_k: [16]A = undefined;
    const want_conf = std.math.isFinite(m.ds_conf_thr);
    for (0..B) |k| {
        const me = try sc.astype(try sc.take(ds.markov_embed, cur, 0), .float32); // [1, rank]
        const row = try sc.add(try sc.slice(logits, 0, k, k + 1), try sc.matmul(me, try sc.t2(ds.markov_head)));
        rows_k[k] = row;
        if (want_conf) confs[k] = try sc.matmul(try sc.concat(&.{ try sc.astype(try sc.slice(hout, 0, k, k + 1), .float32), me }, -1), try sc.t2(ds.conf));
        cur = try sc.astype(try sc.argmax(row, -1), .int32);
        toks[k] = cur;
    }
    const all = try sc.concat(toks[0..B], 0);
    const rows_all = if (keep_logits) try sc.concat(rows_k[0..B], 0) else all;
    var ev = [_]A{ all, if (want_conf) try sc.concat(confs[0..B], 0) else all, rows_all };
    try evalAll(&ev);
    const out_ids = try gpa.alloc(u32, B + 1);
    errdefer gpa.free(out_ids);
    out_ids[0] = t1;
    const data = mlx.mlx_array_data_int32(all) orelse return error.NoData;
    for (0..B) |k| out_ids[k + 1] = @intCast(data[k]);
    const conf = try gpa.alloc(f32, B);
    errdefer gpa.free(conf);
    var len = B;
    if (want_conf) {
        const cd = mlx.mlx_array_data_float32(try sc.astype(ev[1], .float32)) orelse return error.NoData;
        @memcpy(conf, cd[0..B]);
        for (conf, 0..) |v, k| {
            if (v < m.ds_conf_thr) {
                len = k;
                break;
            }
        }
    } else @memset(conf, 0);
    const host_logits = try gpa.alloc(f32, if (keep_logits) B * m.vocab else 0);
    if (keep_logits) @memcpy(host_logits, (mlx.mlx_array_data_float32(rows_all) orelse return error.NoData)[0 .. B * m.vocab]);
    return .{ .ids = out_ids, .len = len, .logits = host_logits, .confidence = conf };
}

pub const DsparkPending = struct {
    vl_g: A,
    b: usize,
    verify: [16]u32,
    phases: DsparkPhases = .{},

    pub fn lapVerify(_: *DsparkPending, _: *const Dsv41Model) void {}

    pub fn deinit(self: *DsparkPending) void {
        _ = mlx.mlx_array_free(self.vl_g);
    }
};

pub fn dsparkBegin(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32) !DsparkPending {
    var d = try dsparkDraft(m, gpa, st, t1);
    defer d.deinit(gpa);
    return dsparkBeginWith(m, gpa, st, t1, &d);
}

/// Arm the rollback anchors and append `[t1, drafts…]` as a verify chunk;
/// the returned `vl_g` holds every position's logits, lazily.
pub fn dsparkBeginWith(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32, d: *const DsparkDraft) !DsparkPending {
    const b = d.len;
    const nl = m.n_layers;
    var v: Verify = .{
        .n0 = st.n,
        .comp_used = try st.gpa.alloc(usize, nl),
        .idx_used = try st.gpa.alloc(usize, nl),
        .pend_kv = try st.gpa.alloc(A, nl),
        .pend_gate = try st.gpa.alloc(A, nl),
    };
    for (st.layers, 0..) |*ls, li| {
        v.comp_used[li] = if (ls.comp) |r| r.used else 0;
        v.idx_used[li] = if (ls.idx_k) |r| r.used else 0;
        v.pend_kv[li] = none;
        v.pend_gate[li] = none;
    }
    st.verify = v;
    var ids: [16]u32 = undefined;
    ids[0] = t1;
    @memcpy(ids[1 .. b + 1], d.ids[1 .. b + 1]);
    const vl = extend(m, gpa, st, ids[0 .. b + 1], .all) catch |e| {
        st.dropVerify();
        return e;
    };
    return .{ .vl_g = vl, .b = b, .verify = ids };
}

/// Keep `accepted + 1` of the verify chunk's positions: offsets roll back,
/// each pooling compressor's open group is re-cut from the chunk's rows, and
/// the kept positions' main rows go to the DSpark windows.
pub fn dsparkFinish(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, pending: *DsparkPending, accepted: usize, next_token: u32) !DsparkRound {
    var sc = Scope.init(gpa, m.s);
    defer sc.deinit();
    defer st.dropVerify();
    const v = &st.verify.?;
    const keep_n = accepted + 1;
    const end = v.n0 + keep_n;
    if (keep_n < pending.b + 1) {
        for (st.layers, 0..) |*ls, li| {
            const r = m.ratio(li);
            if (ls.comp) |*rows| rows.used = v.comp_used[li] + (end / r - v.n0 / r);
            if (ls.idx_k) |*rows| rows.used = v.idx_used[li] + (end / r - v.n0 / r);
            if (r > 1 and v.pend_kv[li].ctx != null) {
                const first = v.n0 - v.n0 % r; // position of pend_kv[li] row 0
                const lo = end - end % r - first;
                const hi = end - first;
                if (ls.pend_kv.ctx != null) _ = mlx.mlx_array_free(ls.pend_kv);
                if (ls.pend_gate.ctx != null) _ = mlx.mlx_array_free(ls.pend_gate);
                ls.pend_kv = if (hi > lo) sc.out(try sc.slice(v.pend_kv[li], 0, lo, hi)) else none;
                ls.pend_gate = if (hi > lo) sc.out(try sc.slice(v.pend_gate[li], 0, lo, hi)) else none;
            }
        }
        st.n = end;
        st.hist.shrinkRetainingCapacity(end);
    }
    if (v.mh.ctx != null) try commitMain(&sc, m, st, try sc.slice(v.mh, 0, 0, keep_n), v.n0);
    const tokens = try gpa.alloc(u32, keep_n);
    @memcpy(tokens, pending.verify[0..keep_n]);
    var ph = pending.phases;
    ph.accepted = @intCast(accepted);
    ph.committed = @intCast(keep_n);
    ph.submitted = @intCast(pending.b);
    return .{ .tokens = tokens, .next_token = next_token, .accepted = @intCast(accepted), .phases = ph };
}

pub fn dsparkObserve(m: *Dsv41Model, ph: DsparkPhases) void {
    if (m.ds_prof) |*p| {
        p.observe(ph);
        if (p.rounds % 16 == 0) p.report();
    }
}

/// One greedy round: draft, verify, keep the longest prefix the trunk's own
/// argmax agrees with, and the trunk's token after it.
pub fn dsparkRound(m: *Dsv41Model, gpa: std.mem.Allocator, st: *State, t1: u32, accepted_cap: usize) !DsparkRound {
    var d = try dsparkDraft(m, gpa, st, t1);
    defer d.deinit(gpa);
    var pending = try dsparkBeginWith(m, gpa, st, t1, &d);
    defer pending.deinit();
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, pending.vl_g, -1, false, m.s));
    var am32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am32);
    try mlx.check(mlx.mlx_astype(&am32, am, .int32, m.s));
    try mlx.check(mlx.mlx_array_eval(am32));
    const best = mlx.mlx_array_data_int32(am32) orelse return error.NoData;
    var accepted: usize = 0;
    while (accepted < pending.b and @as(u32, @intCast(best[accepted])) == d.ids[accepted + 1]) accepted += 1;
    accepted = @min(accepted, accepted_cap);
    return dsparkFinish(m, gpa, st, &pending, accepted, @intCast(best[accepted]));
}

// ── tests ────────────────────────────────────────────────────────────────

const testing = std.testing;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// A tiny pack from tests/dump_dsv41_fixtures.py, its reference outputs,
/// and the model in f32. MLX runs f32 GEMMs as TF32 on M5 unless told not
/// to, and reads the switch once, at the process's first matmul: a full-suite
/// run sets `MLX_ENABLE_TF32=0` itself. The reference is plain f32.
const Tiny = struct {
    cfg: ModelConfig,
    w: model.Weights,
    fx: model.Weights,
    m: *Dsv41Model,

    fn open(root: []const u8, pack: []const u8, fixture: []const u8) !Tiny {
        return openAct(root, pack, fixture, .float32);
    }

    fn openAct(root: []const u8, pack: []const u8, fixture: []const u8, act: mlx.mlx_dtype) !Tiny {
        _ = setenv("MLX_ENABLE_TF32", "0", 1);
        const gpa = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const dir = try std.fmt.allocPrint(gpa, "{s}/{s}", .{ root, pack });
        defer gpa.free(dir);
        var cfg = try model.parseConfig(io, gpa, dir);
        errdefer cfg.deinit(gpa);
        var w = try model.loadModelWeights(io, gpa, dir, &cfg, false);
        errdefer w.deinit();
        var fx = try model.loadWeightsFile(gpa, dir, fixture);
        errdefer fx.deinit();
        const m = try init(gpa, &cfg, &w, mlx.gpuStream(), .{ .act = act, .dspark = true });
        return .{ .cfg = cfg, .w = w, .fx = fx, .m = m };
    }

    fn deinit(self: *Tiny) void {
        self.m.deinit();
        self.fx.deinit();
        self.w.deinit();
        self.cfg.deinit(testing.allocator);
    }

    fn host(self: *const Tiny, name: []const u8) ![]f32 {
        return hostF32(self.fx.get(name) orelse return error.MissingFixture);
    }

    fn ids(self: *const Tiny) ![]u32 {
        const a = self.fx.get("ids") orelse return error.MissingFixture;
        try mlx.check(mlx.mlx_array_eval(a));
        const d = mlx.mlx_array_data_int32(a) orelse return error.NoData;
        const out = try testing.allocator.alloc(u32, mlx.mlx_array_size(a));
        for (out, 0..) |*o, k| o.* = @intCast(d[k]);
        return out;
    }
};

fn hostF32(a: A) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, mlx.gpuStream()));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, c, .float32, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return testing.allocator.dupe(f32, d[0..mlx.mlx_array_size(f)]);
}

/// Rows of `got` against `want` (`rows` x `width`): every row's cosine and
/// norm ratio, and the largest error against the reference's peak.
fn expectRowsClose(label: []const u8, got: []const f32, want: []const f32, width: usize) !void {
    try testing.expectEqual(want.len, got.len);
    var peak: f64 = 0;
    for (want) |v| peak = @max(peak, @abs(v));
    var worst_cos: f64 = 1;
    var worst_ratio: f64 = 1;
    var worst_err: f64 = 0;
    var argmax_miss: usize = 0;
    for (0..want.len / width) |r| {
        const g = got[r * width ..][0..width];
        const w = want[r * width ..][0..width];
        var dot: f64 = 0;
        var gg: f64 = 0;
        var ww: f64 = 0;
        var ag: usize = 0;
        var aw: usize = 0;
        for (g, w, 0..) |a, b, k| {
            dot += @as(f64, a) * b;
            gg += @as(f64, a) * a;
            ww += @as(f64, b) * b;
            worst_err = @max(worst_err, @abs(@as(f64, a) - b) / peak);
            if (a > g[ag]) ag = k;
            if (b > w[aw]) aw = k;
        }
        worst_cos = @min(worst_cos, dot / @sqrt(gg * ww));
        const ratio = @sqrt(gg / ww);
        if (@abs(ratio - 1) > @abs(worst_ratio - 1)) worst_ratio = ratio;
        if (ag != aw) argmax_miss += 1;
    }
    std.debug.print("{s}: min cos {d:.7} rms ratio {d:.6} max err {e:.2} argmax misses {d}\n", .{ label, worst_cos, worst_ratio, worst_err, argmax_miss });
    try testing.expect(worst_cos > 0.9999);
    try testing.expect(@abs(worst_ratio - 1) < 1e-3);
    try testing.expectEqual(@as(usize, 0), argmax_miss);
}

fn logitsHost(m: *Dsv41Model, st: *State, ids: []const u32, want: Want) ![]f32 {
    const l = try extend(m, testing.allocator, st, ids, want);
    defer _ = mlx.mlx_array_free(l);
    return hostF32(l);
}

test "dsv41 fixture: prefill, chunked prefill, spans and decode match DeepSeek's reference (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    for ([_][]const u8{ "pipe", "repack" }) |pack| {
        var t = try Tiny.open(root, pack, "fixture.safetensors");
        defer t.deinit();
        const ids = try t.ids();
        defer gpa.free(ids);
        const V = t.m.vocab;
        const full_ref = try t.host("logits_full");
        defer gpa.free(full_ref);
        const dec_ref = try t.host("logits_decode");
        defer gpa.free(dec_ref);
        const n_prompt = ids.len - dec_ref.len / V;
        var label: [64]u8 = undefined;

        var st = try initState(t.m, gpa);
        const full = try logitsHost(t.m, &st, ids, .all);
        st.deinit();
        defer gpa.free(full);
        try expectRowsClose(try std.fmt.bufPrint(&label, "{s} full prefill", .{pack}), full, full_ref, V);

        // Chunk boundaries inside a ratio-2 group and inside the window.
        st = try initState(t.m, gpa);
        var chunked: std.ArrayList(f32) = .empty;
        defer chunked.deinit(gpa);
        var lo: usize = 0;
        for ([_]usize{ 13, 14, 31, ids.len }) |hi| {
            const part = try logitsHost(t.m, &st, ids[lo..hi], .all);
            defer gpa.free(part);
            try chunked.appendSlice(gpa, part);
            lo = hi;
        }
        st.deinit();
        try expectRowsClose(try std.fmt.bufPrint(&label, "{s} chunked prefill", .{pack}), chunked.items, full_ref, V);

        // Several spans per layer (layer-major) and several indexer score
        // blocks per span give the same rows.
        t.m.span = 7;
        t.m.idx_block_bytes = 3 * 8 * 46 * 4;
        st = try initState(t.m, gpa);
        const spanned = try logitsHost(t.m, &st, ids, .all);
        st.deinit();
        t.m.span = SPAN;
        t.m.idx_block_bytes = INDEXER_BLOCK_BYTES;
        defer gpa.free(spanned);
        try expectRowsClose(try std.fmt.bufPrint(&label, "{s} 7-row spans, 3-row score blocks", .{pack}), spanned, full_ref, V);

        st = try initState(t.m, gpa);
        defer st.deinit();
        gpa.free(try logitsHost(t.m, &st, ids[0..n_prompt], .last));
        var dec: std.ArrayList(f32) = .empty;
        defer dec.deinit(gpa);
        const draft_ids = t.fx.get("draft_ids");
        const draft_ref = if (draft_ids != null) try t.host("draft_logits") else &.{};
        defer if (draft_ids != null) gpa.free(draft_ref);
        var drafted: std.ArrayList(f32) = .empty;
        defer drafted.deinit(gpa);
        for (n_prompt..ids.len) |i| {
            const row = try logitsHost(t.m, &st, ids[i .. i + 1], .last);
            defer gpa.free(row);
            try dec.appendSlice(gpa, row);
            if (draft_ids == null) continue;
            // The draft continues from the reference's own pick.
            const ref_row = dec_ref[(i - n_prompt) * V ..][0..V];
            var t1: u32 = 0;
            for (ref_row, 0..) |v, k| {
                if (v > ref_row[t1]) t1 = @intCast(k);
            }
            var d = try draftRows(t.m, gpa, &st, t1, true);
            defer d.deinit(gpa);
            try drafted.appendSlice(gpa, d.logits);
        }
        try expectRowsClose(try std.fmt.bufPrint(&label, "{s} decode", .{pack}), dec.items, dec_ref, V);
        if (draft_ids != null) try expectRowsClose(try std.fmt.bufPrint(&label, "{s} DSpark draft", .{pack}), drafted.items, draft_ref, V);
    }
}

test "dsv41 fixture ladder: with the QAT round-trips off every layer matches in f32 (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    for ([_][]const u8{ "pipe", "repack" }) |pack| {
        var t = try Tiny.open(root, pack, "fixture_noqat.safetensors");
        defer t.deinit();
        t.m.qat_off = true;
        const ids = try t.ids();
        defer gpa.free(ids);
        var tr: Trace = .{ .gpa = gpa };
        defer tr.deinit();
        t.m.trace = &tr;
        var st = try initState(t.m, gpa);
        defer st.deinit();
        gpa.free(try logitsHost(t.m, &st, ids, .all));
        t.m.trace = null;
        for (st.layers, 0..) |*ls, li| {
            inline for (.{ "comp", "idx_k" }, .{ "comp_{d}", "idxk_{d}" }) |field, name| {
                if (@field(ls, field)) |r| {
                    const v = try r.view(r.used, mlx.gpuStream());
                    defer _ = mlx.mlx_array_free(v);
                    try tr.put(name, .{li}, v);
                }
            }
        }
        std.debug.print("{s}:\n", .{pack});
        for (tr.items.items) |it| {
            const want = t.host(it.name) catch continue;
            defer gpa.free(want);
            const got = try hostF32(it.arr);
            defer gpa.free(got);
            if (std.mem.startsWith(u8, it.name, "topk")) {
                var diff: usize = 0;
                for (got, want) |a, b| diff += @intFromBool(a != b);
                std.debug.print("  {s:<10} {d}/{d} picks differ\n", .{ it.name, diff, got.len });
                try testing.expectEqual(@as(usize, 0), diff);
                continue;
            }
            var dot: f64 = 0;
            var gg: f64 = 0;
            var ww: f64 = 0;
            var worst: f64 = 0;
            var peak: f64 = 0;
            for (got, want) |a, b| {
                dot += @as(f64, a) * b;
                gg += @as(f64, a) * a;
                ww += @as(f64, b) * b;
                worst = @max(worst, @abs(@as(f64, a) - b));
                peak = @max(peak, @abs(b));
            }
            std.debug.print("  {s:<10} cos {d:.7} ratio {d:.5} max err {e:.2}\n", .{ it.name, dot / @sqrt(gg * ww), @sqrt(gg / ww), worst / peak });
            try testing.expect(worst / peak < 1e-4);
        }
    }
}

fn argmaxRow(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, k| {
        if (v > row[best]) best = k;
    }
    return @intCast(best);
}

/// Greedy tokens from `st` after `prompt` (prefilled here), `n` of them.
fn serialGreedy(m: *Dsv41Model, st: *State, prompt: []const u32, n: usize, out: []u32) !u32 {
    var row = try logitsHost(m, st, prompt, .last);
    var t = argmaxRow(row);
    for (0..n) |k| {
        testing.allocator.free(row);
        out[k] = t;
        row = try logitsHost(m, st, &.{t}, .last);
        t = argmaxRow(row);
    }
    testing.allocator.free(row);
    return t;
}

test "dsv41 fixture: DSpark rounds emit serial greedy; rollback and full accept leave serial state (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    const m = t.m;
    try testing.expect(m.n_mtp > 0);
    const ids = try t.ids();
    defer gpa.free(ids);
    const prompt = ids[0..32];
    const N = 24;
    const B = m.ds_block;
    var serial: [N + 16]u32 = undefined;
    var st_s = try initState(m, gpa);
    defer st_s.deinit();
    const next_s = try serialGreedy(m, &st_s, prompt, N, &serial);

    var st_d = try initState(m, gpa);
    defer st_d.deinit();
    const first = try logitsHost(m, &st_d, prompt, .last);
    var tok = argmaxRow(first);
    gpa.free(first);
    var got: usize = 0;
    var rounds: usize = 0;
    var partial: usize = 0;
    var emitted: [N]u32 = undefined;
    while (got < N) {
        var r = try dsparkRound(m, gpa, &st_d, tok, N - got - 1);
        defer r.deinit(gpa);
        @memcpy(emitted[got .. got + r.tokens.len], r.tokens);
        got += r.tokens.len;
        rounds += 1;
        if (r.accepted < B and got < N) partial += 1;
        tok = r.next_token;
    }
    try testing.expectEqualSlices(u32, serial[0..N], &emitted);
    try testing.expectEqual(next_s, tok);
    try testing.expectEqual(st_s.n, st_d.n);
    try testing.expect(partial > 0); // the random head's drafts are mostly rejected

    // Every cache a rejected tail touched is back where serial decode left it.
    var ds = try draftRows(m, gpa, &st_s, tok, true);
    defer ds.deinit(gpa);
    var dd = try draftRows(m, gpa, &st_d, tok, true);
    defer dd.deinit(gpa);
    try expectRowsClose("draft after rollbacks", dd.logits, ds.logits, m.vocab);

    // A draft the trunk agrees with in full commits every row.
    var cont: [16]u32 = undefined;
    var st_c = try initState(m, gpa);
    defer st_c.deinit();
    var full_prompt: [64]u32 = undefined;
    @memcpy(full_prompt[0..prompt.len], prompt);
    @memcpy(full_prompt[prompt.len .. prompt.len + N], serial[0..N]);
    const after = try serialGreedy(m, &st_c, full_prompt[0 .. prompt.len + N], B + 1, &cont);
    const ids_full = try gpa.alloc(u32, B + 1);
    ids_full[0] = tok;
    @memcpy(ids_full[1..], cont[1 .. B + 1]);
    var inj: DsparkDraft = .{ .ids = ids_full, .len = B, .logits = try gpa.alloc(f32, 0), .confidence = try gpa.alloc(f32, 0) };
    defer inj.deinit(gpa);
    try testing.expectEqual(tok, cont[0]);
    var pending = try dsparkBeginWith(m, gpa, &st_d, tok, &inj);
    defer pending.deinit();
    var r = try dsparkFinish(m, gpa, &st_d, &pending, B, after);
    defer r.deinit(gpa);
    try testing.expectEqual(st_c.n, st_d.n);
    var dc = try draftRows(m, gpa, &st_c, after, true);
    defer dc.deinit(gpa);
    var dd2 = try draftRows(m, gpa, &st_d, after, true);
    defer dd2.deinit(gpa);
    try expectRowsClose("draft after full accept", dd2.logits, dc.logits, m.vocab);
}

test "dsv41 fixture: bf16 activations stay within bf16 noise before routing can flip (DSV41_TINY)" {
    // From the first MoE on, a random tiny model's bf16-rounded router inputs
    // flip near-tied picks, each a large per-row change; layer 0's attention
    // carries the dtype plumbing without that chaos.
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    for ([_][]const u8{ "pipe", "repack" }) |pack| {
        var t = try Tiny.openAct(root, pack, "fixture.safetensors", .bfloat16);
        defer t.deinit();
        const ids = try t.ids();
        defer gpa.free(ids);
        var tr: Trace = .{ .gpa = gpa };
        defer tr.deinit();
        t.m.trace = &tr;
        var st = try initState(t.m, gpa);
        defer st.deinit();
        const logits = try logitsHost(t.m, &st, ids, .all);
        defer gpa.free(logits);
        t.m.trace = null;
        for (logits) |v| try testing.expect(std.math.isFinite(v));
        for (tr.items.items) |it| {
            if (!std.mem.eql(u8, it.name, "attn_0")) continue;
            const want = try t.host(it.name);
            defer gpa.free(want);
            const got = try hostF32(it.arr);
            defer gpa.free(got);
            var dot: f64 = 0;
            var gg: f64 = 0;
            var ww: f64 = 0;
            for (got, want) |a, b| {
                dot += @as(f64, a) * b;
                gg += @as(f64, a) * a;
                ww += @as(f64, b) * b;
            }
            const cos = dot / @sqrt(gg * ww);
            std.debug.print("{s} bf16 {s}: cos {d:.6}\n", .{ pack, it.name, cos });
            // bf16 noise flips a few fp8 window codes.
            try testing.expect(cos > 0.999);
        }
    }
}

test "dsv41 fixture: the QAT round-trips are bit-exact with the reference's, midpoints included (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var fx = try model.loadWeightsFile(gpa, root, "qat.safetensors");
    defer fx.deinit();
    var m: Dsv41Model = undefined;
    m.qat_off = false;
    m.s = mlx.gpuStream();
    m.qat_fns = @splat(.{});
    defer for (m.qat_fns) |f| if (f.ctx != null) {
        _ = mlx.mlx_closure_free(f);
    };
    var sc = Scope.init(gpa, mlx.gpuStream());
    defer sc.deinit();
    const x = fx.get("x").?;
    const cases = .{ .{ "fp8_ue8m0_32", Qat.fp8_ue8m0, 32 }, .{ "fp4_ue8m0_32", Qat.fp4_ue8m0, 32 }, .{ "fp4_e4m3_16", Qat.fp4_e4m3, 16 } };
    // The op chain, then the compiled closures.
    for (0..2) |pass| {
        if (pass == 1) try compileQat(&m);
        inline for (cases) |cs| {
            const got = try hostF32(try qat(&sc, &m, x, cs[1], cs[2]));
            defer gpa.free(got);
            const want = try hostF32(fx.get(cs[0]).?);
            defer gpa.free(want);
            for (got, want) |a, b| try testing.expectEqual(@as(u32, @bitCast(b)), @as(u32, @bitCast(a)));
        }
    }
}

test "dsv41 perplexity: non-overlapping windows of a token corpus through extend (DSV41_PPL)" {
    // DSV41_PPL = a pack dir, DSV41_PPL_TEXT = the corpus text (wikitext-2-raw test,
    // rows joined as stored). PipeNetwork's method: windows of 2048 tokens from the
    // start, each scored on its own 2047 next-token predictions.
    const dir = std.mem.span(std.c.getenv("DSV41_PPL") orelse return error.SkipZigTest);
    const text_path = std.mem.span(std.c.getenv("DSV41_PPL_TEXT") orelse return error.SkipZigTest);
    const max_win = if (std.c.getenv("DSV41_PPL_WINDOWS")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else std.math.maxInt(usize);
    const gpa = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var cfg = try model.parseConfig(io, gpa, dir);
    defer cfg.deinit(gpa);
    var w = try model.loadModelWeights(io, gpa, dir, &cfg, false);
    defer w.deinit();
    const m = try init(gpa, &cfg, &w, mlx.gpuStream(), .{});
    defer m.deinit();
    var tok = try @import("tokenizer.zig").loadTokenizer(io, gpa, dir);
    defer tok.deinit();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, text_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    const ids = try tok.encode(gpa, text);
    defer gpa.free(ids);
    const seq = 2048;
    const n_win = @min(ids.len / seq, max_win);
    std.debug.print("[ppl] {d} tokens, {d} windows x {d}\n", .{ ids.len, n_win, seq });
    var nll_sum: f64 = 0;
    var scored: usize = 0;
    for (0..n_win) |win| {
        const wi = ids[win * seq ..][0..seq];
        var st = try initState(m, gpa);
        defer st.deinit();
        const logits = try extend(m, gpa, &st, wi, .all); // [seq, V] f32
        defer _ = mlx.mlx_array_free(logits);
        var sc = Scope.init(gpa, m.s);
        defer sc.deinit();
        const pred = try sc.slice(logits, 0, 0, seq - 1);
        var lo = mlx.mlx_array_new();
        const lse = try sc.res(mlx.mlx_logsumexp_axis(&lo, pred, -1, false, sc.s), &lo);
        const next = try sc.reshape(try sc.keep(mlx.mlx_array_new_data(wi[1..].ptr, &[_]c_int{seq - 1}, 1, .uint32)), &.{ seq - 1, 1 });
        const picked = try sc.reshape(try sc.takeAlong(pred, next, -1), &.{seq - 1});
        var so = mlx.mlx_array_new();
        const total = try sc.res(mlx.mlx_sum(&so, try sc.sub(lse, picked), false, sc.s), &so);
        try mlx.check(mlx.mlx_array_eval(total));
        const v: f64 = (mlx.mlx_array_data_float32(total) orelse return error.NoData)[0];
        nll_sum += v;
        scored += seq - 1;
        std.debug.print("[ppl] window {d}: nll {d:.4} ppl so far {d:.4}\n", .{ win, v, @exp(nll_sum / @as(f64, @floatFromInt(scored))) });
    }
    std.debug.print("[ppl] perplexity {d:.4} over {d} tokens\n", .{ @exp(nll_sum / @as(f64, @floatFromInt(scored))), scored });
}

test "dsv41 resume: a prompt that extends the last one resumes its state and matches a cold prefill (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "repack", "fixture.safetensors");
    defer t.deinit();
    const a = try t.ids();
    defer gpa.free(a);
    const m = t.m;
    // Turn 2: the first prompt plus a reply and a new question, or the first prompt again.
    const b = try gpa.alloc(u32, a.len + 6);
    defer gpa.free(b);
    @memcpy(b[0..a.len], a);
    @memcpy(b[a.len..], &[_]u32{ 5, 9, 2, 7, 3, 11 });
    for ([_][]const u32{ b, a }) |p| {
        // Turn 1: the prompt, then three decode steps.
        try testing.expectEqual(@as(usize, 0), try resumePrompt(m, a));
        _ = mlx.mlx_array_free(try prefill(m, gpa, a));
        for ([_]u32{ 5, 9, 13 }) |tok| _ = mlx.mlx_array_free(try extendResumable(m, gpa, &.{tok}));
        const n = try resumePrompt(m, p);
        try testing.expectEqual(a.len - 1, n);
        const wl = try extendResumable(m, gpa, p[n..]);
        defer _ = mlx.mlx_array_free(wl);
        const warm = try hostF32(wl);
        defer gpa.free(warm);
        m.snap.?.deinit(gpa);
        m.snap = null;
        try testing.expectEqual(@as(usize, 0), try resumePrompt(m, p));
        const cl = try prefill(m, gpa, p);
        defer _ = mlx.mlx_array_free(cl);
        const cold = try hostF32(cl);
        defer gpa.free(cold);
        try testing.expectEqualSlices(f32, cold, warm);
    }
}

test "dsv41 init: every weight the model reads is evaluated before a region traces it (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "pipe", "fixture.safetensors");
    defer t.deinit();
    var xs: std.ArrayList(A) = .empty;
    defer xs.deinit(gpa);
    for (t.m.layers) |ly| try collectArrays(gpa, &xs, ly);
    try collectArrays(gpa, &xs, .{ t.m.embed, t.m.head, t.m.norm, t.m.dspark });
    for (t.m.engrams) |e| try collectArrays(gpa, &xs, .{ e.wkv, e.qk });
    try testing.expect(xs.items.len > 100);
    for (xs.items) |x| {
        var avail = false;
        try mlx.check(mlx._mlx_array_is_available(&avail, x));
        try testing.expect(avail);
    }
}

test "dsv41 per-expert packs: banks stacked at load serve as the stacked pack, the per-expert copies dropped (DSV41_TINY)" {
    const root = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    const gpa = testing.allocator;
    var t = try Tiny.open(root, "pipe", "fixture.safetensors");
    defer t.deinit();
    const ids = try t.ids();
    defer gpa.free(ids);
    const sl = try prefill(t.m, gpa, ids);
    defer _ = mlx.mlx_array_free(sl);
    const stacked = try hostF32(sl);
    defer gpa.free(stacked);

    // The same pack under TensorFold's names: one tensor per expert and projection.
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try std.fmt.allocPrint(gpa, "{s}/pipe", .{root});
    defer gpa.free(dir);
    var w = try model.loadModelWeights(io, gpa, dir, &t.cfg, false);
    defer w.deinit();
    var banks: std.ArrayList([]const u8) = .empty;
    defer banks.deinit(gpa);
    var it = w.map.keyIterator();
    while (it.next()) |k| if (std.mem.indexOf(u8, k.*, ".ffn.experts.") != null and std.mem.indexOf(u8, k.*, "_proj.") != null) try banks.append(gpa, k.*);
    const s = mlx.gpuStream();
    for (banks.items) |key| {
        const at = std.mem.indexOf(u8, key, ".ffn.experts.").? + ".ffn.experts.".len;
        const proj = key[at..std.mem.indexOfScalarPos(u8, key, at, '.').?];
        const short = if (std.mem.eql(u8, proj, "gate_proj")) "w1" else if (std.mem.eql(u8, proj, "up_proj")) "w3" else "w2";
        const x = w.get(key).?;
        const sh = [_]c_int{ ci(dim(x, 1)), ci(dim(x, 2)) };
        for (0..dim(x, 0)) |e| {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, x, &[_]c_int{ ci(e), 0, 0 }, 3, &[_]c_int{ ci(e + 1), sh[0], sh[1] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            var flat = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&flat, part, &sh, 2, s));
            try mlx.check(mlx.mlx_array_eval(flat));
            const name = try std.fmt.allocPrint(gpa, "{s}{d}.{s}{s}", .{ key[0..at], e, short, key[at + proj.len ..] });
            try w.map.put(name, flat);
        }
    }
    for (banks.items) |key| w.remove(key);

    const m = try init(gpa, &t.cfg, &w, s, .{ .act = .float32, .dspark = true });
    defer m.deinit();
    const pl = try prefill(m, gpa, ids);
    defer _ = mlx.mlx_array_free(pl);
    const per_expert = try hostF32(pl);
    defer gpa.free(per_expert);
    try testing.expectEqualSlices(f32, stacked, per_expert);
    try testing.expect(w.get("layers.0.ffn.experts.0.w1.weight") == null);
    try testing.expect(w.get("layers.0.ffn.experts.1.w2.scales") == null);
}

test "dsv41 oMLX layouts: each linear's mode, bits and group come off its own geometry" {
    const s = mlx.gpuStream();
    const Case = struct { w: []const c_int, sc: []const c_int, fp: bool, in: u32, bits: u32, group: u32, mode: model.QuantMode };
    // Jundot oQ3e: 3-bit experts, 6-bit wo_b, the Engram projection in mxfp8, the DSpark experts in mxfp4.
    const cases = [_]Case{
        .{ .w = &.{ 384, 2304, 480 }, .sc = &.{ 384, 2304, 80 }, .fp = false, .in = 5120, .bits = 3, .group = 64, .mode = .affine },
        .{ .w = &.{ 5120, 1536 }, .sc = &.{ 5120, 128 }, .fp = false, .in = 8192, .bits = 6, .group = 64, .mode = .affine },
        .{ .w = &.{ 25600, 1536 }, .sc = &.{ 25600, 192 }, .fp = true, .in = 6144, .bits = 8, .group = 32, .mode = .mxfp8 },
        .{ .w = &.{ 128, 2304, 640 }, .sc = &.{ 128, 2304, 160 }, .fp = true, .in = 5120, .bits = 4, .group = 32, .mode = .mxfp4 },
    };
    for (cases) |c| {
        var w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w);
        try mlx.check(mlx.mlx_zeros(&w, c.w.ptr, c.w.len, .uint32, s));
        var sc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc);
        try mlx.check(mlx.mlx_zeros(&sc, c.sc.ptr, c.sc.len, if (c.fp) .uint8 else .bfloat16, s));
        const qp = transformer.quantParamsFromGeometry(w, sc, !c.fp, c.in).?;
        try testing.expectEqual(c.bits, qp.bits);
        try testing.expectEqual(c.group, qp.group_size);
        try testing.expectEqual(c.mode, qp.mode);
    }
}

test "dsv41 rows: growth keeps every appended row" {
    const s = mlx.gpuStream();
    var r: Rows = .{ .width = 4, .dtype = .float32 };
    defer r.deinit();
    var want: [300 * 4]f32 = undefined;
    for (&want, 0..) |*v, i| v.* = @floatFromInt(i);
    var at: usize = 0;
    // Chunks past the 64-row start and every quarter's growth after it.
    for ([_]usize{ 50, 40, 70, 90, 50 }) |n| {
        const x = mlx.mlx_array_new_data(want[at * 4 ..].ptr, &[_]c_int{ ci(n), 4 }, 2, .float32);
        defer _ = mlx.mlx_array_free(x);
        try r.append(x, s);
        at += n;
    }
    const v = try r.view(at, s);
    defer _ = mlx.mlx_array_free(v);
    const got = try hostF32(v);
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(f32, want[0 .. at * 4], got);
}
