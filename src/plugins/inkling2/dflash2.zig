//! Inkling 2's DFlash2 block drafter (`incoai/Inkling-2-Mini-DFlash2`, in the pack's `drafter/`), as z-lab's MLX
//! reference runs it (`dflash/model_mlx.py`, MIT): five sliding layers over a block of one anchor (the committed
//! token's normed embedding) and `block - 1` copies of the mask token's. The layers attend the trunk's taps, projected
//! once through `fc` and `hidden_norm`, at the newest `window - 1` committed positions and the block in both
//! directions, under a rotary over absolute positions; a grouped dynamic convolution (kernels `base + projected from
//! the input`) wraps attention and the MLP. The block's rows go through the trunk's head, and a selector chains one
//! path through each row's top `top_k` candidates: a candidate's score is its logit plus an edge from the token before
//! it (two per-token codebooks and a projection of the row), soft-capped AFTER the edge is added
//! (`selector_softcap_mode: after_correction`, as the sglang adapter shipped with the weights). Drafts only: the trunk
//! verifies every one. Tests: `tests/dump_inkling2_dflash2_fixtures.py` writes the reference's outputs.

const std = @import("std");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const kernelSource = @import("../../mlx/kernels.zig").source;
const head_block = @import("../../sdk/bases/blocks/head.zig");
const Linear = @import("../../sdk/bases/blocks/linear.zig").Linear;
const wload = @import("../../sdk/weights/load.zig");
const config = @import("config.zig");
const bundled = @import("draft.zig");
const kernels = @import("kernels.zig");
const device = @import("../../mlx/device.zig");

const Array = ops.Array;
const Graph = ops.Graph;
const gpa = std.heap.c_allocator;
const project = head_block.project;

var dflash_conv_kernel: kernels.Kernel = .{ .name = "msv_inkling2_dflash_conv", .ins = &.{ "X", "DYN", "BASE", "RES" }, .outs = &.{"OUT"}, .source = kernelSource("plugins/inkling2/kernels/dflash_conv.metal") };
var dflash_top16_shard_kernel: kernels.Kernel = .{ .name = "msv_inkling2_dflash_top16_shard", .ins = &.{"LOGITS"}, .outs = &.{ "PIDS", "PVALS" }, .source = kernelSource("plugins/inkling2/kernels/dflash_top16_shard.metal") };
var dflash_top16_merge_kernel: kernels.Kernel = .{ .name = "msv_inkling2_dflash_top16_merge", .ins = &.{ "PIDS", "PVALS" }, .outs = &.{ "IDS", "VALS" }, .source = kernelSource("plugins/inkling2/kernels/dflash_top16_merge.metal") };
var dflash_edges_kernel: kernels.Kernel = .{ .name = "msv_inkling2_dflash_edges", .ins = &.{ "T1", "CANDS", "UNARY", "PROJ", "PRED", "SUCC", "CAP" }, .outs = &.{"PROBS"}, .source = kernelSource("plugins/inkling2/kernels/dflash_edges.metal") };
var dflash_attn_kernel: kernels.Kernel = .{ .name = "msv_inkling2_dflash_attn", .ins = &.{ "q", "kb", "vb", "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "LANES", "scale" }, .outs = &.{ "PART", "ML" }, .source = kernelSource("plugins/inkling2/kernels/dflash_attn.metal") };
var dflash_tree_kernel: kernels.Kernel = .{ .name = "msv_inkling2_dflash_tree", .ins = &.{ "T1", "CANDS", "PROBS", "CALIB", "TEMPS", "SEEDS", "COPY_IDS", "COPY_LEN", "COPY_RATE", "STOPS", "PARAMS" }, .outs = &.{ "TOKENS", "PARENT", "TMETA", "NEXT_Q", "DRAWS", "SCORES" }, .source = kernelSource("plugins/inkling2/kernels/dflash_tree.metal") };

/// One sublayer's grouped dynamic convolution: `base` `[2, kernel, hidden]` (prepare, finish) and the projection of
/// the sublayer's normed input to its per-position kernels `[2 * kernel * groups]`.
pub const Conv = struct { base: Array, proj: Linear };

pub const Layer = struct {
    in_norm: Array,
    post_norm: Array,
    q_norm: Array,
    k_norm: Array,
    q: Linear,
    k: Linear,
    v: Linear,
    kv: Linear = .{},
    qkv: Linear = .{},
    o: Linear,
    gate: Linear,
    up: Linear,
    gate_up: Linear = .{},
    down: Linear,
    attn: Conv,
    mlp: Conv,
};

/// `hidden` `[B, L, H]` convolved causally along L (each of the B rows its own sequence), `H / group` groups of `group`
/// channels, by the kernel `base` `[taps, H]` plus the dynamic one `dyn` `[B, L, taps, groups]`: per tap the shifted rows
/// (zeros before the first) times the base, then times the dynamic, summed in the reference's order.
pub fn groupedConv(g: *Graph, hidden: Array, dyn: Array, base: Array, group: u32) !Array {
    const sh = ops.shape(hidden);
    const lanes = sh[0];
    const len = sh[1];
    const groups = @divExact(sh[2], @as(c_int, @intCast(group)));
    const gs: c_int = @intCast(group);
    const taps = ops.shape(base)[0];
    const dt = ops.dtype(hidden);
    const blocks = try g.reshape(hidden, &.{ lanes, len, groups, gs });
    var out: Array = .{};
    for (0..@intCast(taps)) |t| {
        const at: c_int = @intCast(t);
        const values = if (t == 0) blocks else try g.concat(&.{ try g.zeros(&.{ lanes, at, groups, gs }, dt), try g.sliceAxis(blocks, 1, 0, len - at) }, 1);
        const kernel = try g.astype(try g.reshape(try g.sliceAxis(base, 0, at, at + 1), &.{ 1, 1, groups, gs }), dt);
        const fixed = try g.mul(kernel, values);
        out = if (t == 0) fixed else try g.add(out, fixed);
        const moving = try g.mul(try g.reshape(try g.sliceAxis(dyn, 2, at, at + 1), &.{ lanes, len, groups, 1 }), values);
        out = try g.add(out, moving);
    }
    return g.reshape(out, &.{ lanes, len, sh[2] });
}

/// The convolution a sublayer's input takes on the way in, and the kernels the way out needs (`finish`).
const Prepared = struct { out: Array, kernel: Array };

/// The projection `base` of the draft's layer (or the root), as shipped or, with `bits` set, an affine copy of that width
/// in groups of 64 that the drafter owns (the draft's weights are a round's largest read after the head, and drafts are
/// only proposals); one whose input is not a multiple of 64 stays as shipped.
fn narrowed(at: bundled.At, base: []const u8, bits: u32) !Linear {
    var buf: [160]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}.weight", .{base});
    const dense: Linear = .{ .w = try at.raw(name) };
    if (bits == 0 or @mod(ops.shape(dense.w)[1], 64) != 0) return .{ .w = try at.keep(dense.w) };
    var own = try quantized(at.g, &dense, bits);
    for ([_]*Array{ &own.w, &own.s, &own.b }) |x| x.* = try at.keep(x.*);
    return own;
}

/// Stack multiple quantized or dense linears of matching quantization params along axis 0 into one matmul.
fn joinLinears(at: bundled.At, parts: []const *const Linear) !Linear {
    const first = parts[0];
    var ws: [3]Array = undefined;
    var ss: [3]Array = undefined;
    var bs: [3]Array = undefined;
    const quantized_w = first.s.ctx != null;
    for (parts, 0..) |p, i| {
        if ((p.s.ctx != null) != quantized_w or p.bits != first.bits or p.group_size != first.group_size) return .{};
        ws[i] = p.w;
        ss[i] = p.s;
        bs[i] = p.b;
    }
    var out: Linear = .{
        .w = try at.keep(try at.g.concat(ws[0..parts.len], 0)),
        .bits = first.bits,
        .group_size = first.group_size,
    };
    if (quantized_w) {
        out.s = try at.keep(try at.g.concat(ss[0..parts.len], 0));
        if (first.b.ctx != null) out.b = try at.keep(try at.g.concat(bs[0..parts.len], 0));
    }
    return out;
}

/// `lin` as an affine `bits` copy in groups of 64, owned by `g`.
fn quantized(g: *Graph, lin: *const Linear, bits: u32) !Linear {
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_quantize(&parts, lin.w, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, g.s));
    var own: Linear = .{ .bits = @intCast(bits), .group_size = 64 };
    inline for (.{ &own.w, &own.s, &own.b }, 0..) |dst, i| {
        var x = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&x, parts, i));
        dst.* = try g.adopt(x);
    }
    return own;
}

pub const Drafter = struct {
    cfg: config.Dflash2,
    hidden: u32,
    s: mlx.mlx_stream,
    owned: std.ArrayList(Array) = .empty,
    layers: []Layer = &.{},
    /// The trunk's taps side by side to one context row: `fc` `[hidden, taps * hidden]`, then its norm.
    fc: Linear = .{},
    hidden_norm: Array = .{},
    norm: Array = .{},
    /// The selector: each token's two codebook rows `[vocab, rank]` (bf16 gather tables, never quantized) and the
    /// projection of a block row to `rank`.
    pred: Array = .{},
    succ: Array = .{},
    sel: Linear = .{},
    /// The mask token's normed embedding `[1, 1, hidden]`, set by the plugin (`setMask`).
    mask: Array = .{},
    /// An 8-bit copy of a dense trunk head, which the block rows are read through (`draft.headCopy`); null = the trunk's.
    head: ?Linear = null,
    /// Ones `[max_lanes * block * heads * head_dim]` f32, the unit gate `attendKernel` joins its splits under.
    unit: Array = .{},

    /// The draft in `dir` (`config.json` parsed as `cfg`, `model.safetensors`), its projections held as affine `bits`
    /// copies in groups of 64 (0 = as shipped, bf16), its head reading through a copy of `head` when one is given and
    /// dense.
    pub fn load(dir: []const u8, cfg: config.Dflash2, hidden: u32, head: ?*const Linear, s: mlx.mlx_stream, bits: u32) !*Drafter {
        var path: [std.fs.max_path_bytes]u8 = undefined;
        var w = try wload.loadWeightsSingleFile(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{dir}));
        defer w.deinit();
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        d.* = .{ .cfg = cfg, .hidden = hidden, .s = s };
        errdefer d.release();
        var g = Graph.init(gpa, s);
        defer g.deinit();
        var at: bundled.At = .{ .owned = &d.owned, .w = &w, .g = &g };

        d.layers = try gpa.alloc(Layer, cfg.layers);
        for (d.layers, 0..) |*L, i| {
            at.layer = @intCast(i);
            L.* = .{
                .in_norm = try at.need("input_layernorm.weight"),
                .post_norm = try at.need("post_attention_layernorm.weight"),
                .q_norm = try at.need("self_attn.q_norm.weight"),
                .k_norm = try at.need("self_attn.k_norm.weight"),
                .q = try narrowed(at, "self_attn.q_proj", bits),
                .k = try narrowed(at, "self_attn.k_proj", bits),
                .v = try narrowed(at, "self_attn.v_proj", bits),
                .o = try narrowed(at, "self_attn.o_proj", bits),
                .gate = try narrowed(at, "mlp.gate_proj", bits),
                .up = try narrowed(at, "mlp.up_proj", bits),
                .down = try narrowed(at, "mlp.down_proj", bits),
                .attn = .{ .base = try at.need("attention_conv.base_kernel"), .proj = try at.linear("attention_conv.kernel_projection") },
                .mlp = .{ .base = try at.need("mlp_conv.base_kernel"), .proj = try at.linear("mlp_conv.kernel_projection") },
            };
            L.kv = try joinLinears(at, &.{ &L.k, &L.v });
            L.qkv = try joinLinears(at, &.{ &L.q, &L.k, &L.v });
            L.gate_up = try joinLinears(at, &.{ &L.gate, &L.up });
            const want = [3]c_int{ 2, @intCast(cfg.conv_kernel), @intCast(hidden) };
            for ([_]*const Conv{ &L.attn, &L.mlp }) |c| if (!std.mem.eql(c_int, ops.shape(c.base), &want)) return error.Inkling2DraftGeometry;
        }
        at.layer = null;
        d.fc = try narrowed(at, "fc", bits);
        d.hidden_norm = try at.need("hidden_norm.weight");
        d.norm = try at.need("norm.weight");
        d.pred = try at.need("candidate_selector.predecessor_codebook");
        d.succ = try at.need("candidate_selector.successor_codebook");
        d.sel = try at.linear("candidate_selector.hidden_projection");
        const fsh = ops.shape(d.fc.w);
        const fc_in: c_int = if (d.fc.s.ctx != null) @divExact(fsh[1] * 32, @as(c_int, @intCast(bits))) else fsh[1];
        if (fsh.len != 2 or fsh[0] != @as(c_int, @intCast(hidden)) or fc_in != @as(c_int, @intCast(cfg.tap_count * hidden))) return error.Inkling2DraftGeometry;
        if (ops.shape(d.pred)[1] != @as(c_int, @intCast(cfg.rank)) or ops.shape(d.succ)[1] != @as(c_int, @intCast(cfg.rank))) return error.Inkling2DraftGeometry;
        if (head) |h| d.head = try at.draftHead(h);
        d.unit = try at.keep(try g.ones(&.{@intCast(max_lanes * cfg.block * cfg.heads * cfg.head_dim)}, .float32));
        return d;
    }

    /// The head `rows` block rows are read through: the draft's 8-bit copy, or past `dense_head_rows` on the tensor
    /// unit the trunk's dense one, where a pass costs its bytes however many rows ride it and the copy costs by the row.
    fn headFor(d: *const Drafter, trunk: *const Linear, rows: c_int) *const Linear {
        const own = if (d.head) |*h| h else return trunk;
        return if (rows > dense_head_rows and device.naxAvailable()) trunk else own;
    }

    /// `rows` `[B, r, hidden]` through `head`, `[B, r, out]`: on the tensor unit a quantized head reads at most
    /// `head_rows` rows a call, MLX's few-row kernel (past it the tiled one, at the head's width, cost several times one
    /// call). Other GPUs keep one call: their few-row kernels' reach is unmeasured.
    fn headLogits(g: *Graph, head: *const Linear, rows: Array) !Array {
        const sh = ops.shape(rows);
        const total = sh[0] * sh[1];
        if (head.s.ctx == null or total <= head_rows or !device.naxAvailable()) return project(g, head, rows);
        const flat = try g.reshape(rows, &.{ 1, total, sh[2] });
        var parts: [max_lanes * abi_max_round]Array = undefined;
        var n: usize = 0;
        var at: c_int = 0;
        while (at < total) : (at += head_rows) {
            parts[n] = try project(g, head, try g.sliceAxis(flat, 1, at, @min(total, at + head_rows)));
            n += 1;
        }
        return g.reshape(try g.concat(parts[0..n], 1), &.{ sh[0], sh[1], -1 });
    }

    fn release(d: *Drafter) void {
        for (d.owned.items) |a| _ = mlx.mlx_array_free(a);
        d.owned.deinit(gpa);
        gpa.free(d.layers);
    }

    pub fn deinit(d: *Drafter) void {
        d.release();
        gpa.destroy(d);
    }

    /// The mask token's normed embedding `[1, 1, hidden]`, kept by the drafter.
    pub fn setMask(d: *Drafter, g: *Graph, row: Array) !void {
        const x = try g.astype(row, ops.dtype(d.norm));
        try ops.eval(x);
        d.mask = g.escape(x);
        try d.owned.append(gpa, d.mask);
    }

    /// Where a block's rotary starts: nowhere, at one position, or at each batch row's (`[B]` int32).
    const Rope = union(enum) { none, at: c_int, each: Array };

    /// `x` `[B, n, count * head_dim]` as heads `[B, count, n, head_dim]`, normed per head (null = not) and rotated.
    fn heads(d: *const Drafter, g: *Graph, x: Array, count: u32, norm_weight: ?Array, rope: Rope) !Array {
        const lanes = ops.shape(x)[0];
        const n = ops.shape(x)[1];
        const hd: c_int = @intCast(d.cfg.head_dim);
        var h = try g.reshape(x, &.{ lanes, n, @intCast(count), hd });
        if (norm_weight) |w| h = try g.rmsNorm(h, w, d.cfg.eps);
        h = try g.transpose(h, &.{ 0, 2, 1, 3 });
        return switch (rope) {
            .none => h,
            .at => |p| g.rope(h, hd, d.cfg.theta, p),
            .each => |p| g.ropeEach(h, hd, d.cfg.theta, p),
        };
    }

    fn prepare(d: *const Drafter, g: *Graph, c: *const Conv, x: Array) !Prepared {
        const dyn_raw = try project(g, &c.proj, x);
        const lanes = ops.shape(x)[0];
        const len = ops.shape(x)[1];
        const dt = ops.dtype(x);
        const ks: c_int = @intCast(d.cfg.conv_kernel);
        const groups: c_int = @intCast(d.hidden / d.cfg.conv_group);
        const dyn_t = if (ops.dtype(dyn_raw) == dt) dyn_raw else try g.astype(dyn_raw, dt);
        const dyn = try g.reshape(dyn_t, &.{ lanes, len, 2, ks, groups });
        if (mlx.streamIsGpu(g.s) and kernels.enabled() and (dt == .bfloat16 or dt == .float16 or dt == .float32)) {
            const total = lanes * len * @as(c_int, @intCast(d.hidden));
            const grid = @divTrunc(total + 63, 64) * 64;
            const out = (try kernels.launch(g, &dflash_conv_kernel, &.{ x, dyn, c.base, x }, 1, .{.{ .shape = &.{ lanes, len, @intCast(d.hidden) }, .dtype = dt }}, grid, 64, &.{
                .{ "T", .{ .dtype = dt } },
                .{ "BT", .{ .dtype = ops.dtype(c.base) } },
                .{ "H", .{ .int = @intCast(d.hidden) } },
                .{ "GS", .{ .int = @intCast(d.cfg.conv_group) } },
                .{ "KT", .{ .int = ks } },
                .{ "STAGE", .{ .int = 0 } },
            }))[0];
            return .{ .out = out, .kernel = dyn };
        }
        const base = try g.reshape(try g.sliceAxis(c.base, 0, 0, 1), &.{ ks, @intCast(d.hidden) });
        return .{
            .out = try groupedConv(g, x, try g.reshape(try g.sliceAxis(dyn, 2, 0, 1), &.{ lanes, len, ks, groups }), base, d.cfg.conv_group),
            .kernel = dyn,
        };
    }

    fn finishAdd(d: *const Drafter, g: *Graph, c: *const Conv, x: Array, dyn: Array, res: Array) !Array {
        const lanes = ops.shape(x)[0];
        const len = ops.shape(x)[1];
        const dt = ops.dtype(x);
        const ks: c_int = @intCast(d.cfg.conv_kernel);
        const groups: c_int = @intCast(d.hidden / d.cfg.conv_group);
        if (mlx.streamIsGpu(g.s) and kernels.enabled() and ops.dtype(dyn) == dt and ops.dtype(res) == dt and (dt == .bfloat16 or dt == .float16 or dt == .float32)) {
            const total = lanes * len * @as(c_int, @intCast(d.hidden));
            const grid = @divTrunc(total + 63, 64) * 64;
            return (try kernels.launch(g, &dflash_conv_kernel, &.{ x, dyn, c.base, res }, 1, .{.{ .shape = &.{ lanes, len, @intCast(d.hidden) }, .dtype = dt }}, grid, 64, &.{
                .{ "T", .{ .dtype = dt } },
                .{ "BT", .{ .dtype = ops.dtype(c.base) } },
                .{ "H", .{ .int = @intCast(d.hidden) } },
                .{ "GS", .{ .int = @intCast(d.cfg.conv_group) } },
                .{ "KT", .{ .int = ks } },
                .{ "STAGE", .{ .int = 1 } },
            }))[0];
        }
        const base = try g.reshape(try g.sliceAxis(c.base, 0, 1, 2), &.{ ks, @intCast(d.hidden) });
        const k1 = try g.reshape(try g.sliceAxis(dyn, 2, 1, 2), &.{ lanes, len, ks, groups });
        return g.add(res, try groupedConv(g, x, k1, base, d.cfg.conv_group));
    }

    /// One lane of a batched block pass: its trunk taps `[1, n, taps * hidden]` at the `n` committed positions before
    /// `position` (the anchor's), at most `window - 1`, and the context keys and values it carries over its rounds
    /// (null: projected afresh).
    pub const BlockLane = struct { taps: Array, position: c_int, cache: ?*bundled.ContextKv };

    /// The block `noise` `[1, block, hidden]` (the anchor's row, then masks) through the layers: `[1, block, hidden]`
    /// after the final norm. `taps` `[1, n, taps * hidden]` are the trunk's taps at the `n` committed positions before
    /// `position` (the anchor's), at most `window - 1`. `cache` carries each layer's context keys and values over the
    /// slot's rounds (null: projected afresh): a round projects only the rows committed since.
    pub fn blockRows(d: *const Drafter, g: *Graph, noise: Array, taps: Array, position: c_int, cache: ?*bundled.ContextKv) !Array {
        return d.blockRowsBatch(g, noise, &.{.{ .taps = taps, .position = position, .cache = cache }});
    }

    /// `blockRows` for several lanes at once: `noise` `[N, block, hidden]`, lane `j`'s block its row `j`. The layers' dense
    /// work (projections, convolutions, MLP, the context rows' keys and values) runs over all the lanes' rows; each lane's
    /// attention reads its own ring.
    pub fn blockRowsBatch(d: *const Drafter, g: *Graph, noise: Array, lanes: []const BlockLane) !Array {
        const cfg = &d.cfg;
        const block = ops.shape(noise)[1];
        const count = lanes.len;
        if (count == 0 or count > max_lanes or ops.shape(noise)[0] != @as(c_int, @intCast(count)) or block != @as(c_int, @intCast(cfg.block))) return error.Inkling2DraftRound;
        const dt = ops.dtype(d.norm);
        const keep: c_int = @intCast(cfg.window - 1);
        const kvh: c_int = @intCast(cfg.kv_heads);
        const hd: c_int = @intCast(cfg.head_dim);
        // Per lane: its ring (a temporary one without a cache), the context rows it reads and the rows its ring lacks
        // (those committed since its end, while it holds the rest of the window).
        var own: [max_lanes]bundled.ContextKv = @splat(.{});
        defer for (own[0..count]) |*c| c.release();
        var ring: [max_lanes]*bundled.ContextKv = undefined;
        var held: [max_lanes]c_int = undefined;
        var fresh: [max_lanes]c_int = undefined;
        var positions: [max_lanes]c_int = undefined;
        // Every lane's newest `block` taps (the first repeated in front while it has fewer) go through `fc` and each
        // layer's `kv` together, its fresh rows the last of them: a lane's rows depend on the batch's width, never on
        // its neighbours'. A lane that rebuilds its ring projects its own (`srcs[1..]`); `group[j]` is lane j's
        // projection and `first[j]` its first fresh row there.
        var srcs: [max_lanes + 1]Array = undefined;
        var src_pos: [max_lanes + 1]Array = undefined;
        var n_srcs: usize = 1;
        var group: [max_lanes]usize = @splat(0);
        var first: [max_lanes]c_int = @splat(0);
        var newest: [max_lanes]Array = undefined;
        var newest_pos: [max_lanes * abi_max_round]i32 = undefined;
        for (lanes, 0..) |lane, j| {
            const all = ops.shape(lane.taps)[1];
            // Rows older than the newest `window - 1` are out of every block row's reach.
            const n = @min(all, keep);
            if (n == 0) return error.Inkling2DraftRound;
            const at = lane.position;
            const end: u64 = @intCast(at);
            const cc = lane.cache orelse &own[j];
            ring[j] = cc;
            held[j] = n;
            positions[j] = at;
            fresh[j] = n;
            const whole = cc.k[0].ctx != null and ops.shape(cc.k[0])[1] == 2 * kvh and ops.shape(cc.k[0])[2] == ringSlots(n, keep);
            if (whole and cc.end <= end and end - cc.end <= @as(u64, @intCast(n)) and @as(c_int, @intCast(end - cc.end)) + cc.rows >= n) fresh[j] = @intCast(end - cc.end);
            const f = fresh[j];
            const tail = try g.sliceAxis(lane.taps, 1, all - @min(all, block), all);
            newest[j] = if (all >= block) tail else try g.concat(&.{ try g.broadcast(try g.sliceAxis(tail, 1, 0, 1), &.{ 1, block - all, ops.shape(tail)[2] }), tail }, 1);
            for (newest_pos[j * @as(usize, @intCast(block)) ..][0..@intCast(block)], 0..) |*p, r| p.* = at - block + @as(c_int, @intCast(r));
            first[j] = @as(c_int, @intCast(j)) * block + block - f;
            if (f > block) {
                const pos = try gpa.alloc(i32, @intCast(f));
                defer gpa.free(pos);
                for (pos, 0..) |*p, r| p.* = at - f + @as(c_int, @intCast(r));
                group[j] = n_srcs;
                first[j] = 0;
                srcs[n_srcs] = try g.sliceAxis(lane.taps, 1, all - f, all);
                src_pos[n_srcs] = try g.fromSlice(i32, pos, &.{f});
                n_srcs += 1;
            }
        }
        srcs[0] = if (count == 1) newest[0] else try g.concat(newest[0..count], 1);
        src_pos[0] = try g.fromSlice(i32, newest_pos[0 .. count * @as(usize, @intCast(block))], &.{@as(c_int, @intCast(count)) * block});
        var ctx: [max_lanes + 1]Array = undefined;
        for (srcs[0..n_srcs], ctx[0..n_srcs]) |x, *c| c.* = try g.rmsNorm(try project(g, &d.fc, x), d.hidden_norm, cfg.eps);
        const rope: Rope = if (count == 1) .{ .at = lanes[0].position } else .{ .each = try g.fromSlice(i32, positions[0..count], &.{@intCast(count)}) };
        // A row-per-batch `[rows, heads, 1, hd]` as heads `[1, heads, rows, hd]`.
        const as_heads = [_]c_int{ 2, 1, 0, 3 };
        const q_dim: c_int = @intCast(cfg.heads * cfg.head_dim);
        const kv_dim = kvh * hd;
        const inter_dim: c_int = ops.shape(d.layers[0].gate.w)[0];
        const attn = try d.plan(g, dt, positions[0..count], held[0..count]);
        var h = try g.astype(noise, dt);
        for (d.layers, 0..) |*L, i| {
            var lg = g.scope();
            defer lg.deinit();
            // The fresh rows' keys, normed and rotated each at its position, then their values, as heads `[1, 2 kv_heads,
            // rows, hd]`, written into their lanes' rings: in place, the old handle freed before the write runs.
            var ckv: [max_lanes + 1]Array = undefined;
            for (ctx[0..n_srcs], src_pos[0..n_srcs], 0..) |c, pos, s| {
                const rows = ops.shape(c)[1];
                var k_raw: Array = undefined;
                var v_raw: Array = undefined;
                if (L.kv.w.ctx != null) {
                    const kv_c = try project(&lg, &L.kv, c);
                    k_raw = try lg.sliceAxis(kv_c, 2, 0, kv_dim);
                    v_raw = try lg.sliceAxis(kv_c, 2, kv_dim, 2 * kv_dim);
                } else {
                    k_raw = try project(&lg, &L.k, c);
                    v_raw = try project(&lg, &L.v, c);
                }
                const k_rows = try lg.ropeEach(try lg.rmsNorm(try lg.reshape(k_raw, &.{ rows, kvh, 1, hd }), L.k_norm, cfg.eps), hd, cfg.theta, pos);
                ckv[s] = try lg.transpose(try lg.concat(&.{ k_rows, try lg.reshape(v_raw, &.{ rows, kvh, 1, hd }) }, 1), &as_heads);
            }
            var rings: [max_lanes]Array = undefined;
            for (0..count) |j| {
                const cc = ring[j];
                const f = fresh[j];
                const slots = ringSlots(held[j], keep);
                if (f > 0) {
                    const base = if (cc.k[i].ctx != null and ops.shape(cc.k[i])[1] == 2 * kvh and ops.shape(cc.k[i])[2] == slots) cc.k[i] else try lg.zeros(&.{ 1, 2 * kvh, slots, hd }, dt);
                    const next = lg.escape(try ringWrite(&lg, base, try lg.sliceAxis(ckv[group[j]], 2, first[j], first[j] + f), @mod(positions[j] - f, slots)));
                    if (cc.k[i].ctx != null) _ = mlx.mlx_array_free(cc.k[i]);
                    cc.k[i] = next;
                }
                rings[j] = cc.k[i];
            }
            const x = try lg.rmsNorm(h, L.in_norm, cfg.eps);
            const prep = try d.prepare(&lg, &L.attn, x);
            var q: Array = undefined;
            var kb: Array = undefined;
            var vb: Array = undefined;
            if (L.qkv.w.ctx != null) {
                const qkv = try project(&lg, &L.qkv, prep.out);
                q = try d.heads(&lg, try lg.sliceAxis(qkv, 2, 0, q_dim), cfg.heads, L.q_norm, rope);
                kb = try d.heads(&lg, try lg.sliceAxis(qkv, 2, q_dim, q_dim + kv_dim), cfg.kv_heads, L.k_norm, rope);
                vb = try d.heads(&lg, try lg.sliceAxis(qkv, 2, q_dim + kv_dim, q_dim + 2 * kv_dim), cfg.kv_heads, null, .none);
            } else {
                q = try d.heads(&lg, try project(&lg, &L.q, prep.out), cfg.heads, L.q_norm, rope);
                kb = try d.heads(&lg, try project(&lg, &L.k, prep.out), cfg.kv_heads, L.k_norm, rope);
                vb = try d.heads(&lg, try project(&lg, &L.v, prep.out), cfg.kv_heads, null, .none);
            }
            const read = try d.attend(&lg, q, kb, vb, rings[0..count], &attn);
            const mid = try d.finishAdd(&lg, &L.attn, try project(&lg, &L.o, read), prep.kernel, h);
            const y = try lg.rmsNorm(mid, L.post_norm, cfg.eps);
            const prep2 = try d.prepare(&lg, &L.mlp, y);
            const act = if (L.gate_up.w.ctx != null) blk: {
                const gu = try project(&lg, &L.gate_up, prep2.out);
                break :blk try lg.mul(try lg.silu(try lg.sliceAxis(gu, 2, 0, inter_dim)), try lg.sliceAxis(gu, 2, inter_dim, 2 * inter_dim));
            } else try lg.mul(try lg.silu(try project(&lg, &L.gate, prep2.out)), try project(&lg, &L.up, prep2.out));
            const mlp = try project(&lg, &L.down, act);
            h = try g.keep(try d.finishAdd(&lg, &L.mlp, mlp, prep2.kernel, mid));
        }
        for (ring[0..count], held[0..count]) |cc, n| cc.rows = n;
        return g.rmsNorm(h, d.norm, cfg.eps);
    }

    /// The block rows `q` `[N, heads, block, hd]` of N lanes, each over its ring (`rings[j]` `[1, 2 kv_heads, slots, hd]`,
    /// keys then values, position p at slot p % slots; the `ns[j]` positions before `positions[j]` its context, which
    /// row r reads only within `window` of `positions[j] + r`) and its block's own keys and values `kb`, `vb` `[N,
    /// kv_heads, block, hd]`: `[N, block, heads * hd]`. One `dflash_attn` launch on the GPU, else per lane MLX's
    /// attention over the ring and the block under a mask.
    fn attend(d: *const Drafter, g: *Graph, q: Array, kb: Array, vb: Array, rings: []const Array, p: *const Plan) !Array {
        if (p.splits > 0) return d.attendKernel(g, q, kb, vb, rings, p);
        return d.attendChain(g, q, kb, vb, rings, p.positions, p.ns);
    }

    /// A round's draft attention besides each layer's arrays: each lane's position and context rows and, where
    /// `dflash_attn` runs, its launch's (position, n, ring slots, keys a split) a lane, splits, scale and the unit gate
    /// `relcomb` joins the splits under.
    const Plan = struct { positions: []const c_int, ns: []const c_int, meta: Array = .{}, splits: c_int = 0, scale: Array = .{}, unit: Array = .{} };

    fn plan(d: *const Drafter, g: *Graph, dt: ops.Dtype, positions: []const c_int, ns: []const c_int) !Plan {
        var p: Plan = .{ .positions = positions, .ns = ns };
        const c = &d.cfg;
        if (!(mlx.streamIsGpu(g.s) and kernels.enabled() and (dt == .bfloat16 or dt == .float16 or dt == .float32) and
            c.block <= 8 and c.head_dim % 8 == 0 and c.heads % c.kv_heads == 0 and c.heads / c.kv_heads <= 8)) return p;
        const block: c_int = @intCast(c.block);
        const hkv: c_int = @intCast(c.kv_heads);
        const nl: c_int = @intCast(ns.len);
        var meta: [4 * max_lanes]i32 = undefined;
        p.splits = 1;
        // A lane's keys a split: enough splits that the launch fills the GPU, whole 16-key tiles, at least
        // `attn_keys`; its own keys and the lane count set them, so its rows do not move with its neighbours'.
        const want = @divTrunc(attn_groups + nl * hkv - 1, nl * hkv);
        for (positions, ns, 0..) |at, n, j| {
            const keys = n + block;
            const kps = @max(attn_keys, @divTrunc(@divTrunc(keys + want - 1, want) + 15, 16) * 16);
            meta[4 * j ..][0..4].* = .{ at, n, ringSlots(n, @intCast(c.window - 1)), kps };
            p.splits = @max(p.splits, @divTrunc(keys + kps - 1, kps));
        }
        p.meta = try g.fromSlice(i32, meta[0..@intCast(4 * nl)], &.{ nl, 4 });
        p.scale = try g.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim))), .float32);
        p.unit = try g.sliceAxis(d.unit, 0, 0, nl * block * @as(c_int, @intCast(c.heads * c.head_dim)));
        return p;
    }

    fn attendKernel(d: *const Drafter, g: *Graph, q: Array, kb: Array, vb: Array, rings: []const Array, p: *const Plan) !Array {
        const c = &d.cfg;
        const block: c_int = @intCast(c.block);
        const hd: c_int = @intCast(c.head_dim);
        const hq: c_int = @intCast(c.heads);
        const hkv: c_int = @intCast(c.kv_heads);
        const nl: c_int = @intCast(rings.len);
        var inputs: [5 + max_lanes]Array = undefined;
        inputs[0] = q;
        inputs[1] = kb;
        inputs[2] = vb;
        // Every lane input is bound; past the lanes they repeat the first, unread.
        for (inputs[3..][0..max_lanes], 0..) |*x, j| x.* = rings[if (j < rings.len) j else 0];
        inputs[3 + max_lanes] = p.meta;
        inputs[4 + max_lanes] = p.scale;
        const rows = nl * block * hq;
        const part = try kernels.launchGrid(g, &dflash_attn_kernel, &inputs, 2, .{
            .{ .shape = &.{ p.splits, rows, hd }, .dtype = .float32 },
            .{ .shape = &.{ p.splits, rows, 2 }, .dtype = .float32 },
        }, .{ p.splits * 32 * @divExact(hq, hkv), hkv, nl }, .{ 32 * @divExact(hq, hkv), 1, 1 }, &.{
            .{ "T", .{ .dtype = ops.dtype(q) } }, .{ "D", .{ .int = hd } },     .{ "HQ", .{ .int = hq } },
            .{ "HKV", .{ .int = hkv } },          .{ "B", .{ .int = block } }, .{ "WINDOW", .{ .int = @intCast(c.window) } },
        });
        return g.reshape(try kernels.splitJoin(g, part, p.unit, nl * block, ops.dtype(q)), &.{ nl, block, hq * hd });
    }

    fn attendChain(d: *const Drafter, g: *Graph, q: Array, kb: Array, vb: Array, rings: []const Array, positions: []const c_int, ns: []const c_int) !Array {
        const c = &d.cfg;
        const block: c_int = @intCast(c.block);
        const keep: c_int = @intCast(c.window - 1);
        const hkv: c_int = @intCast(c.kv_heads);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(c.head_dim)));
        const count = rings.len;
        var reads: [max_lanes]Array = undefined;
        for (0..count) |j| {
            const at: c_int = @intCast(j);
            const solo = count == 1;
            const slots = ops.shape(rings[j])[2];
            const total: usize = @intCast(slots + block);
            const bits = try gpa.alloc(bool, @as(usize, @intCast(block)) * total);
            defer gpa.free(bits);
            // Slot s holds the position `slots - rank` before `positions[j]`; row r reads ranks from
            // max(slots - n, slots - keep + r), and every block key.
            for (0..@intCast(block)) |r| {
                const from = @max(slots - ns[j], slots - keep + @as(c_int, @intCast(r)));
                for (bits[r * total ..][0..total], 0..) |*b, s| b.* = s >= @as(usize, @intCast(slots)) or @mod(@as(c_int, @intCast(s)) - positions[j], slots) >= from;
            }
            const kb_j = if (solo) kb else try g.sliceAxis(kb, 0, at, at + 1);
            const vb_j = if (solo) vb else try g.sliceAxis(vb, 0, at, at + 1);
            const keys = try g.concat(&.{ try g.sliceAxis(rings[j], 1, 0, hkv), kb_j }, 2);
            const values = try g.concat(&.{ try g.sliceAxis(rings[j], 1, hkv, 2 * hkv), vb_j }, 2);
            const mask = try g.fromSlice(bool, bits, &.{ 1, 1, block, @intCast(total) });
            reads[j] = try g.sdpa(if (solo) q else try g.sliceAxis(q, 0, at, at + 1), keys, values, scale, .{ .array = mask });
        }
        const attn = if (count == 1) reads[0] else try g.concat(reads[0..count], 0);
        return g.reshape(try g.transpose(attn, &.{ 0, 2, 1, 3 }), &.{ @intCast(count), block, -1 });
    }

    /// Top-k candidates `[B, r, k]` i32 and their raw logits `[B, r, k]` f32 from `raw` `[B, r, >= vocab]`.
    pub fn topCandidates(d: *const Drafter, g: *Graph, raw: Array, vocab: c_int) ![2]Array {
        const k: c_int = @intCast(d.cfg.top_k);
        const dt = ops.dtype(raw);
        if (mlx.streamIsGpu(g.s) and kernels.enabled() and k <= 32 and vocab >= k and (dt == .bfloat16 or dt == .float16 or dt == .float32)) {
            const sh = ops.shape(raw);
            const b = sh[0];
            const r = sh[1];
            const stride = sh[2];
            const r_total = b * r;
            const shards: c_int = 16;
            const flat = try g.reshape(raw, &.{ r_total, stride });
            const part = try kernels.launchGrid(g, &dflash_top16_shard_kernel, &.{flat}, 2, .{
                .{ .shape = &.{ r_total, shards, k }, .dtype = .int32 },
                .{ .shape = &.{ r_total, shards, k }, .dtype = .float32 },
            }, .{ shards * 256, r_total, 1 }, .{ 256, 1, 1 }, &.{
                .{ "T", .{ .dtype = dt } },
                .{ "V", .{ .int = vocab } },
                .{ "STRIDE", .{ .int = stride } },
                .{ "SHARDS", .{ .int = shards } },
                .{ "K", .{ .int = k } },
            });
            const merged = try kernels.launchGrid(g, &dflash_top16_merge_kernel, &.{ part[0], part[1] }, 2, .{
                .{ .shape = &.{ r_total, k }, .dtype = .int32 },
                .{ .shape = &.{ r_total, k }, .dtype = .float32 },
            }, .{ 32, r_total, 1 }, .{ 32, 1, 1 }, &.{
                .{ "SHARDS", .{ .int = shards } },
                .{ "K", .{ .int = k } },
            });
            return .{ try g.reshape(merged[0], &.{ b, r, k }), try g.reshape(merged[1], &.{ b, r, k }) };
        }
        const f = if (dt == .float32) raw else try g.astype(raw, .float32);
        const logits = if (ops.shape(f)[2] == vocab) f else try g.sliceAxis(f, 2, 0, vocab);
        const cands = try g.sliceAxis(try g.argpartition(logits, vocab - k, 2), 2, vocab - k, vocab);
        return .{ try g.astype(cands, .int32), try g.takeAlong(logits, cands, 2) };
    }

    /// The path the selector chains over `rows` `[1, r, hidden]` (block rows after the anchor's, final-normed) and their
    /// raw head logits `logits` `[1, r, >= vocab]`, from the committed token `t1`: each row's `top_k` candidates by
    /// logit, scored `softcap(logit + edge(previous token, candidate))`; argmax, or a draw under `law`.
    pub fn pick(d: *const Drafter, g: *Graph, t1: Array, rows: Array, logits: Array, law: ?bundled.Drafter.Law) !bundled.Drafter.Block {
        return d.pickVocab(g, t1, rows, logits, ops.shape(logits)[2], law);
    }

    pub fn pickVocab(d: *const Drafter, g: *Graph, t1: Array, rows: Array, logits: Array, vocab: c_int, law: ?bundled.Drafter.Law) !bundled.Drafter.Block {
        const b = try d.pickBatch(g, t1, rows, logits, vocab, if (law) |l| .{ .temperatures = &.{l.temperature}, .seed = l.seed } else null);
        const r = ops.shape(rows)[1];
        return .{
            .tokens = try g.reshape(b.tokens, &.{r}),
            .confidence = try g.reshape(b.confidence, &.{r}),
            .probs = if (b.probs.ctx != null) try g.reshape(b.probs, &.{ r, vocab }) else .{},
        };
    }

    /// How a batch's drafts are drawn: each lane's temperature, one key from `seed` and the step for the batch.
    pub const Draws = struct { temperatures: []const f32, seed: u64 };

    /// `pickVocab` for `B` lanes at once: `t1` `[B]`, `rows` `[B, r, hidden]`, `logits` `[B, r, >= vocab]`; the lanes'
    /// tokens and confidence `[B, r]` and, drawn, the laws `[B, r, vocab]`. A lane's draws are independent of its
    /// neighbours' (one key keys the whole `[B, k]` draw).
    pub fn pickBatch(d: *const Drafter, g: *Graph, t1: Array, rows: Array, logits: Array, vocab: c_int, draws: ?Draws) !bundled.Drafter.Block {
        const k: c_int = @intCast(d.cfg.top_k);
        const rank: c_int = @intCast(d.cfg.rank);
        const lanes = ops.shape(rows)[0];
        const r = ops.shape(rows)[1];
        const cap = d.cfg.softcap;
        const top = try d.topCandidates(g, logits, vocab);
        const cands = top[0];
        const unary = top[1];
        const proj = try project(g, &d.sel, rows);
        var inverse: Array = .{};
        if (draws) |dr| {
            var inv: [max_lanes]f32 = undefined;
            for (dr.temperatures, inv[0..dr.temperatures.len]) |t, *x| x.* = 1.0 / t;
            inverse = try g.fromSlice(f32, inv[0..dr.temperatures.len], &.{ lanes, 1 });
        }
        var prev = t1;
        var drafted: [abi_max_round]Array = undefined;
        var sure: [abi_max_round]Array = undefined;
        var probs: [abi_max_round]Array = undefined;
        for (0..@intCast(r)) |i| {
            const at: c_int = @intCast(i);
            const ids = try g.reshape(try g.sliceAxis(cands, 1, at, at + 1), &.{ lanes, k });
            const pe = try g.take(d.pred, prev, 0);
            const pi = try g.reshape(try g.sliceAxis(proj, 1, at, at + 1), &.{ lanes, rank });
            const edges = try g.sum(try g.mul(try g.expandDims(try g.mul(pe, pi), 1), try g.take(d.succ, ids, 0)), -1, false);
            const scores = try g.add(try g.reshape(try g.sliceAxis(unary, 1, at, at + 1), &.{ lanes, k }), try g.astype(edges, .float32));
            const capped = try g.mulScalar(try g.tanh(try g.mulScalar(scores, 1.0 / cap)), cap);
            if (draws) |dr| {
                const scaled = try g.mul(capped, inverse);
                const key = try g.call(mlx.mlx_random_key, .{dr.seed +% @as(u64, i) *% 0x9E3779B97F4A7C15});
                const which = try g.reshape(try g.call(mlx.mlx_random_categorical, .{ scaled, @as(c_int, -1), key }), &.{ lanes, 1 });
                const q = try g.softmax(scaled, -1, true);
                prev = try g.astype(try g.reshape(try g.takeAlong(ids, which, 1), &.{lanes}), .int32);
                // The law over the whole vocabulary: its mass on the candidates, none elsewhere.
                probs[i] = try g.call(mlx.mlx_put_along_axis, .{ try g.zeros(&.{ lanes, vocab }, .float32), ids, q, @as(c_int, 1) });
                sure[i] = try g.max(q, -1, false);
            } else {
                const which = try g.reshape(try g.argmax(scores, -1, false), &.{ lanes, 1 });
                prev = try g.astype(try g.reshape(try g.takeAlong(ids, which, 1), &.{lanes}), .int32);
                sure[i] = try g.max(try g.softmax(capped, -1, true), -1, false);
            }
            drafted[i] = prev;
        }
        const steps: usize = @intCast(r);
        for (drafted[0..steps], sure[0..steps]) |*x, *y| {
            x.* = try g.reshape(x.*, &.{ lanes, 1 });
            y.* = try g.reshape(y.*, &.{ lanes, 1 });
        }
        if (draws != null) for (probs[0..steps]) |*x| {
            x.* = try g.reshape(x.*, &.{ lanes, 1, vocab });
        };
        return .{
            .tokens = try g.concat(drafted[0..steps], 1),
            .confidence = try g.concat(sure[0..steps], 1),
            .probs = if (draws != null) try g.concat(probs[0..steps], 1) else .{},
        };
    }

    /// The blocks of `anchors` (each `[1, 1, hidden]`, a committed token's normed embedding): `[N, block, hidden]`, each
    /// anchor's row then the mask token's.
    fn blockNoise(d: *const Drafter, g: *Graph, anchors: []const Array) !Array {
        const first = try g.astype(if (anchors.len == 1) anchors[0] else try g.concat(anchors, 0), ops.dtype(d.norm));
        const masks = try g.broadcast(d.mask, &.{ @intCast(anchors.len), @as(c_int, @intCast(d.cfg.block)) - 1, @intCast(d.hidden) });
        return g.concat(&.{ first, masks }, 1);
    }

    /// One lane of a batched draft: the committed token `t1` (`[1]` int32) and its normed embedding `anchor` `[1, 1, hidden]`,
    /// the lane's trunk taps and context (`BlockLane`), and the drafts it asks for (`steps`, at most `block - 1`).
    pub const DraftLane = struct { t1: Array, anchor: Array, taps: Array, position: c_int, cache: ?*bundled.ContextKv, steps: u32 };

    /// `draft` for several lanes at once: the layers, the head and the selector run over all their rows, each lane reading
    /// its own context. `out[j]` is lane `j`'s block: its `steps` tokens and confidence and, drawn (`draws`: each lane's
    /// temperature), the law of each `[steps, vocab]`.
    pub fn draftBatch(d: *const Drafter, g: *Graph, lanes: []const DraftLane, draws: ?Draws, head: *const Linear, vocab: c_int, out: []bundled.Drafter.Block) !void {
        const block: c_int = @intCast(d.cfg.block);
        const count = lanes.len;
        var anchors: [max_lanes]Array = undefined;
        var lane_in: [max_lanes]BlockLane = undefined;
        var t1s: [max_lanes]Array = undefined;
        var most: c_int = 1;
        for (lanes, 0..) |lane, j| {
            anchors[j] = lane.anchor;
            lane_in[j] = .{ .taps = lane.taps, .position = lane.position, .cache = lane.cache };
            t1s[j] = lane.t1;
            most = @max(most, @min(block - 1, @as(c_int, @intCast(lane.steps))));
        }
        const rows_all = try d.blockRowsBatch(g, try d.blockNoise(g, anchors[0..count]), lane_in[0..count]);
        for (lanes) |lane| if (lane.cache) |cc| {
            cc.end = @intCast(lane.position);
        };
        const rows = try g.sliceAxis(rows_all, 1, 1, 1 + most);
        const b = try d.pickBatch(g, if (count == 1) t1s[0] else try g.concat(t1s[0..count], 0), rows, try headLogits(g, d.headFor(head, ops.shape(rows)[0] * ops.shape(rows)[1]), rows), vocab, draws);
        for (lanes, out[0..count], 0..) |lane, *o, j| {
            const at: c_int = @intCast(j);
            const steps: c_int = @min(block - 1, @as(c_int, @intCast(lane.steps)));
            o.* = .{
                .tokens = try g.reshape(try g.sliceAxis(try g.sliceAxis(b.tokens, 0, at, at + 1), 1, 0, steps), &.{steps}),
                .confidence = try g.reshape(try g.sliceAxis(try g.sliceAxis(b.confidence, 0, at, at + 1), 1, 0, steps), &.{steps}),
                .probs = if (b.probs.ctx != null) try g.reshape(try g.sliceAxis(try g.sliceAxis(b.probs, 0, at, at + 1), 1, 0, steps), &.{ steps, vocab }) else .{},
            };
        }
    }

    /// The `block - 1` tokens drafted after the committed token `t1` (`[1]` int32) at `position`. `anchor` `[1, 1,
    /// hidden]` is `t1`'s normed embedding, `taps` the ring (`blockRows`), `head` the trunk's LM head over `vocab` real
    /// rows. `law` draws the drafts at the request's temperature, as many as it allows.
    pub fn draft(d: *const Drafter, g: *Graph, t1: Array, anchor: Array, taps: Array, position: c_int, head: *const Linear, vocab: c_int, cache: ?*bundled.ContextKv, law: ?bundled.Drafter.Law) !bundled.Drafter.Block {
        var out: [1]bundled.Drafter.Block = undefined;
        const steps: u32 = if (law) |l| @min(l.steps, d.cfg.block - 1) else d.cfg.block - 1;
        try d.draftBatch(g, &.{.{ .t1 = t1, .anchor = anchor, .taps = taps, .position = position, .cache = cache, .steps = steps }}, if (law) |l| .{ .temperatures = &.{l.temperature}, .seed = l.seed } else null, head, vocab, &out);
        return out[0];
    }

    pub const max_copy = 15;

    /// One lane of a GPU-side draft tree round (`draftTreeBatch`).
    pub const TreeLane = struct {
        t1: u32,
        anchor: Array,
        taps: Array,
        position: c_int,
        cache: ?*bundled.ContextKv,
        /// <= 0 = greedy top-k tree; > 0 = sampled Gumbel-top-k without replacement at this temperature.
        temperature: f32 = 0,
        seed: u64 = 0,
        calib_slope: f32 = 0.7,
        calib_intercepts: [8]f32 = .{ 0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45, -0.45 },
        copy: []const u32 = &.{},
        copy_rate: f32 = 0.9,
    };

    /// The GPU-built draft tree for `B` lanes of `w` verify rows each (`draftTreeBatch`).
    pub const TreeBlock = struct {
        tokens: Array, // [B, W] i32
        parent: Array, // [B, W] i32
        tmeta: Array,  // [B * W, 2] i32
        cands: Array,  // [B, R, K] i32
        next_q: Array, // [B, W, K] f32
        draws: Array,  // [B, W, 4] u8
        scores: Array, // [B, W] f32
        w: c_int,
        r: c_int,
        k: c_int,
    };

    /// Builds the `B`-lane `w`-row draft tree on GPU from block rows `[B, R, hidden]` and raw logits `[B, R, >= vocab]`.
    pub fn buildTreeFromLogits(
        d: *const Drafter,
        g: *Graph,
        lanes: []const TreeLane,
        rows: Array,
        logits: Array,
        vocab: c_int,
        w: c_int,
        max_children: c_int,
        stops: []const u32,
    ) !TreeBlock {
        const b: c_int = @intCast(lanes.len);
        const r: c_int = ops.shape(rows)[1];
        const k: c_int = @intCast(d.cfg.top_k);
        const rank: c_int = @intCast(d.cfg.rank);
        const top = try d.topCandidates(g, logits, vocab);
        const cands = top[0];
        const unary = top[1];
        const proj = try project(g, &d.sel, rows);

        var t1_buf: [max_lanes]i32 = undefined;
        var calib_buf: [max_lanes * 9]f32 = undefined;
        var temp_buf: [max_lanes]f32 = undefined;
        var seed_buf: [max_lanes * 2]u32 = undefined;
        var copy_ids_buf: [max_lanes * max_copy]i32 = @splat(0);
        var copy_len_buf: [max_lanes]i32 = @splat(0);
        var copy_rate_buf: [max_lanes]f32 = @splat(0);
        const cal_stride: usize = @intCast(1 + r);
        for (lanes, 0..) |lane, i| {
            t1_buf[i] = @intCast(lane.t1);
            calib_buf[i * cal_stride + 0] = lane.calib_slope;
            for (0..@intCast(r)) |p| calib_buf[i * cal_stride + 1 + p] = lane.calib_intercepts[@min(p, lane.calib_intercepts.len - 1)];
            temp_buf[i] = lane.temperature;
            seed_buf[i * 2 + 0] = @truncate(lane.seed);
            seed_buf[i * 2 + 1] = @truncate(lane.seed >> 32);
            const cn = @min(lane.copy.len, max_copy);
            copy_len_buf[i] = @intCast(cn);
            copy_rate_buf[i] = lane.copy_rate;
            for (0..cn) |j| copy_ids_buf[i * max_copy + j] = @intCast(lane.copy[j]);
        }
        const t1_arr = try g.fromSlice(i32, t1_buf[0..lanes.len], &.{b});
        const cap_arr = try g.fromSlice(f32, &.{d.cfg.softcap}, &.{1});
        const pred_dt = ops.dtype(d.pred);
        const succ = if (ops.dtype(d.succ) == pred_dt) d.succ else try g.astype(d.succ, pred_dt);
        const probs = (try kernels.launchGrid(g, &dflash_edges_kernel, &.{ t1_arr, cands, unary, proj, d.pred, succ, cap_arr }, 1, .{.{ .shape = &.{ b, r, k, k }, .dtype = .float32 }}, .{ k * 32, r, b }, .{ 32, 1, 1 }, &.{
            .{ "PT", .{ .dtype = ops.dtype(proj) } },
            .{ "CT", .{ .dtype = pred_dt } },
            .{ "R", .{ .int = r } },
            .{ "K", .{ .int = k } },
            .{ "RANK", .{ .int = rank } },
        }))[0];

        var stop_buf: [16]i32 = @splat(-1);
        const n_stops: usize = @min(stops.len, stop_buf.len);
        for (stops[0..n_stops], stop_buf[0..n_stops]) |sid, *dst| dst.* = @intCast(sid);
        const tree = try kernels.launchGrid(g, &dflash_tree_kernel, &.{
            t1_arr,
            cands,
            probs,
            try g.fromSlice(f32, calib_buf[0 .. lanes.len * cal_stride], &.{ b, 1 + r }),
            try g.fromSlice(f32, temp_buf[0..lanes.len], &.{b}),
            try g.fromSlice(u32, seed_buf[0 .. lanes.len * 2], &.{ b, 2 }),
            try g.fromSlice(i32, copy_ids_buf[0 .. lanes.len * max_copy], &.{ b, max_copy }),
            try g.fromSlice(i32, copy_len_buf[0..lanes.len], &.{b}),
            try g.fromSlice(f32, copy_rate_buf[0..lanes.len], &.{b}),
            try g.fromSlice(i32, &stop_buf, &.{@intCast(stop_buf.len)}),
            try g.fromSlice(i32, &.{ @intCast(n_stops), max_children, w }, &.{3}),
        }, 6, .{
            .{ .shape = &.{ b, w }, .dtype = .int32 },
            .{ .shape = &.{ b, w }, .dtype = .int32 },
            .{ .shape = &.{ b * w, 2 }, .dtype = .int32 },
            .{ .shape = &.{ b, w, k }, .dtype = .float32 },
            .{ .shape = &.{ b, w, 4 }, .dtype = .uint8 },
            .{ .shape = &.{ b, w }, .dtype = .float32 },
        }, .{ b, 1, 1 }, .{ 1, 1, 1 }, &.{
            .{ "W", .{ .int = w } },
            .{ "R", .{ .int = r } },
            .{ "K", .{ .int = k } },
            .{ "MAX_COPY", .{ .int = max_copy } },
        });
        return .{
            .tokens = tree[0],
            .parent = tree[1],
            .tmeta = tree[2],
            .cands = cands,
            .next_q = tree[3],
            .draws = tree[4],
            .scores = tree[5],
            .w = w,
            .r = r,
            .k = k,
        };
    }

    /// Runs the 5-layer DFlash2 block forward for `lanes` and builds each lane's `w`-row draft tree on GPU with zero
    /// mid-round syncs and no `[B, R, V]` dense probability materialization.
    pub fn draftTreeBatch(
        d: *const Drafter,
        g: *Graph,
        lanes: []const TreeLane,
        head: *const Linear,
        vocab: c_int,
        w: c_int,
        max_children: c_int,
        stops: []const u32,
    ) !TreeBlock {
        const block: c_int = @intCast(d.cfg.block);
        const count = lanes.len;
        if (count == 0 or count > max_lanes) return error.Inkling2DraftRound;
        var anchors: [max_lanes]Array = undefined;
        var lane_in: [max_lanes]BlockLane = undefined;
        for (lanes, 0..) |lane, j| {
            anchors[j] = lane.anchor;
            lane_in[j] = .{ .taps = lane.taps, .position = lane.position, .cache = lane.cache };
        }
        const noise = try d.blockNoise(g, anchors[0..count]);
        const rows_all = try d.blockRowsBatch(g, noise, lane_in[0..count]);
        for (lanes) |lane| if (lane.cache) |cc| {
            cc.end = @intCast(lane.position);
        };
        const r: c_int = @min(block - 1, @max(@as(c_int, 1), w - 1));
        const rows = try g.sliceAxis(rows_all, 1, 1, 1 + r);
        const logits = try headLogits(g, d.headFor(head, @as(c_int, @intCast(count)) * r), rows);
        return d.buildTreeFromLogits(g, lanes, rows, logits, vocab, w, max_children, stops);
    }
};

const abi_max_round = @import("../../abi/plugin.zig").max_round_tokens;


/// The slots of a ring that holds `n` context rows, at most `keep`: a power of two from 128 up, so a short context holds
/// little, and the ring is rebuilt the few times it grows.
fn ringSlots(n: c_int, keep: c_int) c_int {
    return @min(keep, std.math.ceilPowerOfTwoAssert(u32, @intCast(@max(n, 128))));
}

/// `rows` `[1, heads, m, hd]` written into the ring `buf` `[1, heads, slots, hd]` from slot `start` on, wrapping.
fn ringWrite(g: *Graph, buf: Array, rows: Array, start: c_int) !Array {
    const sh = ops.shape(buf);
    const m = ops.shape(rows)[2];
    const head = @min(m, sh[2] - start);
    var out = try g.sliceUpdate(buf, if (head == m) rows else try g.sliceAxis(rows, 2, 0, head), &.{ 0, 0, start, 0 }, &.{ sh[0], sh[1], start + head, sh[3] });
    if (head < m) out = try g.sliceUpdate(out, try g.sliceAxis(rows, 2, head, m), &.{ 0, 0, 0, 0 }, &.{ sh[0], sh[1], m - head, sh[3] });
    return out;
}

/// The lanes one batched pass of the drafter carries.
pub const max_lanes = 16;

/// The threadgroups a `dflash_attn` launch aims for, and its fewest keys a split.
const attn_groups = 512;
const attn_keys = 32;

/// The block rows past which a draft reads the trunk's dense head (`Drafter.headFor`).
pub const dense_head_rows = 16;
/// The rows one call through a quantized head takes (`Drafter.headLogits`).
const head_rows = 16;

// ── tests ──

const testing = std.testing;

// Bar: a draft of a few rows reads its own head copy, one of many rows the trunk's dense head where the tensor unit is.
test "inkling2 dflash2: many block rows read the trunk's dense head, a few the draft's copy" {
    var d: Drafter = .{ .cfg = undefined, .hidden = 0, .s = undefined, .head = .{} };
    const trunk: Linear = .{};
    try testing.expectEqual(&d.head.?, d.headFor(&trunk, dense_head_rows));
    try testing.expectEqual(if (device.naxAvailable()) &trunk else &d.head.?, d.headFor(&trunk, dense_head_rows + 1));
    d.head = null;
    try testing.expectEqual(&trunk, d.headFor(&trunk, 1));
}

fn cpu() mlx.mlx_stream {
    return mlx.mlx_default_cpu_stream_new();
}

fn host(g: *Graph, x: Array) ![]const f32 {
    const f = try g.astype(x, .float32);
    try ops.eval(f);
    const data = mlx.mlx_array_data_float32(f) orelse return error.NoData;
    return data[0..mlx.mlx_array_size(f)];
}

/// A drafter of `layers` random f32 layers (hidden 32, 2 heads of 16 over one KV head, block 4) on `g`'s stream; layer 0
/// runs the joined projections, the others the separate ones.
fn tinyDrafter(g: *Graph, layers: []Layer, window: u32, seed: u64) !Drafter {
    const H = 32;
    const hd = 16;
    const inter = 32;
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    const rand = struct {
        fn make(gr: *Graph, r: std.Random, sh: []const c_int, sc: f32) !Array {
            var n: usize = 1;
            for (sh) |x| n *= @intCast(x);
            const buf = try gpa.alloc(f32, n);
            defer gpa.free(buf);
            for (buf) |*x| x.* = r.floatNorm(f32) * sc;
            return gr.fromSlice(f32, buf, sh);
        }
    }.make;
    var d: Drafter = .{
        .cfg = .{ .layers = @intCast(layers.len), .heads = 2, .kv_heads = 1, .head_dim = hd, .intermediate = inter, .window = window, .block = 4, .mask_id = 0, .tap_count = 2, .conv_kernel = 2, .conv_group = 8, .rank = 4, .top_k = 4, .softcap = 30 },
        .hidden = H,
        .s = g.s,
        .layers = layers,
    };
    const lin = struct {
        fn make(gr: *Graph, r: std.Random, out: c_int, in: c_int) !Linear {
            return .{ .w = try rand(gr, r, &.{ out, in }, 1.0 / @sqrt(@as(f32, @floatFromInt(in)))) };
        }
    }.make;
    for (layers, 0..) |*L, i| {
        L.* = .{
            .in_norm = try rand(g, rng, &.{H}, 0.1),
            .post_norm = try rand(g, rng, &.{H}, 0.1),
            .q_norm = try rand(g, rng, &.{hd}, 0.1),
            .k_norm = try rand(g, rng, &.{hd}, 0.1),
            .q = try lin(g, rng, 2 * hd, H),
            .k = try lin(g, rng, hd, H),
            .v = try lin(g, rng, hd, H),
            .o = try lin(g, rng, H, 2 * hd),
            .gate = try lin(g, rng, inter, H),
            .up = try lin(g, rng, inter, H),
            .down = try lin(g, rng, H, inter),
            .attn = .{ .base = try rand(g, rng, &.{ 2, 2, H }, 0.3), .proj = try lin(g, rng, 2 * 2 * 4, H) },
            .mlp = .{ .base = try rand(g, rng, &.{ 2, 2, H }, 0.3), .proj = try lin(g, rng, 2 * 2 * 4, H) },
        };
        for ([_]*Array{ &L.in_norm, &L.post_norm, &L.q_norm, &L.k_norm }) |x| x.* = try g.addScalar(x.*, 1);
        if (i == 0) {
            L.kv = .{ .w = try g.concat(&.{ L.k.w, L.v.w }, 0) };
            L.qkv = .{ .w = try g.concat(&.{ L.q.w, L.k.w, L.v.w }, 0) };
            L.gate_up = .{ .w = try g.concat(&.{ L.gate.w, L.up.w }, 0) };
        }
    }
    d.fc = try lin(g, rng, H, 2 * H);
    d.hidden_norm = try g.addScalar(try rand(g, rng, &.{H}, 0.1), 1);
    d.norm = try g.addScalar(try rand(g, rng, &.{H}, 0.1), 1);
    return d;
}

// Bar: a lane's context ring carried over rounds (growing through its sizes to the window, sliding past it, wrapping
// its slots, rebuilt after a jump) drafts what a fresh projection of the same taps does, and is written in place.
test "inkling2 dflash2: a carried context ring is written in place and drafts what a fresh one does" {
    const s = cpu();
    defer _ = mlx.mlx_stream_free(s);
    var g0 = Graph.init(testing.allocator, s);
    defer g0.deinit();
    var layers: [2]Layer = undefined;
    const d = try tinyDrafter(&g0, &layers, 300, 41);
    var prng = std.Random.DefaultPrng.init(42);
    const T = 420;
    const all_host = try testing.allocator.alloc(f32, T * 64);
    defer testing.allocator.free(all_host);
    for (all_host) |*x| x.* = prng.random().floatNorm(f32);
    const all = try g0.fromSlice(f32, all_host, &.{ 1, T, 64 });
    const noise_host = try testing.allocator.alloc(f32, 4 * 32);
    defer testing.allocator.free(noise_host);
    for (noise_host) |*x| x.* = prng.random().floatNorm(f32);
    const noise = try g0.fromSlice(f32, noise_host, &.{ 1, 4, 32 });
    // (position, ring rows): growing a few rows a round and by a long run, through 128 and 256 slots to the window's
    // 299, sliding and wrapping, a jump past the ring onto a short one, regrowing.
    const rounds = [_][2]c_int{ .{ 5, 5 }, .{ 8, 8 }, .{ 120, 120 }, .{ 127, 127 }, .{ 130, 130 }, .{ 255, 255 }, .{ 258, 258 }, .{ 300, 299 }, .{ 303, 299 }, .{ 307, 299 }, .{ 350, 299 }, .{ 400, 6 }, .{ 403, 9 } };
    var carried: bundled.ContextKv = .{};
    defer carried.release();
    var where: usize = 0;
    var slots: c_int = 0;
    for (rounds) |rd| {
        var g = Graph.init(testing.allocator, s);
        defer g.deinit();
        const ring = try g.sliceAxis(all, 1, rd[0] - rd[1], rd[0]);
        const got = try d.blockRows(&g, noise, ring, rd[0], &carried);
        carried.end = @intCast(rd[0]);
        const want = try d.blockRows(&g, noise, ring, rd[0], null);
        const x = try host(&g, got);
        const y = try host(&g, want);
        for (x, y) |u, v| try testing.expectApproxEqAbs(v, u, 1e-4 * @max(1, @abs(v)));
        const at = @intFromPtr(mlx.mlx_array_data_float32(carried.k[0]).?);
        if (ops.shape(carried.k[0])[2] == slots) try testing.expectEqual(where, at);
        where = at;
        slots = ops.shape(carried.k[0])[2];
    }
}

// Bar: on the GPU the draft attention kernel (`dflash_attn`, joined by `relcomb`) reads each lane's ring and its block's
// keys as the op chain does (MLX's attention under the ring mask): full and short rings of their own sizes, wrapped
// slots, several splits, 1 to 10 lanes, in bf16, f16 and f32.
test "inkling2 dflash2: the draft attention kernel reads each lane's ring as the op chain does" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0xd2a7);
    const rng = prng.random();
    const H = 8;
    const KV = 2;
    const HD = 128;
    const B = 8;
    const W = 600;
    const keep = W - 1;
    var d: Drafter = .{
        .cfg = .{ .layers = 1, .heads = H, .kv_heads = KV, .head_dim = HD, .intermediate = 64, .window = W, .block = B, .mask_id = 0, .tap_count = 2, .conv_kernel = 2, .conv_group = 16, .rank = 16, .top_k = 16, .softcap = 30 },
        .hidden = 64,
        .s = s,
    };
    const ns = [_]c_int{ keep, keep, 5, 300, keep, 1, keep, 450, keep, 12 };
    const pos = [_]c_int{ 5000, 1234, 5, 300, 640, 9, 2 * keep + 3, 450, 9999, 12 };
    const rand = struct {
        fn make(gr: *Graph, r: std.Random, sh: []const c_int, sc: f32, dt: ops.Dtype) !Array {
            var n: usize = 1;
            for (sh) |x| n *= @intCast(x);
            const buf = try gpa.alloc(f32, n);
            defer gpa.free(buf);
            for (buf) |*x| x.* = r.floatNorm(f32) * sc;
            return gr.astype(try gr.fromSlice(f32, buf, sh), dt);
        }
    }.make;
    for ([_]ops.Dtype{ .bfloat16, .float16, .float32 }) |dt| for ([_]usize{ 1, 3, ns.len }) |count| {
        var g = Graph.init(testing.allocator, s);
        defer g.deinit();
        d.unit = try g.ones(&.{max_lanes * B * H * HD}, .float32);
        const nl: c_int = @intCast(count);
        const q = try rand(&g, rng, &.{ nl, H, B, HD }, 1.0, dt);
        const kb = try rand(&g, rng, &.{ nl, KV, B, HD }, 1.0, dt);
        const vb = try rand(&g, rng, &.{ nl, KV, B, HD }, 1.0, dt);
        var rings: [ns.len]Array = undefined;
        for (rings[0..count], ns[0..count]) |*r, n| r.* = try rand(&g, rng, &.{ 1, 2 * KV, ringSlots(n, keep), HD }, 1.0, dt);
        const plan = try d.plan(&g, dt, pos[0..count], ns[0..count]);
        try testing.expect(plan.splits > 0);
        const got = try host(&g, try d.attendKernel(&g, q, kb, vb, rings[0..count], &plan));
        const want = try host(&g, try d.attendChain(&g, q, kb, vb, rings[0..count], pos[0..count], ns[0..count]));
        try testing.expectEqual(want.len, got.len);
        const tol: f32 = if (dt == .float32) 2e-3 else 3e-2;
        for (got, want) |u, v| try testing.expectApproxEqAbs(v, u, tol);
    };
}

// Bar: the grouped dynamic convolution is, per group of channels and per tap, the base plus the dynamic kernel times
// the row that many rows back (zeros before the first), a row's own tap first.
test "inkling2 dflash2: a grouped dynamic convolution sums each tap's kernel over the shifted rows" {
    const s = cpu();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const L = 5;
    const H = 4;
    const gs = 2;
    var hidden: [L * H]f32 = undefined;
    for (&hidden, 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) * 0.5 - 1;
    var dyn: [L * 2 * 2]f32 = undefined; // [L, taps, groups]
    for (&dyn, 0..) |*x, i| x.* = @as(f32, @floatFromInt((i * 3) % 5)) * 0.25 - 0.4;
    const base = [_]f32{ 1, 2, 3, 4, 0.5, -1, 0.25, 2 }; // [taps, H]
    const got = try host(&g, try groupedConv(&g, try g.fromSlice(f32, &hidden, &.{ 1, L, H }), try g.fromSlice(f32, &dyn, &.{ 1, L, 2, 2 }), try g.fromSlice(f32, &base, &.{ 2, H }), gs));
    for (0..L) |l| for (0..H) |c| {
        var want: f32 = 0;
        for (0..2) |t| {
            if (l < t) continue;
            const x = hidden[(l - t) * H + c];
            want += base[t * H + c] * x + dyn[(l * 2 + t) * 2 + c / gs] * x;
        }
        try testing.expectApproxEqAbs(want, got[l * H + c], 1e-5);
    };
}

// Bar: the selector chains a path over each row's top candidates: a candidate scores its logit plus an edge from the
// token before it, capped after the sum; the path's first predecessor is the committed token.
test "inkling2 dflash2: the selector chains each row's best candidate off the token before it" {
    const s = cpu();
    defer _ = mlx.mlx_stream_free(s);
    var g = Graph.init(testing.allocator, s);
    defer g.deinit();
    const V = 24;
    const R = 4;
    const H = 6;
    const K = 3;
    var d: Drafter = .{ .cfg = .{ .layers = 1, .heads = 1, .kv_heads = 1, .head_dim = 2, .intermediate = 2, .window = 16, .block = R + 1, .mask_id = 0, .tap_count = 2, .conv_kernel = 2, .conv_group = 2, .rank = R, .top_k = K, .softcap = 5 }, .hidden = H, .s = s };
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    var pred: [V * R]f32 = undefined;
    var succ: [V * R]f32 = undefined;
    var sel: [R * H]f32 = undefined;
    var rows: [R * H]f32 = undefined;
    var logits: [R * V]f32 = undefined;
    for (&pred) |*x| x.* = rng.floatNorm(f32);
    for (&succ) |*x| x.* = rng.floatNorm(f32);
    for (&sel) |*x| x.* = rng.floatNorm(f32);
    for (&rows) |*x| x.* = rng.floatNorm(f32);
    for (&logits, 0..) |*x, i| x.* = @as(f32, @floatFromInt((i * 7919) % 101)) * 0.07 + rng.floatNorm(f32) * 0.001;
    d.pred = try g.fromSlice(f32, &pred, &.{ V, R });
    d.succ = try g.fromSlice(f32, &succ, &.{ V, R });
    d.sel = .{ .w = try g.fromSlice(f32, &sel, &.{ R, H }) };
    const t1: i32 = 5;
    const t1_a = try g.fromSlice(i32, &.{t1}, &.{1});
    const rows_a = try g.fromSlice(f32, &rows, &.{ 1, R, H });
    const logits_a = try g.fromSlice(f32, &logits, &.{ 1, R, V });
    const block = try d.pick(&g, t1_a, rows_a, logits_a, null);
    try ops.evalMany(&.{ block.tokens, block.confidence });
    const got = mlx.mlx_array_data_int32(block.tokens).?[0..R];
    var prev: usize = @intCast(t1);
    for (0..R) |r| {
        // The row's top K by logit, then the best of them by logit + edge.
        var ids: [K]usize = undefined;
        var taken: [V]bool = @splat(false);
        for (&ids) |*id| {
            var best: usize = V;
            for (0..V) |v| if (!taken[v] and (best == V or logits[r * V + v] > logits[r * V + best])) {
                best = v;
            };
            taken[best] = true;
            id.* = best;
        }
        var best_id: usize = 0;
        var best_score: f32 = -std.math.inf(f32);
        for (ids) |id| {
            var edge: f32 = 0;
            for (0..R) |c| {
                var proj: f32 = 0;
                for (0..H) |h| proj += sel[c * H + h] * rows[r * H + h];
                edge += pred[prev * R + c] * proj * succ[id * R + c];
            }
            const score = logits[r * V + id] + edge;
            if (score > best_score) {
                best_score = score;
                best_id = id;
            }
        }
        try testing.expectEqual(@as(i32, @intCast(best_id)), got[r]);
        prev = best_id;
    }
    const sure = mlx.mlx_array_data_float32(block.confidence).?[0..R];
    for (sure) |x| try testing.expect(x > 1.0 / @as(f32, K) - 1e-6 and x <= 1.0);
}

// Bar: on the published weights (`INKLING2_DFLASH2_DIR`) and the reference's outputs (`INKLING2_DFLASH2_FIXTURES`), the
// block rows after the final norm agree with z-lab's MLX model (cosine and scale) on a short context and on one past the
// window, where the ring's oldest keys fall out of a later row's reach, and the selector chains the reference's path.
test "inkling2 dflash2: the published draft's block rows and selector path match the reference" {
    const env = @import("../../env.zig");
    const dir = std.mem.span(env.get(.INKLING2_DFLASH2_DIR) orelse return error.SkipZigTest);
    const fixtures = std.mem.span(env.get(.INKLING2_DFLASH2_FIXTURES) orelse return error.SkipZigTest);
    if (mlx.noGpuBackend() or mlx.virtualGpu()) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const json = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/config.json", .{dir}), a, .limited(1 << 20));
    const cfg = config.readDflash2(json, 2304, 48) orelse return error.TestUnexpectedResult;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const d = try Drafter.load(dir, cfg, 2304, null, s, 0);
    defer d.deinit();
    var fx = try wload.loadWeightsSingleFile(gpa, fixtures);
    defer fx.deinit();
    var g = Graph.init(gpa, s);
    defer g.deinit();
    const cases = [_]struct { name: []const u8, rows: c_int, first: c_int }{ .{ .name = "short", .rows = 40, .first = 100 }, .{ .name = "long", .rows = 2100, .first = 5000 } };
    for (cases) |c| {
        var buf: [32]u8 = undefined;
        const noise = fx.get(try std.fmt.bufPrint(&buf, "{s}.noise", .{c.name})).?;
        const taps = fx.get(try std.fmt.bufPrint(&buf, "{s}.taps", .{c.name})).?;
        const want = fx.get(try std.fmt.bufPrint(&buf, "{s}.hidden", .{c.name})).?;
        // The ring keeps the newest `window - 1` rows; the reference drops the older ones itself.
        const keep: c_int = @intCast(cfg.window - 1);
        const ring = try g.sliceAxis(taps, 1, @max(0, c.rows - keep), c.rows);
        const got = try d.blockRows(&g, noise, ring, c.first + c.rows, null);
        const x = try host(&g, got);
        const y = try host(&g, want);
        try testing.expectEqual(y.len, x.len);
        var dot: f64 = 0;
        var xx: f64 = 0;
        var yy: f64 = 0;
        for (x, y) |u, v| {
            dot += @as(f64, u) * v;
            xx += @as(f64, u) * u;
            yy += @as(f64, v) * v;
        }
        const cos = dot / @sqrt(xx * yy);
        const rms_ratio = @sqrt(xx / yy);
        std.debug.print("dflash2 {s}: cos {d:.6} rms ratio {d:.4}\n", .{ c.name, cos, rms_ratio });
        try testing.expect(cos > 0.999);
        try testing.expect(rms_ratio > 0.99 and rms_ratio < 1.01);
    }

    // The 8-bit copies the plugin serves (`draft_bits`) stay within rounding of the reference on the short context.
    {
        const d8 = try Drafter.load(dir, cfg, 2304, null, s, 8);
        defer d8.deinit();
        const taps = fx.get("short.taps").?;
        const got = try d8.blockRows(&g, fx.get("short.noise").?, taps, 140, null);
        const x = try host(&g, got);
        const y = try host(&g, fx.get("short.hidden").?);
        var dot: f64 = 0;
        var xx: f64 = 0;
        var yy: f64 = 0;
        for (x, y) |u, v| {
            dot += @as(f64, u) * v;
            xx += @as(f64, u) * u;
            yy += @as(f64, v) * v;
        }
        std.debug.print("dflash2 8-bit: cos {d:.6} rms ratio {d:.4}\n", .{ dot / @sqrt(xx * yy), @sqrt(xx / yy) });
        try testing.expect(dot / @sqrt(xx * yy) > 0.995);
        try testing.expect(@sqrt(xx / yy) > 0.97 and @sqrt(xx / yy) < 1.03);
    }

    // The selector, on the reference's rows and sparse logits.
    const rows = fx.get("sel.rows").?;
    const ids = fx.get("sel.logits_ids").?;
    const vals = fx.get("sel.logits_vals").?;
    const path = fx.get("sel.path").?;
    const vocab: usize = 200058;
    const r: usize = @intCast(ops.shape(rows)[1]);
    const dense = try a.alloc(f32, r * vocab);
    @memset(dense, -10);
    const id_host = try g.astype(ids, .int32);
    try ops.eval(id_host);
    const idp = mlx.mlx_array_data_int32(id_host).?;
    const vp = try host(&g, vals);
    const per = @as(usize, @intCast(ops.shape(ids)[2]));
    for (0..r) |pos| for (0..per) |j| {
        dense[pos * vocab + @as(usize, @intCast(idp[pos * per + j]))] = vp[pos * per + j];
    };
    const block = try d.pick(&g, try g.fromSlice(i32, &.{4242}, &.{1}), rows, try g.fromSlice(f32, dense, &.{ 1, @intCast(r), @intCast(vocab) }), null);
    try ops.evalMany(&.{block.tokens});
    const got_path = mlx.mlx_array_data_int32(block.tokens).?[0..r];
    const path_i32 = try g.astype(path, .int32);
    try ops.eval(path_i32);
    const want_path = mlx.mlx_array_data_int32(path_i32).?[0..r];
    std.debug.print("dflash2 selector path {any} reference {any}\n", .{ got_path, want_path });
    try testing.expectEqualSlices(i32, want_path, got_path);
}

// Bar: on GPU, the fused grouped dynamic convolution (`dflash_conv.metal` prepare + finishAdd) matches `groupedConv` +
// residual add, and the 16-shard top-K selector (`dflash_top16_*`) matches exact sorted top-K (lowest id on ties).
test "inkling2 dflash2: GPU fused groupedConv and topCandidates match CPU reference" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const gs_s = mlx.gpuStream();
    var g = Graph.init(a, gs_s);
    defer g.deinit();

    var prng = std.Random.DefaultPrng.init(0xdf12);
    const rng = prng.random();
    const B: c_int = 3;
    const L: c_int = 8;
    const H: c_int = 64;
    const GS: u32 = 16;
    const KT: c_int = 2;
    const G: c_int = @divExact(H, @as(c_int, @intCast(GS)));

    const x_buf = try a.alloc(f32, @intCast(B * L * H));
    defer a.free(x_buf);
    const res_buf = try a.alloc(f32, @intCast(B * L * H));
    defer a.free(res_buf);
    const base_buf = try a.alloc(f32, @intCast(2 * KT * H));
    defer a.free(base_buf);
    const proj_buf = try a.alloc(f32, @intCast(2 * KT * G * H));
    defer a.free(proj_buf);
    for (x_buf) |*v| v.* = rng.floatNorm(f32) * 0.5;
    for (res_buf) |*v| v.* = rng.floatNorm(f32) * 0.5;
    for (base_buf) |*v| v.* = rng.floatNorm(f32) * 0.3;
    for (proj_buf) |*v| v.* = rng.floatNorm(f32) * 0.1;

    var d: Drafter = .{
        .cfg = .{ .layers = 1, .heads = 2, .kv_heads = 1, .head_dim = 32, .intermediate = 64, .window = 16, .block = L, .mask_id = 0, .tap_count = 2, .conv_kernel = @intCast(KT), .conv_group = GS, .rank = 16, .top_k = 16, .softcap = 30 },
        .hidden = @intCast(H),
        .s = gs_s,
    };
    const conv: Conv = .{
        .base = try g.astype(try g.fromSlice(f32, base_buf, &.{ 2, KT, H }), .bfloat16),
        .proj = .{ .w = try g.astype(try g.fromSlice(f32, proj_buf, &.{ 2 * KT * G, H }), .bfloat16) },
    };
    const x_bf = try g.astype(try g.fromSlice(f32, x_buf, &.{ B, L, H }), .bfloat16);
    const res_bf = try g.astype(try g.fromSlice(f32, res_buf, &.{ B, L, H }), .bfloat16);

    const prep = try d.prepare(&g, &conv, x_bf);
    const got_fin = try d.finishAdd(&g, &conv, prep.out, prep.kernel, res_bf);

    // Reference via groupedConv on same bf16 tensors.
    const dyn0 = try g.reshape(try g.sliceAxis(prep.kernel, 2, 0, 1), &.{ B, L, KT, G });
    const dyn1 = try g.reshape(try g.sliceAxis(prep.kernel, 2, 1, 2), &.{ B, L, KT, G });
    const base0 = try g.reshape(try g.sliceAxis(conv.base, 0, 0, 1), &.{ KT, H });
    const base1 = try g.reshape(try g.sliceAxis(conv.base, 0, 1, 2), &.{ KT, H });
    const ref_prep = try groupedConv(&g, x_bf, dyn0, base0, GS);
    const ref_fin = try g.add(res_bf, try groupedConv(&g, ref_prep, dyn1, base1, GS));

    const gp = try host(&g, prep.out);
    const rp = try host(&g, ref_prep);
    for (gp, rp) |u, v| try testing.expectApproxEqAbs(v, u, 2e-2);
    const gf = try host(&g, got_fin);
    const rf = try host(&g, ref_fin);
    for (gf, rf) |u, v| try testing.expectApproxEqAbs(v, u, 3e-2);

    // Sharded top-16 vs exact host top-16 with padding columns (STRIDE > V).
    const R_rows: usize = 7;
    const V: usize = 4096;
    const STRIDE: usize = 4200;
    const K: usize = 16;
    const l_buf = try a.alloc(f32, 2 * R_rows * STRIDE);
    defer a.free(l_buf);
    for (0..2 * R_rows) |row| {
        for (0..V) |c| l_buf[row * STRIDE + c] = @as(f32, @floatFromInt(((row * 131 + c * 7919) % 10007))) * 0.001;
        // Padding past V has larger values to verify V bound is respected.
        for (V..STRIDE) |c| l_buf[row * STRIDE + c] = 999.0;
    }
    const raw = try g.fromSlice(f32, l_buf, &.{ 2, @intCast(R_rows), @intCast(STRIDE) });
    const top = try d.topCandidates(&g, raw, @intCast(V));
    try ops.evalMany(&top);
    const got_ids = mlx.mlx_array_data_int32(top[0]).?;
    const got_vals = mlx.mlx_array_data_float32(top[1]).?;
    for (0..2 * R_rows) |row| {
        var taken = try a.alloc(bool, V);
        defer a.free(taken);
        @memset(taken, false);
        for (0..K) |m| {
            var best: usize = V;
            for (0..V) |c| {
                if (taken[c]) continue;
                const val = l_buf[row * STRIDE + c];
                if (best == V or val > l_buf[row * STRIDE + best]) best = c;
            }
            taken[best] = true;
            try testing.expectEqual(@as(i32, @intCast(best)), got_ids[row * K + m]);
            try testing.expectApproxEqAbs(l_buf[row * STRIDE + best], got_vals[row * K + m], 1e-6);
        }
    }
}

// Bar: GPU `buildTreeFromLogits` produces valid topological trees, extends greedy copy chains past R, excludes stops
// from greedy trees, and maintains exact draw-order child placement prefixes for sampled trees.
test "inkling2 dflash2: GPU draft tree builder builds valid greedy, copy-merged, and sampled trees" {
    if (mlx.noGpuBackend() or mlx.virtualGpu() or !mlx.metalKernelsAvailable()) return error.SkipZigTest;
    const a = testing.allocator;
    const gs_s = mlx.gpuStream();
    var g = Graph.init(a, gs_s);
    defer g.deinit();

    const V: c_int = 128;
    const R: c_int = 7;
    const K: c_int = 16;
    const RANK: c_int = 16;
    const H: c_int = 32;
    const W: c_int = 16;
    var prng = std.Random.DefaultPrng.init(0x7ee);
    const rng = prng.random();

    const allocRand = struct {
        fn make(al: std.mem.Allocator, r: std.Random, n: usize, sc: f32) ![]f32 {
            const buf = try al.alloc(f32, n);
            for (buf) |*x| x.* = r.floatNorm(f32) * sc;
            return buf;
        }
    }.make;
    const pred_b = try allocRand(a, rng, @intCast(V * RANK), 0.3);
    defer a.free(pred_b);
    const succ_b = try allocRand(a, rng, @intCast(V * RANK), 0.3);
    defer a.free(succ_b);
    const sel_b = try allocRand(a, rng, @intCast(RANK * H), 0.2);
    defer a.free(sel_b);
    const rows_b = try allocRand(a, rng, @intCast(2 * R * H), 0.5);
    defer a.free(rows_b);
    const logits_b = try allocRand(a, rng, @intCast(2 * R * V), 2.0);
    defer a.free(logits_b);
    // Make token 2 (stop id) the highest logit on lane 0 pos 0 to prove greedy excludes stops.
    logits_b[2] = 50.0;

    var d: Drafter = .{
        .cfg = .{ .layers = 1, .heads = 1, .kv_heads = 1, .head_dim = 16, .intermediate = 32, .window = 16, .block = R + 1, .mask_id = 0, .tap_count = 2, .conv_kernel = 2, .conv_group = 16, .rank = @intCast(RANK), .top_k = @intCast(K), .softcap = 30 },
        .hidden = @intCast(H),
        .s = gs_s,
    };
    d.pred = try g.astype(try g.fromSlice(f32, pred_b, &.{ V, RANK }), .bfloat16);
    d.succ = try g.astype(try g.fromSlice(f32, succ_b, &.{ V, RANK }), .bfloat16);
    d.sel = .{ .w = try g.astype(try g.fromSlice(f32, sel_b, &.{ RANK, H }), .bfloat16) };

    const copy_seq = [_]u32{ 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21 };
    const lanes = [_]Drafter.TreeLane{
        .{ .t1 = 5, .anchor = .{}, .taps = .{}, .position = 10, .cache = null, .temperature = 0.0, .seed = 1, .copy = &copy_seq, .copy_rate = 0.99 },
        .{ .t1 = 9, .anchor = .{}, .taps = .{}, .position = 10, .cache = null, .temperature = 0.8, .seed = 42 },
    };
    const tb = try d.buildTreeFromLogits(
        &g,
        &lanes,
        try g.astype(try g.fromSlice(f32, rows_b, &.{ 2, R, H }), .bfloat16),
        try g.astype(try g.fromSlice(f32, logits_b, &.{ 2, R, V }), .bfloat16),
        V,
        W,
        4,
        &.{2},
    );
    try ops.evalMany(&.{ tb.tokens, tb.parent, tb.tmeta, tb.cands, tb.next_q, tb.draws });
    const toks = mlx.mlx_array_data_int32(tb.tokens).?;
    const pars = mlx.mlx_array_data_int32(tb.parent).?;
    const meta = mlx.mlx_array_data_int32(tb.tmeta).?;
    const cands = mlx.mlx_array_data_int32(tb.cands).?;
    const nq = mlx.mlx_array_data_float32(tb.next_q).?;
    const draws = mlx.mlx_array_data_uint8(tb.draws).?;

    // Check topological invariants for both lanes.
    for (0..2) |b| {
        const base = b * @as(usize, @intCast(W));
        try testing.expectEqual(@as(i32, @intCast(lanes[b].t1)), toks[base]);
        try testing.expectEqual(@as(i32, -1), pars[base]);
        try testing.expectEqual(@as(i32, 0), meta[base * 2]);
        try testing.expectEqual(@as(i32, 1), meta[base * 2 + 1]);
        for (1..@intCast(W)) |r| {
            const p = pars[base + r];
            try testing.expect(p >= 0 and p < @as(i32, @intCast(r)));
            const pu: usize = @intCast(p);
            try testing.expectEqual(meta[(base + pu) * 2] + 1, meta[(base + r) * 2]);
            try testing.expectEqual(meta[(base + pu) * 2 + 1] | (@as(i32, 1) << @intCast(r)), meta[(base + r) * 2 + 1]);
            if (b == 0) try testing.expect(toks[base + r] != 2); // greedy never places stop id
        }
    }
    // Lane 0 has copy_rate = 0.99 over 12 copy tokens: verify copy chain extends past R = 7.
    var max_dep0: i32 = 0;
    for (0..@intCast(W)) |r| max_dep0 = @max(max_dep0, meta[r * 2]);
    try testing.expect(max_dep0 > R);

    // Lane 1 (sampled): verify next_q sums to 1 for expanded nodes and every placed draw slot matches a child.
    const b1: usize = @intCast(W);
    for (0..@intCast(W)) |r| {
        const dep: usize = @intCast(meta[(b1 + r) * 2]);
        if (dep < @as(usize, @intCast(R))) {
            var qsum: f32 = 0;
            for (0..@intCast(K)) |k| qsum += nq[(b1 + r) * @as(usize, @intCast(K)) + k];
            if (qsum > 0) try testing.expectApproxEqAbs(@as(f32, 1.0), qsum, 1e-4);
        }
        var ended = false;
        for (0..4) |ch| {
            const di = draws[(b1 + r) * 4 + ch];
            if (di == 255) {
                ended = true;
            } else {
                try testing.expect(!ended);
                try testing.expect(di < @as(u8, @intCast(K)));
                const expect_tok = cands[(1 * @as(usize, @intCast(R)) + dep) * @as(usize, @intCast(K)) + di];
                var found = false;
                for (r + 1..@intCast(W)) |child| {
                    if (pars[b1 + child] == @as(i32, @intCast(r)) and toks[b1 + child] == expect_tok) {
                        found = true;
                        break;
                    }
                }
                try testing.expect(found);
            }
        }
    }
}
