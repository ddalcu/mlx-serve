//! Reading a pack's tensors during one load (`LoadCtx.weights`) into a base's blocks: projections and expert banks
//! through the pack's quant format, tables kept `[out, in]`, raw tensors as new handles, row-joined groups, and the
//! list of handles the loaded model owns. Shared by the hybrid and MoE plugins (lfm2, nemotron_h, hy_v3, mimo_v2).
//! copied_from: plugins/llama/plugin.zig (`Reader`, `join`, `Loaded.own`; that one is private to llama).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const formats = @import("../quant/formats.zig");
const native = @import("../quant/native.zig");
const gguf_format = @import("../quant/gguf/format.zig");
const transposeBf16Weight = @import("../../mlx/array.zig").transposeBf16Weight;
const linear = @import("blocks/linear.zig");
const moe = @import("blocks/moe.zig");
const row_exact = @import("blocks/row_exact.zig");
const fuseRowGroup = @import("../blocks/linear/fused_rows.zig").fuseRowGroup;
const model = @import("../weights/load.zig");

const Array = mlx.mlx_array;
const Linear = linear.Linear;
const gpa = std.heap.c_allocator;

/// The handles a loaded model holds, freed together.
pub const Owned = struct {
    list: std.ArrayList(Array) = .empty,
    /// A projection registered through `linear` is outside the row-exact kernels (`row_exact.fits`).
    outside_rows: bool = false,

    pub fn own(o: *Owned, a: Array) !void {
        if (a.ctx != null) try o.list.append(gpa, a);
    }

    /// A projection's arrays; `gather` = a table only gathered from (no matmul reads it).
    pub fn linear(o: *Owned, l: Linear) !void {
        if (!row_exact.fits(&l)) o.outside_rows = true;
        try o.gather(l);
    }

    pub fn gather(o: *Owned, l: Linear) !void {
        for ([_]Array{ l.w, l.s, l.b, l.bias }) |a| try o.own(a);
    }

    pub fn deinit(o: *Owned) void {
        for (o.list.items) |a| _ = mlx.mlx_array_free(a);
        o.list.deinit(gpa);
    }
};

pub const Reader = struct {
    src: *const quant_abi.Source,
    fmt: *const quant_abi.QuantTable,
    qctx: quant_abi.LoadCtx,
    prefix: []const u8,
    s: mlx.mlx_stream,

    /// Opens the pack's quant format over `ctx.weights`; `close` when the load is done.
    pub fn open(ctx: *const abi.LoadCtx, block_json: []const u8, prefix: []const u8) !Reader {
        const src = ctx.weights orelse return error.NoWeightSource;
        const fmt = (formats.claimant(block_json, "{}") catch null) orelse &native.affine;
        var r: Reader = .{
            .src = src,
            .fmt = fmt,
            .qctx = .{ .block_json = abi.Str.of(block_json), .source = src, .stream = ctx.stream },
            .prefix = prefix,
            .s = .{ .ctx = ctx.stream.ctx },
        };
        if (!fmt.open(&r.qctx).isOk()) return error.QuantFormatRefused;
        return r;
    }

    pub fn close(r: *Reader) void {
        r.fmt.close(&r.qctx);
    }

    /// `<prefix>.<fmt>` in `buf`.
    pub fn name(r: *const Reader, buf: []u8, comptime fmt: []const u8, args: anytype) ![]const u8 {
        if (r.prefix.len == 0) return std.fmt.bufPrint(buf, fmt, args);
        return std.fmt.bufPrint(buf, "{s}." ++ fmt, .{r.prefix} ++ args);
    }

    pub fn has(r: *const Reader, full: []const u8) bool {
        return r.src.tensor(full).ctx != null;
    }

    /// A new handle on the source's tensor `full`, empty when the pack has none.
    pub fn share(r: *const Reader, full: []const u8) !Array {
        const a: Array = .{ .ctx = r.src.tensor(full).ctx };
        if (a.ctx == null) return a;
        var h = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(h);
        try mlx.check(mlx.mlx_array_set(&h, a));
        return h;
    }

    pub fn need(r: *const Reader, full: []const u8) !Array {
        const a = try r.share(full);
        if (a.ctx == null) return error.MissingWeight;
        return a;
    }

    /// The projection (or bank) `base` through the pack's quant format (`in_dim` pins its quant params), unowned.
    pub fn proj(r: *Reader, base: []const u8, in_dim: u32) !Linear {
        // By name: under `-Dunits` another unit may hand over its own copy of the table.
        if (std.mem.eql(u8, r.fmt.name.slice(), gguf_format.table.name.slice())) return r.ggufProj(base, in_dim);
        var w: quant_abi.Weight = .{};
        const v = r.fmt.load_tensor(&r.qctx, abi.Str.of(base), in_dim, &w);
        if (!v.isOk()) return if (std.mem.eql(u8, v.why.slice(), "MissingWeight")) error.MissingWeight else error.QuantFormatRefused;
        // e4m3 (Sushi's row-scaled MiMo q/k/v, a release's 64 x 128 grid): the matmul knows it by its dtypes; no affine
        // kernel may claim it.
        const fp8 = w.layout == @intFromEnum(native.Layout.fp8_rows) or w.layout == @intFromEnum(native.Layout.fp8_block128);
        const bits: u32 = if (fp8) 0 else w.bits;
        return .{ .w = ctxArr(w.w), .s = ctxArr(w.scales), .b = ctxArr(w.biases), .bits = bits, .group_size = w.group_size, .mode = native.modeOf(&w) };
    }

    /// A GGUF's projection: block-quantized beside its sentinel (the gguf kernels read it, `QuantMode.gguf`), or one
    /// of its float tensors, loaded bf16 and dense as the engine binds a scale-less weight.
    fn ggufProj(r: *Reader, base: []const u8, in_dim: u32) !Linear {
        var nb: [256]u8 = undefined;
        if (!r.has(try std.fmt.bufPrint(&nb, "{s}.scales", .{base}))) {
            const raw = r.src.tensor(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}));
            if (raw.ctx == null) return error.MissingWeight;
            return .{ .w = try transposeBf16Weight(.{ .ctx = raw.ctx }, r.s) };
        }
        var w: quant_abi.Weight = .{};
        if (!r.fmt.load_tensor(&r.qctx, abi.Str.of(base), in_dim, &w).isOk()) return error.QuantFormatRefused;
        return .{ .w = ctxArr(w.w), .s = ctxArr(w.scales), .mode = .gguf };
    }

    /// A routed bank the format runs itself (`QuantTable.swiglu`, EXL3), unowned.
    pub fn formatBank(r: *Reader, base: []const u8, in_dim: u32) !quant_abi.Weight {
        var w: quant_abi.Weight = .{};
        const v = r.fmt.load_bank(&r.qctx, abi.Str.of(base), in_dim, &w);
        if (!v.isOk()) return if (std.mem.eql(u8, v.why.slice(), "MissingWeight")) error.MissingWeight else error.QuantFormatRefused;
        return w;
    }

    /// `proj`, or null when the pack has no `<base>.weight`.
    pub fn projOpt(r: *Reader, base: []const u8, in_dim: u32) !?Linear {
        var nb: [256]u8 = undefined;
        if (!r.has(try std.fmt.bufPrint(&nb, "{s}.weight", .{base}))) return null;
        return try r.proj(base, in_dim);
    }

    /// A table kept in its checkpoint layout `[out, in]` (the embedding, the head): dense by its dtype, as the engine
    /// decides it, else through the format.
    pub fn table(r: *Reader, base: []const u8, in_dim: u32) !Linear {
        var nb: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&nb, "{s}.weight", .{base});
        const w = r.src.tensor(key);
        if (w.ctx == null) return error.MissingWeight;
        switch (mlx.mlx_array_dtype(.{ .ctx = w.ctx })) {
            .bfloat16, .float16, .float32 => return .{ .w = try r.need(key) },
            else => {},
        }
        if (r.src.tensor(try std.fmt.bufPrint(&nb, "{s}.scales", .{base})).ctx == null) return error.MissingWeight;
        return r.proj(base, in_dim);
    }
};

/// The weight prefix the checkpoint uses: `declared` when `<declared>.<leaf>` exists, else the first of `others`
/// that holds it (a converter's nesting is its own choice).
pub fn probePrefix(ctx: *const abi.LoadCtx, declared: []const u8, others: []const []const u8, leaf: []const u8) ![]const u8 {
    const src = ctx.weights orelse return error.NoWeightSource;
    var nb: [160]u8 = undefined;
    if (src.tensor(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ declared, leaf })).ctx != null) return declared;
    for (others) |p| if (src.tensor(try std.fmt.bufPrint(&nb, "{s}.{s}", .{ p, leaf })).ctx != null) return p;
    return declared;
}

fn ctxArr(a: abi.Array) Array {
    return .{ .ctx = a.ctx };
}

/// An expert bank: a quantized one carries its own geometry, a dense one the pack-wide scheme (the engine's
/// `quantParamsHinted` for a scale-less weight).
pub fn bank(l: Linear, q: abi.Quant) moe.Bank {
    if (l.s.ctx != null) return .{ .w = l.w, .s = l.s, .b = l.b, .qp = .{ .bits = l.bits, .group_size = l.group_size, .mode = l.mode } };
    return .{ .w = l.w, .qp = .{ .bits = q.bits, .group_size = if (q.group_size > 0) q.group_size else 64, .mode = switch (q.mode) {
        .nvfp4 => .nvfp4,
        .mxfp4 => .mxfp4,
        .mxfp8 => .mxfp8,
        else => .affine,
    } } };
}

/// Joins quantized projections reading one input into one buffer (rows) when they share a packed geometry, rewriting
/// each to a row view of it and freeing its own copy; empty when they do not.
pub fn join(o: *Owned, parts: []const *Linear, s: mlx.mlx_stream) !linear.Joined {
    var triples: [4][3]*Array = undefined;
    var originals: [4][3]Array = undefined;
    for (parts, 0..) |p, i| {
        triples[i] = .{ &p.w, &p.s, &p.b };
        originals[i] = .{ p.w, p.s, p.b };
    }
    const fused = (try fuseRowGroup(triples[0..parts.len], s)) orelse return .{};
    for (originals[0..parts.len]) |set| for (set) |a| if (a.ctx != null) {
        _ = mlx.mlx_array_free(a);
    };
    for ([_]Array{ fused.w, fused.s, fused.b }) |a| try o.own(a);
    const q = parts[0];
    return .{ .lin = .{ .w = fused.w, .s = fused.s, .b = fused.b, .bits = q.bits, .group_size = q.group_size, .mode = q.mode }, .widths = fused.widths };
}

/// A torch-layout depthwise conv weight `[C, 1, K]` as the `[C, K, 1]` `conv1d` contracts against (materialized),
/// else `w` itself. Takes `w`'s reference.
pub fn depthwiseConv(w: Array, s: mlx.mlx_stream) !Array {
    const sh = ops.shape(w);
    if (!(sh.len == 3 and sh[1] == 1 and sh[2] > 1)) return w;
    defer _ = mlx.mlx_array_free(w);
    var t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t);
    try mlx.check(mlx.mlx_transpose_axes(&t, w, &[_]c_int{ 0, 2, 1 }, 3, s));
    var c = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, t, false, s));
    return c;
}

/// A `[n]` f32 zero vector (an absent selection bias).
pub fn zerosF32(n: u32, s: mlx.mlx_stream) !Array {
    var z = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(z);
    try mlx.check(mlx.mlx_zeros(&z, &[_]c_int{@intCast(n)}, 1, .float32, s));
    return z;
}

/// `SpecHooks.head` over a loaded model with a `head: ?*Head` beside its base: `load` builds it once (null = the pack
/// ships none), `forward` runs it on the trunk's embedding table. The engine keeps the head's cache and selects
/// drafts, so the per-slot calls have nothing to do.
/// An MTP head's tensors in the legacy `resolveMtpSource` order: the first of `sidecars` that holds `marker`, else the
/// index's shards holding a `prefix` key, else `model.safetensors`. Loads are lazy: only what the head reads loads.
/// A 2-bit group-128 pack is ternary when every bias of the tensors it loads is exactly -scale (the engine's
/// `firstNonTernaryBias`: one eval; `gather_only`, an untied embedding table, is skipped; a pack without biases is not
/// ternary). Then every 2-bit projection under `roots` (structs walked through fields, unions, optionals, arrays and
/// slices) reads qmv2's kernels (`Linear.ternary`), as the engine's `qmatmul`; the projections inside an `Unread`
/// type count toward the verdict but keep stock kernels (the engine runs that block outside `qmatmul`).
pub fn armTernary(roots: anytype, comptime Unread: ?type, gather_only: Array, s: mlx.mlx_stream) !void {
    var t: Ternary = .{ .skip = gather_only };
    defer t.lins.deinit(gpa);
    defer t.pairs.deinit(gpa);
    inline for (roots) |r| try t.visit(r, Unread, true);
    if (!try ternaryPack(t.pairs.items, s)) return;
    for (t.lins.items) |l| l.ternary = l.bits == 2 and l.group_size == 128 and l.mode == .affine;
}

/// The embedding table when only gathered (the head is another tensor); null when it is also the head.
pub fn gatherOnly(head: *const Linear, embed: Linear) Array {
    return if (head.w.ctx == embed.w.ctx) .{ .ctx = null } else embed.w;
}

const Ternary = struct {
    skip: Array,
    lins: std.ArrayList(*Linear) = .empty,
    pairs: std.ArrayList([2]Array) = .empty,

    fn visit(t: *Ternary, p: anytype, comptime Unread: ?type, comptime flag: bool) !void {
        const T = @typeInfo(@TypeOf(p)).pointer.child;
        if (T == Linear) {
            if (p.s.ctx == null or p.w.ctx == t.skip.ctx) return;
            if (flag) try t.lins.append(gpa, p);
            return t.pairs.append(gpa, .{ p.s, p.b });
        }
        if (T == moe.Bank) return if (p.s.ctx != null) t.pairs.append(gpa, .{ p.s, p.b });
        const f2 = flag and T != Unread;
        switch (@typeInfo(T)) {
            .@"struct" => |st| inline for (st.field_names) |f| try t.visit(&@field(p.*, f), Unread, f2),
            .@"union" => switch (p.*) {
                inline else => |*v| try t.visit(v, Unread, f2),
            },
            .optional => if (p.*) |*v| try t.visit(v, Unread, f2),
            .array => for (p) |*v| try t.visit(v, Unread, f2),
            .pointer => |pt| if (pt.size == .slice and @typeInfo(pt.child) == .@"struct" and !pt.attrs.@"const") for (p.*) |*v| try t.visit(v, Unread, f2),
            else => {},
        }
    }
};

/// The engine's ternary verdict over the quantized `pairs` (scales, biases), in one eval: every bias exactly -scale.
/// A pack with no biases, or a pair with scales and no biases, is not ternary.
fn ternaryPack(pairs: []const [2]Array, s: mlx.mlx_stream) !bool {
    var g = ops.Graph.init(gpa, s);
    defer g.deinit();
    const peaks = try gpa.alloc(Array, pairs.len);
    defer gpa.free(peaks);
    var n: usize = 0;
    for (pairs) |p| {
        if (p[0].ctx == null) continue;
        if (p[1].ctx == null) return false;
        const sum = try g.add(p[0], p[1]);
        peaks[n] = try g.astype(try g.call(mlx.mlx_max, .{ try g.call(mlx.mlx_abs, .{sum}), false }), .float32);
        n += 1;
    }
    if (n == 0) return false;
    try ops.evalMany(peaks[0..n]);
    for (peaks[0..n]) |p| {
        var v: f32 = 1;
        try mlx.check(mlx.mlx_array_item_float32(&v, p));
        if (v != 0) return false;
    }
    return true;
}

pub fn headWeights(ctx: *const abi.LoadCtx, sidecars: []const []const u8, marker: []const u8, prefix: []const u8) !model.Weights {
    var path: [std.fs.max_path_bytes]u8 = undefined;
    const dir = ctx.model_dir.slice();
    for (sidecars) |rel| {
        if (!ctx.dir.exists(ctx.dir.ctx, abi.Str.of(rel))) continue;
        var w = try model.loadWeightsSingleFile(gpa, try std.fmt.bufPrint(&path, "{s}/{s}", .{ dir, rel }));
        if (w.get(marker) != null) return w;
        w.deinit();
    }
    const index = "model.safetensors.index.json";
    if (!ctx.dir.exists(ctx.dir.ctx, abi.Str.of(index))) return model.loadWeightsSingleFile(gpa, try std.fmt.bufPrint(&path, "{s}/model.safetensors", .{dir}));
    const cap: usize = @intCast(ctx.dir.file_bytes(ctx.dir.ctx, abi.Str.of(index)));
    const buf = try gpa.alloc(u8, cap);
    defer gpa.free(buf);
    var len: usize = 0;
    if (!ctx.dir.read(ctx.dir.ctx, abi.Str.of(index), buf.ptr, cap, &len).isOk()) return error.IndexUnreadable;
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, buf[0..len], .{});
    defer parsed.deinit();
    const map = (if (parsed.value == .object) parsed.value.object.get("weight_map") else null) orelse return error.IndexUnreadable;
    if (map != .object) return error.IndexUnreadable;
    var weights = model.Weights.init(gpa);
    errdefer weights.deinit();
    var seen: std.ArrayList([]const u8) = .empty;
    defer seen.deinit(gpa);
    const s = mlx.weightLoadStream();
    defer _ = mlx.mlx_stream_free(s);
    var it = map.object.iterator();
    outer: while (it.next()) |e| {
        if (!std.mem.startsWith(u8, e.key_ptr.*, prefix) or e.value_ptr.* != .string) continue;
        const shard = e.value_ptr.string;
        for (seen.items) |x| if (std.mem.eql(u8, x, shard)) continue :outer;
        try seen.append(gpa, shard);
        const p = try std.fmt.allocPrintSentinel(gpa, "{s}/{s}", .{ dir, shard }, 0);
        defer gpa.free(p);
        try model.loadSafetensorsFile(gpa, &weights, p, s, .{});
    }
    return weights;
}

pub fn headHooks(
    comptime Loaded: type,
    comptime Head: type,
    comptime loadHead: fn (*const abi.LoadCtx) anyerror!?*Head,
    comptime forwardHead: fn (*const Head, *const Linear, mlx.mlx_stream, *const abi.Forward, *abi.ForwardOut) anyerror!void,
) *const abi.HeadHooks {
    const H = struct {
        fn load(mp: *anyopaque, ctx: *const abi.LoadCtx) callconv(.c) abi.Verdict {
            const m: *Loaded = @ptrCast(@alignCast(mp));
            if (m.head != null) return .ok;
            m.head = (loadHead(ctx) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) }) orelse
                return abi.Verdict.refuse(.unsupported, "the pack ships no MTP head");
            return .ok;
        }
        fn forward(mp: *anyopaque, f: *const abi.Forward, out: *abi.ForwardOut) callconv(.c) abi.Verdict {
            const m: *Loaded = @ptrCast(@alignCast(mp));
            const h = m.head orelse return abi.Verdict.refuse(.failed, "MTP head not loaded");
            forwardHead(h, &m.base.weights.embed.table, m.base.s, f, out) catch |e| return .{ .code = .failed, .why = abi.Str.of(@errorName(e)) };
            return .ok;
        }
        fn slot(_: *anyopaque, _: u32) callconv(.c) abi.Verdict {
            return .ok;
        }
        fn release(_: *anyopaque, _: u32) callconv(.c) void {}
        fn truncate(_: *anyopaque, _: u32, _: u64) callconv(.c) void {}
        fn adopt(_: *anyopaque, _: u32, _: abi.Handle) callconv(.c) abi.Verdict {
            return .ok;
        }
        fn select(_: *anyopaque, _: abi.Array, _: *abi.Array) callconv(.c) abi.Verdict {
            return abi.Verdict.refuse(.unsupported, "the engine selects drafts");
        }
        fn bill(_: *const abi.Parsed) callconv(.c) u64 {
            return 0;
        }
        const table: abi.HeadHooks = .{ .load = &load, .activate = &slot, .release = &release, .forward = &forward, .truncate = &truncate, .adopt = &adopt, .mark = &release, .draft_select = &select, .bill = &bill };
    };
    return &H.table;
}
