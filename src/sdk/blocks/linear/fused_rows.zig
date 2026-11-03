//! sdk/blocks/linear/fused_rows.
//! Owns FusedQkv, fused_qkv_engaged, fused_qkv_override, fused_qkv_env, fusedQkvEnabled, FusedRows, and 7 more.

const std = @import("std");
const env = @import("../../../env.zig");
const mlx = @import("../../../mlx/ffi.zig");
const Weights = @import("../../weights/resident.zig").Weights;
const testing = std.testing;
const transposeBf16Weight = @import("../../../mlx/array.zig").transposeBf16Weight;
const QuantParams = @import("../../quant/params.zig").QuantParams;
const testReadF32 = @import("../../testing/parity.zig").testReadF32;

/// Quantized projections that read the same input, joined along their
/// output rows so one matmul serves them all. `widths[i]` is part i's row
/// count. A null `w` means the group was not built.
pub const FusedRows = struct {
    w: mlx.mlx_array = .{ .ctx = null },
    s: mlx.mlx_array = .{ .ctx = null },
    b: mlx.mlx_array = .{ .ctx = null },
    widths: [4]c_int = @splat(0),
    count: u8 = 0,

    /// Rows [start, start+width) of the last axis of a matmul result.
    pub fn part(self: *const FusedRows, out: *mlx.mlx_array, y: mlx.mlx_array, i: usize, s: mlx.mlx_stream) !void {
        var start: c_int = 0;
        for (self.widths[0..i]) |w| start += w;
        try sliceQkvPart(out, y, start, self.widths[i], s);
    }
};

/// A fused row group serves DECODE width only: at M == 1 every row is the
/// same per-row dot product whatever the output width, while from M == 2 the
/// verify lanes (`vqmmLaneForTile`) and MLX's own split-K pick kernels by N,
/// so the joined matrix would reduce K in another order. Pinned by the
/// `fused row projections` test.
pub const FUSED_ROWS_MAX_M: c_int = 1;

/// Groups joined by the loader in flight (reported once per load).
pub var fused_row_groups: usize = 0;

/// Join up to four quantized `{w, s, b}` triples along axis 0 into one
/// evaluated buffer and rewrite each triple to a zero-copy row VIEW of it,
/// so no weight is resident twice. Null (nothing changed) unless every part
/// shares one packed geometry: same packed width and dtype, same scales
/// width and dtype, same bias presence — the single matmul must dequantize
/// them all the same way.
pub fn fuseRowGroup(parts: []const [3]*mlx.mlx_array, s: mlx.mlx_stream) !?FusedRows {
    if (parts.len < 2 or parts.len > 4) return null;
    const w0 = parts[0][0].*;
    const s0 = parts[0][1].*;
    if (w0.ctx == null or s0.ctx == null) return null;
    const have_b = parts[0][2].ctx != null;
    const wsh0 = mlx.getShape(w0);
    const ssh0 = mlx.getShape(s0);
    if (wsh0.len != 2 or ssh0.len != 2) return null;
    var widths: [4]c_int = @splat(0);
    for (parts, 0..) |p, i| {
        const w = p[0].*;
        const sc = p[1].*;
        if (w.ctx == null or sc.ctx == null) return null;
        if ((p[2].ctx != null) != have_b) return null;
        const wsh = mlx.getShape(w);
        const ssh = mlx.getShape(sc);
        if (wsh.len != 2 or ssh.len != 2) return null;
        if (wsh[1] != wsh0[1] or ssh[1] != ssh0[1] or wsh[0] != ssh[0]) return null;
        if (mlx.mlx_array_dtype(w) != mlx.mlx_array_dtype(w0) or mlx.mlx_array_dtype(sc) != mlx.mlx_array_dtype(s0)) return null;
        if (have_b) {
            const bsh = mlx.getShape(p[2].*);
            if (bsh.len != 2 or bsh[0] != ssh[0] or bsh[1] != ssh[1]) return null;
            if (mlx.mlx_array_dtype(p[2].*) != mlx.mlx_array_dtype(parts[0][2].*)) return null;
        }
        widths[i] = wsh[0];
    }
    const join = struct {
        fn run(ps: []const [3]*mlx.mlx_array, which: usize, str: mlx.mlx_stream) !mlx.mlx_array {
            var arrs: [4]mlx.mlx_array = undefined;
            for (ps, 0..) |p, i| arrs[i] = p[which].*;
            const vec = mlx.mlx_vector_array_new_data(&arrs, ps.len);
            defer _ = mlx.mlx_vector_array_free(vec);
            var joined = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(joined);
            try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 0, str));
            var out = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(out);
            try mlx.check(mlx.mlx_contiguous(&out, joined, false, str));
            try mlx.check(mlx.mlx_array_eval(out));
            return out;
        }
    }.run;
    var fused = FusedRows{ .widths = widths, .count = @intCast(parts.len) };
    fused.w = try join(parts, 0, s);
    errdefer _ = mlx.mlx_array_free(fused.w);
    fused.s = try join(parts, 1, s);
    errdefer _ = mlx.mlx_array_free(fused.s);
    if (have_b) fused.b = try join(parts, 2, s);
    errdefer if (have_b) {
        _ = mlx.mlx_array_free(fused.b);
    };
    // Row views: the map's originals are released with the map, the views
    // keep the joined buffer alive.
    var start: c_int = 0;
    for (parts, 0..) |p, i| {
        const stop = start + widths[i];
        const srcs = [_]mlx.mlx_array{ fused.w, fused.s, fused.b };
        for (srcs, 0..) |src, which| {
            if (src.ctx == null) continue;
            const cols = mlx.getShape(src)[1];
            const lo = [_]c_int{ start, 0 };
            const hi = [_]c_int{ stop, cols };
            const st = [_]c_int{ 1, 1 };
            var view = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&view, src, &lo, 2, &hi, 2, &st, 2, s));
            p[which].* = view;
        }
        start = stop;
    }
    return fused;
}

/// `fuseRowGroup` plus ownership: the joined buffers go to `owned`; each
/// part's original is released — map parts (`names`, from index `name_from`)
/// have their entries swapped for the view, handle parts (Spark's split
/// copies, `parts[0..name_from]`) are dropped from `owned` and their views
/// parked there instead. Returns an empty group when nothing was fused.
pub fn fuseRowsInPlace(
    parts: [][3]*mlx.mlx_array,
    name_from: usize,
    names: []const []const u8,
    weights: *Weights,
    name_buf: *[256]u8,
    prefix: []const u8,
    li: u32,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) !FusedRows {
    var originals: [4][3]mlx.mlx_array = undefined;
    for (parts, 0..) |p, i| originals[i] = .{ p[0].*, p[1].*, p[2].* };
    const leaves = [_][]const u8{ "weight", "scales", "biases" };
    // Only map-owned handles are fused: a part the map does not hold under
    // its name (a materialized copy, another spelling) would stay resident
    // beside the joined buffer.
    for (parts[name_from..], name_from..) |p, i| {
        for (0..3) |which| {
            if (p[which].*.ctx == null) continue;
            const name = std.fmt.bufPrint(name_buf, "{s}.layers.{d}.{s}.{s}", .{ prefix, li, names[i - name_from], leaves[which] }) catch unreachable;
            const held = weights.get(name) orelse return .{};
            if (held.ctx != p[which].*.ctx) return .{};
        }
    }
    const fused = (try fuseRowGroup(parts, s)) orelse return .{};
    fused_row_groups += 1;
    try owned.append(allocator, fused.w);
    try owned.append(allocator, fused.s);
    if (fused.b.ctx != null) try owned.append(allocator, fused.b);
    for (parts, 0..) |p, i| {
        for (0..3) |which| {
            const view = p[which].*;
            if (view.ctx == null) continue;
            if (i < name_from) {
                try owned.append(allocator, view);
                for (owned.items, 0..) |a, j| {
                    if (a.ctx == originals[i][which].ctx) {
                        _ = mlx.mlx_array_free(a);
                        _ = owned.swapRemove(j);
                        break;
                    }
                }
            } else {
                const name = std.fmt.bufPrint(name_buf, "{s}.layers.{d}.{s}.{s}", .{ prefix, li, names[i - name_from], leaves[which] }) catch unreachable;
                weights.replace(name, view);
            }
        }
    }
    return fused;
}


/// Slice `[B, S, total]` back into the three projections. At decode (B*S == 1)
/// every slice is contiguous memory; at prefill they are strided views whose
/// later reshape to [B,S,H,D] splits the stride-1 axis, which MLX expresses as
/// a view rather than a copy.
pub fn sliceQkvPart(out: *mlx.mlx_array, qkv: mlx.mlx_array, start: c_int, width: c_int, s: mlx.mlx_stream) !void {
    const sh = mlx.getShape(qkv);
    var lo: [3]c_int = .{ 0, 0, start };
    var hi: [3]c_int = .{ sh[0], sh[1], start + width };
    const st = [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_slice(out, qkv, &lo, 3, &hi, 3, &st, 3, s));
}

/// A packed bank holds each expert's gate and up projections in one tensor
/// `experts.gate_up_proj` of shape [E, 2*M, X], gate rows first (HF chunks
/// the projected output in halves: gate = [..., :M], up = [..., M:]).
/// Slicing axis 1 maps the packed layout onto MoeMlpWeights' separate
/// switch_gate_*/switch_up_* fields with zero forward-pass changes; the same
/// split serves the packed-u32 weight, its scales, and its biases (all share
/// the [E, rows, X] expert-output layout — quant groups run along axis 2).
/// Caller owns both returned arrays.
/// Join N same-input-width projections into one by stacking their OUTPUT rows.
/// Quantization survives the join untouched: affine groups run along the INPUT
/// axis, so rows stay independent and the scales/biases concatenate on the same
/// axis as the packed codes. Returns an owned array (caller records it in
/// `owned_bf16`); a null-ctx member means "this triple is dense", which the
/// caller resolves before calling.
///
/// Used by KDA, whose checkpoint ships q/k/v as three separate projections and
/// three separate depthwise convs where the forward wants one of each — doing
/// it at load keeps the recurrence's hot path identical to GatedDeltaNet's.
/// Whether a set of same-role projections can be JOINED on the output-row
/// axis. `cols` is each one's trailing dim — the PACKED width for a quantized
/// weight, the input width for a dense one. They must all agree: mlx's
/// concatenate needs identical trailing dims (a mismatch is an MLX-level
/// error, which is uncatchable and takes the process with it), and a joined
/// weight resolves ONE (bits, group_size) from its packed geometry, so a trio
/// quantized at mixed widths would serve two thirds of its rows at the wrong
/// scale even if the concatenate somehow succeeded. Pure so the predicate is
/// testable without building arrays.
pub fn joinableCols(cols: []const c_int) bool {
    if (cols.len == 0) return false;
    for (cols[1..]) |c| {
        if (c != cols[0]) return false;
    }
    return true;
}

/// A projection triple (`scales`/`biases` null-ctx when dense). Borrowed handles.
pub const QW = struct {
    w: mlx.mlx_array = .{ .ctx = null },
    s: mlx.mlx_array = .{ .ctx = null },
    b: mlx.mlx_array = .{ .ctx = null },
};

// ── tests ──

const qmatmulBits = @import("qmatmul.zig").qmatmulBits;

test "fused row projections are bit-identical to the separate matmuls at decode widths" {
    // Bar: for M in 1..FUSED_ROWS_MAX_M the joined matmul's slices equal the
    // separate matmuls byte for byte, and each rewritten VIEW equals its
    // original at a prefill width too (the separate path must not move).
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xF05ED);
    const rnd = prng.random();
    const K: c_int = 2560;
    const Q = struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array };
    const quant = struct {
        fn f(a: std.mem.Allocator, r: std.Random, st: mlx.mlx_stream, rows: c_int, cols: c_int) !Q {
            const buf = try a.alloc(f32, @intCast(rows * cols));
            defer a.free(buf);
            for (buf) |*v| v.* = (r.float(f32) - 0.5) * 0.2;
            const sh = [_]c_int{ rows, cols };
            const w32 = mlx.mlx_array_new_data(buf.ptr, &sh, 2, .float32);
            defer _ = mlx.mlx_array_free(w32);
            var wb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wb);
            try mlx.check(mlx.mlx_astype(&wb, w32, .bfloat16, st));
            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, st));
            var q: Q = undefined;
            q.w = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&q.w, triple, 0));
            q.sc = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&q.sc, triple, 1));
            q.bi = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&q.bi, triple, 2));
            return q;
        }
    }.f;
    const groups = [_][]const c_int{ &.{ 4096, 1024, 1024, 16 }, &.{ 2048, 512, 512 }, &.{ 10240, 10240 }, &.{ 10240, 6144, 48, 48 } };
    for (groups) |widths| {
        var orig: [4]Q = undefined;
        var live: [4]Q = undefined;
        for (widths, 0..) |n, i| {
            orig[i] = try quant(allocator, rnd, s, n, K);
            live[i] = orig[i];
        }
        defer for (orig[0..widths.len]) |q| {
            _ = mlx.mlx_array_free(q.w);
            _ = mlx.mlx_array_free(q.sc);
            _ = mlx.mlx_array_free(q.bi);
        };
        var parts: [4][3]*mlx.mlx_array = undefined;
        for (0..widths.len) |i| parts[i] = .{ &live[i].w, &live[i].sc, &live[i].bi };
        const fused = (try fuseRowGroup(parts[0..widths.len], s)) orelse return error.FusedRowsDeclined;
        defer {
            _ = mlx.mlx_array_free(fused.w);
            _ = mlx.mlx_array_free(fused.s);
            _ = mlx.mlx_array_free(fused.b);
            for (live[0..widths.len]) |q| {
                _ = mlx.mlx_array_free(q.w);
                _ = mlx.mlx_array_free(q.sc);
                _ = mlx.mlx_array_free(q.bi);
            }
        }
        for ([_]c_int{ 1, 2, 3, 5, 8, 64 }) |m| {
            const xb = try allocator.alloc(f32, @intCast(m * K));
            defer allocator.free(xb);
            for (xb) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
            const xsh = [_]c_int{ 1, m, K };
            const x32 = mlx.mlx_array_new_data(xb.ptr, &xsh, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
            var y_all = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(y_all);
            if (m <= FUSED_ROWS_MAX_M) y_all = try qmatmulBits(x, fused.w, fused.s, fused.b, 4, 64, .affine, s);
            for (0..widths.len) |i| {
                const ref = try qmatmulBits(x, orig[i].w, orig[i].sc, orig[i].bi, 4, 64, .affine, s);
                defer _ = mlx.mlx_array_free(ref);
                const via_view = try qmatmulBits(x, live[i].w, live[i].sc, live[i].bi, 4, 64, .affine, s);
                defer _ = mlx.mlx_array_free(via_view);
                const cnt: usize = @intCast(m * widths[i]);
                const h1 = try allocator.alloc(f32, cnt);
                defer allocator.free(h1);
                const h2 = try allocator.alloc(f32, cnt);
                defer allocator.free(h2);
                try testReadF32(ref, h1, s);
                try testReadF32(via_view, h2, s);
                try testing.expectEqualSlices(f32, h1, h2);
                if (m <= FUSED_ROWS_MAX_M) {
                    var part = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(part);
                    try fused.part(&part, y_all, i, s);
                    try testReadF32(part, h2, s);
                    try testing.expectEqualSlices(f32, h1, h2);
                }
            }
        }
    }
}

test "joinableCols: a KDA q/k/v trio joins only when all three are quantized alike" {
    const t = std.testing;
    // Uniform 4-bit: all three pack to the same width.
    try t.expect(joinableCols(&[_]c_int{ 192, 192, 192 }));
    // Dense bf16: the trailing dim is the input width, still uniform.
    try t.expect(joinableCols(&[_]c_int{ 1536, 1536, 1536 }));
    // Bias-less quant: no scales at all, all zeros — uniform.
    try t.expect(joinableCols(&[_]c_int{ 0, 0, 0 }));
    // q at 8-bit beside k/v at 4-bit: mlx's concatenate rejects this at a
    // level we cannot catch, so the loader has to see it first.
    try t.expect(!joinableCols(&[_]c_int{ 384, 192, 192 }));
    try t.expect(!joinableCols(&[_]c_int{ 192, 192, 96 }));
    try t.expect(!joinableCols(&[_]c_int{}));
}
