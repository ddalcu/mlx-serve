//! C2, the routed-expert `quant`: the routed-expert half of PLG's `quant` kind (weight load + the
//! MoE matmul). Given the slot ids and the bank, an MoE layer asks the quant that claimed the
//! weight group for its call's expert matmuls.
//!
//! The arch keeps the routing (gate, top-k, weights), the combine, the shared expert and the
//! choice between decode and prefill width (made on the logical M, a genuine runtime value). C1
//! `expert_source` keeps which slot holds which expert, the bank arrays, the wait gates and the
//! growth.
//!
//! The load path asks every quant `claims(peek)` once per weight group and takes the highest
//! claim. The peek is the group's description: C1's manifest reader for a streamed bank
//! (`expert_bank.peek`), the checkpoint's quantization_config for resident experts. The claimed
//! quant is `accept`ed once per backend (built, and self-checked where it brings kernels), and
//! every entry point of its `Accepted(G)` is prebound there: nothing is looked up or revalidated
//! per call. Clients:
//!   - `exl3_quant`: the streamed EXL3 bank's pinned Metal texts;
//!   - `FromGatherMatmul(Q)`: any gather-matmul quant, e.g. `GatherQmm` (mlx-serve's gather_qmm /
//!     gather_mm path; the V4.1 DSpark head's mxfp4 switch).
//!
//! A quant `Q` declares (`check(Q)` names a missing or mistyped declaration at comptime):
//!   name: []const u8
//!   Arrays(T): type                                   one projection's per-slot arrays
//!   claims(*const BankPeek, ?*Diag) ?Priority         at load, once per weight group
//!   Accepted(G): type                                 the quant accepted on backend G
//!   accept(G, Allocator, *G, Context, Spec, *Diag) !*Accepted(G)   once, before the weights bind
//! and its `Accepted(G)` (`checkAccepted(Q, G)`):
//!   max_decode_rows: u32                              the widest call gateUp / down serve
//!   checkBank(*const, *G, BankArrays(Arrays(G.T)), *Diag) !void    per bank bind and per grow
//!   gateUp(*const, *G, x, slot_ids, gate, up) !G.T   the activation of the gate and up rows
//!   down(*const, *G, h, slot_ids, down) !G.T
//!   prefill(*, *G, layer, x, PrefillRows, BankArrays(..)) !G.T     any width: the call's down rows
//!   finishPrefill(*, *G) !void                        the prefill boundary: drains what is in flight
//!   deinit(*, *G) void

const std = @import("std");
const mlx = @import("sdk").mlx;
const pk = @import("sdk");
const kernel_reg = @import("kernels.zig");
const QuantMode = @import("sdk").QuantMode;

const Allocator = std.mem.Allocator;
const Dtype = mlx.mlx_dtype;

pub const Diag = pk.Diag;

/// How strongly a quant claims a weight group: the load path asks every quant and takes the highest.
pub const Priority = pk.Priority;

/// A weight group's per-expert tensor, its layers, and its description at load (the SDK's `GroupPeek`).
pub const Segment = pk.Segment;
pub const LayerPeek = pk.LayerPeek;
pub const BankPeek = pk.GroupPeek;

/// The routed experts' activation between gate / up and down. It is the arch's: a quant whose
/// texts fuse it declares the one they fuse and refuses the rest at accept.
pub const Activation = union(enum) {
    /// silu(gate) * up
    swiglu,
    /// silu(min(gate, limit)) * clip(up, -limit, limit) (DeepSeek-V4.1's swiglu_limit)
    swiglu_clamped: f64,
};

/// What the arch asks of the quant at accept.
pub const Spec = struct {
    hidden: u32,
    inter: u32,
    top_k: u32,
    n_layers: u32,
    act: Activation,
    /// the MoE input's dtype
    input: Dtype,
};

/// What the load context hands a quant's accept: the kernel set it owns (a quant with pinned
/// texts takes it back as its own registry's set and self-checks its subset there) and the
/// claimed group's description.
pub const Context = struct {
    kernels: ?kernel_reg.SetRef = null,
    peek: ?*const BankPeek = null,
};

/// One prefill call's routed rows: `slot[i]` is routed row i's bank slot; row i reads x row i,
/// or x row `act_row[i]` when given (the chunk's tokens: routed row / top_k).
pub const PrefillRows = struct { slot: []const u32, act_row: ?[]const u32 = null };

/// A layer bank's three projections, each one `Q.Arrays(T)`.
pub fn BankArrays(comptime A: type) type {
    return struct { gate: A, up: A, down: A };
}

pub const Proj = enum { gate, up, down };

/// Refusals of accept (construction time; never a per-call branch).
pub const Refusal = error{
    /// the arch's activation is not the one the quant's texts fuse
    ActivationNotFused,
    /// dims the quant's kernels are not compiled for
    DimsNotImplemented,
    /// an MoE input dtype the quant's kernels do not read
    InputDtype,
    /// decode rows (8 tokens x top_k) wider than the quant's decode tables
    TopKTooWide,
    /// a quant with pinned texts accepted without the load context's kernel set
    NoKernelSet,
    /// a quant whose parameters come from the group's description accepted without it
    NoWeightDescription,
    /// a description the quant does not serve (its claims would have declined it)
    NotClaimed,
    /// bank arrays that are not the kernels' signature
    BankArrays,
};

pub fn refuse(diag: ?*Diag, err: Refusal, comptime fmt: []const u8, args: anytype) Refusal {
    if (diag) |d| d.set(fmt, args);
    return err;
}

/// A declined claim, the first mismatch in `why`.
pub fn decline(why: ?*Diag, comptime fmt: []const u8, args: anytype) ?Priority {
    if (why) |d| d.set(fmt, args);
    return null;
}

/// The string field `name` of a JSON object (null when absent or not a string).
pub fn str(v: std.json.Value, name: []const u8) ?[]const u8 {
    if (v != .object) return null;
    const f = v.object.get(name) orelse return null;
    return if (f == .string) f.string else null;
}

/// The integer field `name` of a JSON object (null when absent or not an integer).
pub fn int(v: std.json.Value, name: []const u8) ?i64 {
    if (v != .object) return null;
    const f = v.object.get(name) orelse return null;
    return if (f == .integer) f.integer else null;
}

/// The object field `name` of a JSON object (`.null` when absent).
pub fn obj(v: std.json.Value, name: []const u8) std.json.Value {
    if (v != .object) return .null;
    return v.object.get(name) orelse .null;
}

// ── The interface checks (comptime) ──

/// Comptime: `Q` declares the quant half of the contract; a compile error names the first
/// declaration missing or mistyped.
pub fn check(comptime Q: type) void {
    comptime {
        for ([_][]const u8{ "name", "Arrays", "claims", "Accepted", "accept" }) |d| {
            if (!@hasDecl(Q, d)) @compileError("quant " ++ @typeName(Q) ++ ": no " ++ d);
        }
        if (@TypeOf(Q.name) != []const u8 and @TypeOf(Q.name) != *const [Q.name.len:0]u8) @compileError("quant " ++ @typeName(Q) ++ ": name is not a string");
        pk.check.fnDecl("quant " ++ @typeName(Q), Q, "claims", &.{ *const BankPeek, ?*Diag }, ?Priority);
    }
}

/// Comptime: `Q.Accepted(G)` declares the call half of the contract on backend G.
pub fn checkAccepted(comptime Q: type, comptime G: type) void {
    comptime {
        check(Q);
        const A = Q.Accepted(G);
        const Ar = Q.Arrays(G.T);
        const Bank = BankArrays(Ar);
        const where = "quant " ++ @typeName(Q) ++ ".Accepted";
        if (!@hasDecl(A, "max_decode_rows") or @TypeOf(A.max_decode_rows) != u32) @compileError(where ++ ": no max_decode_rows: u32");
        pk.check.fnDecl(where, A, "checkBank", &.{ *const A, *G, Bank, *Diag }, void);
        pk.check.fnDecl(where, A, "gateUp", &.{ *const A, *G, G.T, G.T, Ar, Ar }, G.T);
        pk.check.fnDecl(where, A, "down", &.{ *const A, *G, G.T, G.T, Ar }, G.T);
        pk.check.fnDecl(where, A, "prefill", &.{ *A, *G, u32, G.T, PrefillRows, Bank }, G.T);
        pk.check.fnDecl(where, A, "finishPrefill", &.{ *A, *G }, void);
        pk.check.fnDecl(where, A, "deinit", &.{ *A, *G }, void);
        // accept(G, a, g, ctx, spec, diag) !*A: instantiated through a wrapper of the contract's
        // signature, so a mistyped accept fails here, named
        const Wrap = struct {
            fn call(a: Allocator, g: *G, ctx: Context, spec: Spec, diag: *Diag) anyerror!*A {
                return Q.accept(G, a, g, ctx, spec, diag);
            }
        };
        _ = &Wrap.call;
    }
}

// ── The activation on the backend's graph (the adapter's; a fused quant runs its own) ──

/// The backend methods `activation` uses.
pub const activation_methods = [_][]const u8{ "mul", "silu", "clip", "minimum", "scalar", "dtypeOf" };

/// The arch's activation of the gate and up rows as backend ops, in mlx-lm's `SwitchGLU` +
/// `ClampedSwiGLU` order (m1 `deepseek_v41_dspark_head.zig` Resident.routed): up clipped to
/// +-limit and gate capped at limit (the scalars in the operand's dtype), then silu(gate) * up.
pub fn activation(comptime G: type, g: *G, act: Activation, gate: G.T, up: G.T) !G.T {
    switch (act) {
        .swiglu => return g.mul(try g.silu(gate), up),
        .swiglu_clamped => |lim| {
            const du = g.dtypeOf(up);
            const uc = try g.clip(up, try g.scalar(-lim, du), try g.scalar(lim, du));
            const gc = try g.minimum(gate, try g.scalar(lim, g.dtypeOf(gate)));
            return g.mul(try g.silu(gc), uc);
        },
    }
}

// ── FromGatherMatmul: the stock path as a C2 client ──

/// A C2 quant from a gather-matmul quant `Q` (PLG's `gatherMatmul`). This makes the stock path
/// an explicit construction-time client, not a fallback. `Q` declares:
///   name, Arrays(T), claims (as a C2 quant);
///   Params: type, and params(*const BankPeek, Spec, *Diag) !Params (what the description fixes);
///   gatherMatmul(G, *G, *const Params, x [rows, in], w: Arrays(G.T), rhs [rows] u32, sorted: bool)
///       !G.T -> [rows, out];
///   checkArrays(G, *G, *const Params, Proj, Arrays(G.T), in: u32, out: u32, *Diag) !void.
/// The adapter derives the rest:
///   - gateUp: two gathers, then the arch's activation on the backend's graph;
///   - down: one gather;
///   - prefill: the routed rows sorted by slot on the host, the three gathers with sorted indices,
///     the rows put back (mlx-lm's SwitchGLU at 64 routed rows and above);
///   - finishPrefill: nothing is in flight;
///   - checkBank: Q's check per projection;
///   - accept: MLX's own kernels, nothing to self-check.
pub fn FromGatherMatmul(comptime Q: type) type {
    return struct {
        pub const name = Q.name;
        pub const Arrays = Q.Arrays;
        pub const claims = Q.claims;
        /// mlx-lm's SwitchGLU sorts the routed rows from 64 on (`do_sort = indices.size >= 64`)
        pub const sort_rows = 64;

        pub fn Accepted(comptime G: type) type {
            comptime {
                for (activation_methods ++ [_][]const u8{ "take", "hostArray", "reshape", "shapeOf" }) |m| {
                    if (!@hasDecl(G, m)) @compileError("quant " ++ Q.name ++ ": the backend " ++ @typeName(G) ++ " lacks " ++ m);
                }
            }
            return struct {
                const Self = @This();
                const A = Q.Arrays(G.T);
                pub const max_decode_rows: u32 = sort_rows - 1;
                a: Allocator,
                params: Q.Params,
                spec: Spec,
                // host scratch of prefill, reused across calls
                order: std.ArrayList(u32) = .empty,
                counts: std.ArrayList(u32) = .empty,
                ids: std.ArrayList(u32) = .empty,
                src: std.ArrayList(u32) = .empty,
                inv: std.ArrayList(u32) = .empty,

                pub fn checkBank(self: *const Self, g: *G, bank: BankArrays(A), diag: *Diag) !void {
                    const s = &self.spec;
                    try Q.checkArrays(G, g, &self.params, .gate, bank.gate, s.hidden, s.inter, diag);
                    try Q.checkArrays(G, g, &self.params, .up, bank.up, s.hidden, s.inter, diag);
                    try Q.checkArrays(G, g, &self.params, .down, bank.down, s.inter, s.hidden, diag);
                }

                /// x [rows, hidden], slot_ids u32 [rows] -> the activation [rows, inter].
                pub fn gateUp(self: *const Self, g: *G, x: G.T, slot_ids: G.T, gate: A, up: A) !G.T {
                    const gg = try Q.gatherMatmul(G, g, &self.params, x, gate, slot_ids, false);
                    const uu = try Q.gatherMatmul(G, g, &self.params, x, up, slot_ids, false);
                    return activation(G, g, self.spec.act, gg, uu);
                }

                /// h [rows, inter], slot_ids u32 [rows] -> [rows, hidden].
                pub fn down(self: *const Self, g: *G, h: G.T, slot_ids: G.T, d: A) !G.T {
                    return Q.gatherMatmul(G, g, &self.params, h, d, slot_ids, false);
                }

                /// x [tokens or rows, hidden], the call's routed rows -> [rows, hidden] in routed-row order.
                pub fn prefill(self: *Self, g: *G, layer: u32, x: G.T, rows: PrefillRows, bank: BankArrays(A)) !G.T {
                    _ = layer;
                    const n = rows.slot.len;
                    try self.order.resize(self.a, n);
                    try self.ids.resize(self.a, n);
                    try self.src.resize(self.a, n);
                    try self.inv.resize(self.a, n);
                    // Stable by slot: equal slots keep their routed order (mlx-lm's argsort of the flat indices), as a
                    // counting sort over the slots' range.
                    const max_slot = if (n == 0) 0 else std.mem.max(u32, rows.slot);
                    try self.counts.resize(self.a, max_slot + 2);
                    @memset(self.counts.items, 0);
                    for (rows.slot) |sl| self.counts.items[sl + 1] += 1;
                    for (1..self.counts.items.len) |i| self.counts.items[i] += self.counts.items[i - 1];
                    for (rows.slot, 0..) |sl, i| {
                        self.order.items[self.counts.items[sl]] = @intCast(i);
                        self.counts.items[sl] += 1;
                    }
                    for (self.order.items, 0..) |o, i| {
                        self.ids.items[i] = rows.slot[o];
                        self.src.items[i] = if (rows.act_row) |ar| ar[o] else o;
                        self.inv.items[o] = @intCast(i);
                    }
                    const n_c: c_int = @intCast(n);
                    const src = try g.hostArray(std.mem.sliceAsBytes(self.src.items), &.{n_c}, .uint32);
                    const rhs = try g.hostArray(std.mem.sliceAsBytes(self.ids.items), &.{n_c}, .uint32);
                    const xs = try g.take(x, src, 0);
                    const gg = try Q.gatherMatmul(G, g, &self.params, xs, bank.gate, rhs, true);
                    const uu = try Q.gatherMatmul(G, g, &self.params, xs, bank.up, rhs, true);
                    const h = try activation(G, g, self.spec.act, gg, uu);
                    const y = try Q.gatherMatmul(G, g, &self.params, h, bank.down, rhs, true);
                    return g.take(y, try g.hostArray(std.mem.sliceAsBytes(self.inv.items), &.{n_c}, .uint32), 0);
                }

                /// Nothing is left in flight (every call's graph is the caller's).
                pub fn finishPrefill(self: *Self, g: *G) !void {
                    _ = self;
                    _ = g;
                }

                pub fn deinit(self: *Self, g: *G) void {
                    _ = g;
                    inline for (.{ &self.order, &self.counts, &self.ids, &self.src, &self.inv }) |l| l.deinit(self.a);
                    self.a.destroy(self);
                }
            };
        }

        /// Q's parameters from the claimed group's description (refused: none given, or one Q
        /// does not serve). No kernels of its own: nothing to self-check.
        pub fn accept(comptime G: type, a: Allocator, g: *G, ctx: Context, spec: Spec, diag: *Diag) !*Accepted(G) {
            _ = g;
            const peek = ctx.peek orelse return refuse(diag, error.NoWeightDescription, "quant {s}: accepted without the group's description", .{Q.name});
            const p = try Q.params(peek, spec, diag);
            const acc = try a.create(Accepted(G));
            acc.* = .{ .a = a, .params = p, .spec = spec };
            return acc;
        }
    };
}

// ── GatherQmm: mlx-serve's built-in routed-expert quant ──

/// mlx-serve's built-in routed-expert quant as a C2 client: the body of `transformer.zig`
/// gatherExpertMm.
///   - MLX's gather_qmm for affine / nvfp4 / mxfp4 / mxfp8 experts: bits and group size from the
///     group's quantization_config; affine carries biases.
///   - MLX's gather_mm for dense experts, with sorted indices forced off (mlx's dense gather_mm is
///     wrong with them: transformer.zig:40641).
/// The backend runs it:
///     gatherMatmul(x [rows, 1, in], w, scales: ?T, biases: ?T, rhs [rows] u32, bits, group_size,
///                  mode: ?QuantMode, sorted) !T -> [rows, 1, out]
/// (the model lane adds it to MlxOps over mlx_gather_qmm / mlx_gather_mm). V4.1's DSpark head
/// (`mtp.0.ffn.experts.*`: mxfp4, group 32) is its client.
pub const GatherQmm = struct {
    pub const name = "mlx-gather-qmm";

    /// w: u32 [slots, out, in * bits / 32] (dense: [slots, out, in]); scales [slots, out, in /
    /// group_size]; biases (affine) as scales.
    pub fn Arrays(comptime T: type) type {
        return struct { w: T, scales: ?T = null, biases: ?T = null };
    }

    /// mode null = dense experts (gather_mm).
    pub const Params = struct { mode: ?QuantMode, bits: u32, group_size: u32 };

    /// The mode's (bits, group sizes) MLX 0.32.2 quantizes.
    fn valid(mode: QuantMode, bits: u32, group: u32) bool {
        return switch (mode) {
            .affine => (bits >= 2 and bits <= 8 and bits != 7) and (group == 32 or group == 64 or group == 128),
            .mxfp4 => bits == 4 and group == 32,
            .nvfp4 => bits == 4 and group == 16,
            .mxfp8 => bits == 8 and group == 32,
            // Raw ggml blocks are no MLX quantization.
            .gguf => false,
        };
    }

    fn parse(peek: *const BankPeek, why: ?*Diag) ?Params {
        const q = peek.quantization;
        if (q == .null) return .{ .mode = null, .bits = 16, .group_size = 0 };
        if (q != .object) {
            _ = decline(why, "quant mlx-gather-qmm: quantization is not an object", .{});
            return null;
        }
        const mode_name = str(q, "mode") orelse "affine";
        const mode = QuantMode.fromString(mode_name) orelse {
            _ = decline(why, "quant mlx-gather-qmm: quantization.mode \"{s}\" is not an MLX quantization mode", .{mode_name});
            return null;
        };
        const bits = int(q, "bits") orelse {
            _ = decline(why, "quant mlx-gather-qmm: quantization.bits missing", .{});
            return null;
        };
        const group = int(q, "group_size") orelse {
            _ = decline(why, "quant mlx-gather-qmm: quantization.group_size missing", .{});
            return null;
        };
        if (bits < 0 or group < 0 or !valid(mode, @intCast(bits), @intCast(group))) {
            _ = decline(why, "quant mlx-gather-qmm: {t} at {d} bits, group {d} is not an MLX quantization", .{ mode, bits, group });
            return null;
        }
        return .{ .mode = mode, .bits = @intCast(bits), .group_size = @intCast(group) };
    }

    pub fn claims(peek: *const BankPeek, why: ?*Diag) ?Priority {
        _ = parse(peek, why) orelse return null;
        return .generic;
    }

    pub fn params(peek: *const BankPeek, spec: Spec, diag: *Diag) !Params {
        _ = spec;
        return parse(peek, diag) orelse error.NotClaimed;
    }

    /// x [rows, in], rhs u32 [rows] (each row's slot) -> [rows, out].
    pub fn gatherMatmul(comptime G: type, g: *G, p: *const Params, x: G.T, w: Arrays(G.T), rhs: G.T, sorted: bool) !G.T {
        const sx = g.shapeOf(x);
        const rows = sx.d[0];
        const x3 = try g.reshape(x, &.{ rows, 1, sx.d[1] });
        const y = try g.gatherMatmul(x3, w.w, w.scales, w.biases, rhs, p.bits, p.group_size, p.mode, sorted and p.mode != null);
        const sy = g.shapeOf(y);
        return g.reshape(y, &.{ rows, sy.d[sy.n - 1] });
    }

    /// A projection's per-slot arrays against the params and the projection's (in, out).
    pub fn checkArrays(comptime G: type, g: *G, p: *const Params, which: Proj, w: Arrays(G.T), in: u32, out: u32, diag: *Diag) !void {
        const sw = g.shapeOf(w.w);
        const packed_in: c_int = if (p.mode != null) @intCast(in * p.bits / 32) else @intCast(in);
        if (sw.n != 3 or sw.d[1] != @as(c_int, @intCast(out)) or sw.d[2] != packed_in)
            return refuse(diag, error.BankArrays, "quant mlx-gather-qmm: {t} w is {any}, the params read [slots, {d}, {d}]", .{ which, sw.slice(), out, packed_in });
        if (p.mode) |mode| {
            const sc = w.scales orelse return refuse(diag, error.BankArrays, "quant mlx-gather-qmm: {t} has no scales", .{which});
            const ss = g.shapeOf(sc);
            if (ss.n != 3 or ss.d[0] != sw.d[0] or ss.d[1] != sw.d[1] or ss.d[2] != @as(c_int, @intCast(in / p.group_size)))
                return refuse(diag, error.BankArrays, "quant mlx-gather-qmm: {t} scales are {any}, want [slots, {d}, {d}]", .{ which, ss.slice(), out, in / p.group_size });
            if (mode.hasBiases() != (w.biases != null))
                return refuse(diag, error.BankArrays, "quant mlx-gather-qmm: {t} biases {s} for {t}", .{ which, if (w.biases != null) "given" else "missing", mode });
        }
    }
};

const testing = std.testing;

/// A host backend for the adapter's tests: nodes of shape and dtype, and every graph op in call order. No MLX.
const HostOps = struct {
    pub const T = u32;
    pub const Shape = struct {
        d: [4]c_int = @splat(0),
        n: usize = 0,
        pub fn slice(s: *const Shape) []const c_int {
            return s.d[0..s.n];
        }
    };
    const Node = struct { shape: Shape, dtype: Dtype, words: []const u32 = &.{} };
    pub const Op = struct { kind: []const u8, f: f64 = 0, dtype: Dtype = .float32, sorted: bool = false, mode: ?QuantMode = null };

    a: Allocator,
    nodes: std.ArrayList(Node) = .empty,
    ops: std.ArrayList(Op) = .empty,

    fn deinit(g: *HostOps) void {
        for (g.nodes.items) |n| g.a.free(n.words);
        g.nodes.deinit(g.a);
        g.ops.deinit(g.a);
    }
    fn node(g: *HostOps, shape: []const c_int, dt: Dtype) !T {
        var s: Shape = .{ .n = shape.len };
        @memcpy(s.d[0..shape.len], shape);
        try g.nodes.append(g.a, .{ .shape = s, .dtype = dt });
        return @intCast(g.nodes.items.len - 1);
    }
    fn op(g: *HostOps, o: Op, shape: []const c_int, dt: Dtype) !T {
        try g.ops.append(g.a, o);
        return g.node(shape, dt);
    }
    pub fn shapeOf(g: *HostOps, x: T) Shape {
        return g.nodes.items[x].shape;
    }
    pub fn dtypeOf(g: *HostOps, x: T) Dtype {
        return g.nodes.items[x].dtype;
    }
    pub fn mul(g: *HostOps, x: T, y: T) !T {
        return g.op(.{ .kind = "mul" }, g.shapeOf(x).slice(), g.dtypeOf(y));
    }
    pub fn silu(g: *HostOps, x: T) !T {
        return g.op(.{ .kind = "silu" }, g.shapeOf(x).slice(), g.dtypeOf(x));
    }
    pub fn clip(g: *HostOps, x: T, lo: T, hi: T) !T {
        _ = .{ lo, hi };
        return g.op(.{ .kind = "clip" }, g.shapeOf(x).slice(), g.dtypeOf(x));
    }
    pub fn minimum(g: *HostOps, x: T, y: T) !T {
        _ = y;
        return g.op(.{ .kind = "minimum" }, g.shapeOf(x).slice(), g.dtypeOf(x));
    }
    pub fn scalar(g: *HostOps, v: f64, dt: Dtype) !T {
        return g.op(.{ .kind = "scalar", .f = v, .dtype = dt }, &.{}, dt);
    }
    pub fn take(g: *HostOps, x: T, idx: T, axis: c_int) !T {
        std.debug.assert(axis == 0);
        var s = g.shapeOf(x);
        s.d[0] = g.shapeOf(idx).d[0];
        return g.op(.{ .kind = "take" }, s.slice(), g.dtypeOf(x));
    }
    pub fn hostArray(g: *HostOps, bytes: []const u8, shape: []const c_int, dt: Dtype) !T {
        const x = try g.node(shape, dt);
        const words = try g.a.alloc(u32, bytes.len / 4);
        @memcpy(std.mem.sliceAsBytes(words), bytes);
        g.nodes.items[x].words = words;
        return x;
    }
    pub fn reshape(g: *HostOps, x: T, shape: []const c_int) !T {
        return g.node(shape, g.dtypeOf(x));
    }
    pub fn gatherMatmul(g: *HostOps, x: T, w: T, scales: ?T, biases: ?T, rhs: T, bits: u32, group: u32, mode: ?QuantMode, sorted: bool) !T {
        _ = .{ scales, biases, rhs, bits, group };
        return g.op(.{ .kind = "gather", .sorted = sorted, .mode = mode }, &.{ g.shapeOf(x).d[0], 1, g.shapeOf(w).d[1] }, g.dtypeOf(x));
    }
    fn kinds(g: *const HostOps, from: usize, buf: []u8) []const u8 {
        var w: std.Io.Writer = .fixed(buf);
        for (g.ops.items[from..]) |o| w.print("{s} ", .{o.kind}) catch break;
        return w.buffered();
    }
};

comptime {
    if (@import("builtin").is_test) checkAccepted(FromGatherMatmul(GatherQmm), HostOps);
}

fn groupOf(arena: Allocator, quantization: []const u8) !BankPeek {
    const q = try std.json.parseFromSliceLeaky(std.json.Value, arena, quantization, .{});
    return .{ .quantization = q, .hidden = 64, .inter = 32, .n_experts = 4, .n_layers = 1, .layers = &.{} };
}

test "sdk quant: a refusal and a decline write their reason when asked and return their verdict" {
    var d: Diag = .{};
    try testing.expectEqual(error.TopKTooWide, refuse(&d, error.TopKTooWide, "top_k {d}", .{9}));
    try testing.expectEqualStrings("top_k 9", d.message());
    try testing.expectEqual(error.BankArrays, refuse(null, error.BankArrays, "unused {d}", .{1}));
    try testing.expectEqual(@as(?Priority, null), decline(&d, "{s}", .{"declined"}));
    try testing.expectEqualStrings("declined", d.message());
    try testing.expectEqual(@as(?Priority, null), decline(null, "x", .{}));
}

test "sdk quant: the JSON readers return null (or .null) for a missing field, a mistyped field and a non-object" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const v = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), "{\"mode\":\"mxfp4\",\"bits\":4,\"o\":{\"k\":1},\"f\":4.5}", .{});
    try testing.expectEqualStrings("mxfp4", str(v, "mode").?);
    try testing.expect(str(v, "bits") == null and str(v, "absent") == null);
    try testing.expectEqual(@as(?i64, 4), int(v, "bits"));
    try testing.expect(int(v, "f") == null and int(v, "mode") == null);
    try testing.expectEqual(@as(i64, 1), obj(v, "o").object.get("k").?.integer);
    try testing.expect(obj(v, "absent") == .null);
    const arr = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), "[1]", .{});
    try testing.expect(str(arr, "mode") == null and int(arr, "bits") == null and obj(arr, "o") == .null);
}

test "sdk quant: the gather quant claims MLX's own quantizations and dense experts, generically; each decline names its field" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const Case = struct { q: []const u8, want: ?Priority, why: []const u8 = "" };
    const cases = [_]Case{
        .{ .q = "null", .want = .generic },
        .{ .q = "{\"bits\":4,\"group_size\":64}", .want = .generic },
        .{ .q = "{\"mode\":\"mxfp4\",\"bits\":4,\"group_size\":32}", .want = .generic },
        .{ .q = "{\"mode\":\"nvfp4\",\"bits\":4,\"group_size\":16}", .want = .generic },
        .{ .q = "{\"mode\":\"mxfp8\",\"bits\":8,\"group_size\":32}", .want = .generic },
        .{ .q = "4", .want = null, .why = "quantization is not an object" },
        .{ .q = "{\"mode\":\"int3\",\"bits\":3,\"group_size\":32}", .want = null, .why = "quantization.mode \"int3\" is not an MLX quantization mode" },
        .{ .q = "{\"mode\":\"affine\",\"group_size\":32}", .want = null, .why = "quantization.bits missing" },
        .{ .q = "{\"mode\":\"affine\",\"bits\":4.0,\"group_size\":32}", .want = null, .why = "quantization.bits missing" },
        .{ .q = "{\"mode\":\"affine\",\"bits\":4}", .want = null, .why = "quantization.group_size missing" },
        .{ .q = "{\"mode\":\"affine\",\"bits\":-4,\"group_size\":32}", .want = null, .why = "affine at -4 bits, group 32 is not an MLX quantization" },
        .{ .q = "{\"mode\":\"affine\",\"bits\":4,\"group_size\":-32}", .want = null, .why = "group -32" },
        .{ .q = "{\"mode\":\"affine\",\"bits\":7,\"group_size\":64}", .want = null, .why = "affine at 7 bits" },
        .{ .q = "{\"mode\":\"affine\",\"bits\":1,\"group_size\":64}", .want = null, .why = "affine at 1 bits" },
        .{ .q = "{\"mode\":\"affine\",\"bits\":9,\"group_size\":64}", .want = null, .why = "affine at 9 bits" },
        .{ .q = "{\"mode\":\"affine\",\"bits\":4,\"group_size\":16}", .want = null, .why = "group 16" },
        .{ .q = "{\"mode\":\"mxfp4\",\"bits\":4,\"group_size\":64}", .want = null, .why = "mxfp4 at 4 bits, group 64" },
        .{ .q = "{\"mode\":\"nvfp4\",\"bits\":4,\"group_size\":32}", .want = null, .why = "nvfp4 at 4 bits, group 32" },
        .{ .q = "{\"mode\":\"mxfp8\",\"bits\":4,\"group_size\":32}", .want = null, .why = "mxfp8 at 4 bits" },
        .{ .q = "{\"mode\":\"gguf\",\"bits\":4,\"group_size\":32}", .want = null, .why = "gguf at 4 bits" },
    };
    for (cases) |c| {
        const g = try groupOf(a, c.q);
        var why: Diag = .{};
        testing.expectEqual(c.want, GatherQmm.claims(&g, &why)) catch |e| {
            std.debug.print("claims {s}: {s}\n", .{ c.q, why.message() });
            return e;
        };
        if (c.want == null) try testing.expect(std.mem.indexOf(u8, why.message(), c.why) != null);
    }
    // every affine width MLX quantizes, at every affine group
    for ([_]u32{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]u32{ 32, 64, 128 }) |group| {
        try testing.expect(GatherQmm.valid(.affine, bits, group));
    };
}

test "sdk quant: the gather quant's params are what its claim read; a declined description is NotClaimed at accept" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const spec: Spec = .{ .hidden = 64, .inter = 32, .top_k = 2, .n_layers = 1, .act = .swiglu, .input = .bfloat16 };
    var d: Diag = .{};
    try testing.expectEqual(GatherQmm.Params{ .mode = null, .bits = 16, .group_size = 0 }, try GatherQmm.params(&try groupOf(a, "null"), spec, &d));
    try testing.expectEqual(GatherQmm.Params{ .mode = .affine, .bits = 6, .group_size = 128 }, try GatherQmm.params(&try groupOf(a, "{\"bits\":6,\"group_size\":128}"), spec, &d));
    try testing.expectError(error.NotClaimed, GatherQmm.params(&try groupOf(a, "{\"mode\":\"mxfp4\",\"bits\":8,\"group_size\":32}"), spec, &d));
    try testing.expect(std.mem.indexOf(u8, d.message(), "mxfp4 at 8 bits") != null);

    var g: HostOps = .{ .a = testing.allocator };
    defer g.deinit();
    const Q = FromGatherMatmul(GatherQmm);
    try testing.expectError(error.NoWeightDescription, Q.accept(HostOps, testing.allocator, &g, .{}, spec, &d));
    try testing.expectEqualStrings("quant mlx-gather-qmm: accepted without the group's description", d.message());
    const declined = try groupOf(a, "{\"mode\":\"gguf\",\"bits\":4,\"group_size\":32}");
    try testing.expectError(error.NotClaimed, Q.accept(HostOps, testing.allocator, &g, .{ .peek = &declined }, spec, &d));
    try testing.expectEqual(@as(u32, 63), Q.Accepted(HostOps).max_decode_rows);
}

test "sdk quant: a projection's arrays are checked against the params, by projection, and the biases follow the mode" {
    var g: HostOps = .{ .a = testing.allocator };
    defer g.deinit();
    var d: Diag = .{};
    const A = GatherQmm.Arrays(u32);
    const mx4: GatherQmm.Params = .{ .mode = .mxfp4, .bits = 4, .group_size = 32 };
    const aff: GatherQmm.Params = .{ .mode = .affine, .bits = 4, .group_size = 64 };
    const dense: GatherQmm.Params = .{ .mode = null, .bits = 16, .group_size = 0 };
    // gate: in 128 -> out 64; packed in = 128 * 4 / 32 = 16 words; scales in / group
    const w = try g.node(&.{ 4, 64, 16 }, .uint32);
    const s32 = try g.node(&.{ 4, 64, 4 }, .uint8);
    const s64 = try g.node(&.{ 4, 64, 2 }, .bfloat16);
    try GatherQmm.checkArrays(HostOps, &g, &mx4, .gate, A{ .w = w, .scales = s32 }, 128, 64, &d);
    try GatherQmm.checkArrays(HostOps, &g, &aff, .gate, A{ .w = w, .scales = s64, .biases = s64 }, 128, 64, &d);
    try GatherQmm.checkArrays(HostOps, &g, &dense, .down, A{ .w = try g.node(&.{ 4, 64, 128 }, .bfloat16) }, 128, 64, &d);
    const Bad = struct { p: *const GatherQmm.Params, arrays: A, why: []const u8 };
    const bad = [_]Bad{
        .{ .p = &mx4, .arrays = .{ .w = try g.node(&.{ 64, 16 }, .uint32), .scales = s32 }, .why = "gate w is { 64, 16 }" },
        .{ .p = &mx4, .arrays = .{ .w = try g.node(&.{ 4, 32, 16 }, .uint32), .scales = s32 }, .why = "the params read [slots, 64, 16]" },
        .{ .p = &mx4, .arrays = .{ .w = try g.node(&.{ 4, 64, 32 }, .uint32), .scales = s32 }, .why = "gate w is" },
        .{ .p = &dense, .arrays = .{ .w = w }, .why = "the params read [slots, 64, 128]" },
        .{ .p = &mx4, .arrays = .{ .w = w }, .why = "gate has no scales" },
        .{ .p = &mx4, .arrays = .{ .w = w, .scales = try g.node(&.{ 4, 64 }, .uint8) }, .why = "gate scales are" },
        .{ .p = &mx4, .arrays = .{ .w = w, .scales = try g.node(&.{ 3, 64, 4 }, .uint8) }, .why = "want [slots, 64, 4]" },
        .{ .p = &mx4, .arrays = .{ .w = w, .scales = try g.node(&.{ 4, 63, 4 }, .uint8) }, .why = "gate scales are" },
        .{ .p = &mx4, .arrays = .{ .w = w, .scales = try g.node(&.{ 4, 64, 8 }, .uint8) }, .why = "gate scales are" },
        .{ .p = &mx4, .arrays = .{ .w = w, .scales = s32, .biases = s32 }, .why = "gate biases given for mxfp4" },
        .{ .p = &aff, .arrays = .{ .w = w, .scales = s64 }, .why = "gate biases missing for affine" },
    };
    for (bad) |b| {
        try testing.expectError(error.BankArrays, GatherQmm.checkArrays(HostOps, &g, b.p, .gate, b.arrays, 128, 64, &d));
        testing.expect(std.mem.indexOf(u8, d.message(), b.why) != null) catch |e| {
            std.debug.print("want \"{s}\" in \"{s}\"\n", .{ b.why, d.message() });
            return e;
        };
    }
}

test "sdk quant: the activation runs in mlx-lm's order, its limit scalars in each operand's dtype" {
    var g: HostOps = .{ .a = testing.allocator };
    defer g.deinit();
    const gate = try g.node(&.{ 2, 8 }, .bfloat16);
    const up = try g.node(&.{ 2, 8 }, .float16);
    var buf: [256]u8 = undefined;
    const plain = try activation(HostOps, &g, .swiglu, gate, up);
    try testing.expectEqualStrings("silu mul ", g.kinds(0, &buf));
    try testing.expectEqualSlices(c_int, &.{ 2, 8 }, g.shapeOf(plain).slice());
    const from = g.ops.items.len;
    _ = try activation(HostOps, &g, .{ .swiglu_clamped = 7.0 }, gate, up);
    try testing.expectEqualStrings("scalar scalar clip scalar minimum silu mul ", g.kinds(from, &buf));
    const o = g.ops.items[from..];
    try testing.expect(o[0].f == -7.0 and o[0].dtype == .float16 and o[1].f == 7.0 and o[1].dtype == .float16);
    try testing.expect(o[3].f == 7.0 and o[3].dtype == .bfloat16);
}

test "sdk quant: the gather adapter's prefill sorts the routed rows stably by slot, gathers sorted, and restores routed order; dense experts never sort" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var g: HostOps = .{ .a = testing.allocator };
    defer g.deinit();
    var d: Diag = .{};
    const Q = FromGatherMatmul(GatherQmm);
    const spec: Spec = .{ .hidden = 64, .inter = 32, .top_k = 2, .n_layers = 1, .act = .swiglu, .input = .bfloat16 };
    const mx = try groupOf(arena.allocator(), "{\"mode\":\"mxfp4\",\"bits\":4,\"group_size\":32}");
    const acc = try Q.accept(HostOps, testing.allocator, &g, .{ .peek = &mx }, spec, &d);
    defer acc.deinit(&g);
    const bank: BankArrays(GatherQmm.Arrays(u32)) = .{
        .gate = .{ .w = try g.node(&.{ 4, 32, 8 }, .uint32) },
        .up = .{ .w = try g.node(&.{ 4, 32, 8 }, .uint32) },
        .down = .{ .w = try g.node(&.{ 4, 64, 4 }, .uint32) },
    };
    // the bank is checked per projection against the spec's (hidden 64, inter 32): down reads [slots, 64, 32 * 4 / 32]
    const scaled = struct {
        fn of(gg: *HostOps, out: c_int, in: c_int) !GatherQmm.Arrays(u32) {
            return .{ .w = try gg.node(&.{ 4, out, @divExact(in * 4, 32) }, .uint32), .scales = try gg.node(&.{ 4, out, @divExact(in, 32) }, .uint8) };
        }
    };
    try acc.checkBank(&g, .{ .gate = try scaled.of(&g, 32, 64), .up = try scaled.of(&g, 32, 64), .down = try scaled.of(&g, 64, 32) }, &d);
    try testing.expectError(error.BankArrays, acc.checkBank(&g, .{ .gate = try scaled.of(&g, 32, 64), .up = try scaled.of(&g, 32, 64), .down = try scaled.of(&g, 32, 64) }, &d));
    try testing.expect(std.mem.startsWith(u8, d.message(), "quant mlx-gather-qmm: down w is"));
    // no act_row: routed row i reads x row i
    const x = try g.node(&.{ 5, 64 }, .bfloat16);
    const from = g.ops.items.len;
    const y = try acc.prefill(&g, 0, x, .{ .slot = &.{ 3, 1, 3, 0, 1 } }, bank);
    try testing.expectEqualSlices(u32, &.{ 3, 1, 4, 0, 2 }, acc.order.items);
    try testing.expectEqualSlices(u32, &.{ 0, 1, 1, 3, 3 }, acc.ids.items);
    try testing.expectEqualSlices(u32, acc.order.items, acc.src.items);
    for (acc.order.items, 0..) |o, i| try testing.expectEqual(@as(u32, @intCast(i)), acc.inv.items[o]);
    try testing.expectEqualSlices(c_int, &.{ 5, 64 }, g.shapeOf(y).slice());
    var buf: [256]u8 = undefined;
    try testing.expectEqualStrings("take gather gather silu mul gather take ", g.kinds(from, &buf));
    for (g.ops.items[from..]) |o| if (std.mem.eql(u8, o.kind, "gather")) try testing.expect(o.sorted and o.mode.? == .mxfp4);
    // decode width: no sort
    const ids = try g.node(&.{2}, .uint32);
    const from2 = g.ops.items.len;
    _ = try acc.down(&g, try acc.gateUp(&g, try g.node(&.{ 2, 64 }, .bfloat16), ids, bank.gate, bank.up), ids, bank.down);
    try testing.expectEqualStrings("gather gather silu mul gather ", g.kinds(from2, &buf));
    for (g.ops.items[from2..]) |o| if (std.mem.eql(u8, o.kind, "gather")) try testing.expect(!o.sorted);
    try acc.finishPrefill(&g);

    // dense experts: MLX's dense gather_mm is wrong with sorted indices, so the prefill passes them unsorted
    const dense = try groupOf(arena.allocator(), "null");
    const dacc = try Q.accept(HostOps, testing.allocator, &g, .{ .peek = &dense }, spec, &d);
    defer dacc.deinit(&g);
    const from3 = g.ops.items.len;
    _ = try dacc.prefill(&g, 0, x, .{ .slot = &.{ 2, 0 }, .act_row = &.{ 4, 1 } }, bank);
    try testing.expectEqualSlices(u32, &.{ 1, 4 }, dacc.src.items);
    for (g.ops.items[from3..]) |o| if (std.mem.eql(u8, o.kind, "gather")) try testing.expect(!o.sorted and o.mode == null);
}
