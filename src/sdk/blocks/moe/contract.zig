//! The moe area's kernel contract. The decode gathers that read an expert bank in place: `gather_qmv` (one
//! projection, shared or per-expert x, optional ReLU^2) is held to float64. The fused gate+up+SwiGLU (solo and rows)
//! replaces two gathers and MLX's SwiGLU chain bit for bit; the rows down+reduce replaces the solo one per row; the
//! solo down+reduce (8 or 16 lanes a row, a bf16 sum over the slots) is held to float64 with its rounding chain priced.
//! Listed in `kernel_contract/registry.zig`. Shapes: rows are top-k slots (solo kernels) or batch rows (rows kernels,
//! top-k 8); a bank holds 8 experts (16 and 24 for Nemotron's wider top-k), N is cut to a few hundred rows (a runtime
//! argument). NaN enters through x where an x row feeds only some outputs, else through the bank's scales: affine
//! scales take NaN and Inf, an NVFP4 scale only e4m3's NaN code 0x7f.
//! The rows down+reduce is compared with the solo kernel through its declared launch: the shipped `gatherQmvDownReduce`
//! and `gatherQmvDownReduceRows` cache their config without K (KP is a template argument), so a second K at the same
//! (top-k, N, bits, group, dtype) reuses the first K's KP (reported; not a gate's finding).
//! `moe_fp4`'s two dispatches (gate+up+SwiGLU, weighted down with the gated shared expert) are held to float64, the
//! SwiGLU and the scales' roundings priced. The fused router replaces its routing chains: the same expert set and,
//! per expert, the chain's weight bits (the shared-expert gate's logit, which no chain computes, to float64); its NaN
//! rule is stated, and gate 4 is not asked of it (compares, and outputs holding uint32 ids). The verify kernels
//! replace MLX's: gather_qmm row by row (`verify_expert_reuse`, N cut to 256 from 2560), the unsort-weight-sum (`verify_expert_reduce`, summing in bf16 as MLX does), the stable argsort
//! (`verify_route_pack`, `prefill_route_scatter`); the prefill histogram and its scan are held exact.
//! The NAX sorted gather (prefill; NAX GPUs only, no pre-NAX skip) replaces MLX's sorted gather_qmm bit for bit at
//! one shape per plan, its mapped arm the row take first, its paired arm also MLX's SwiGLU; its tile scan is held
//! exact. A mapped launch under 8 routed rows does not compile (MLX binds the row map in `constant`); the wrapper
//! takes 16 rows or more, so the registration's mapped rows are tokens of four routed rows.
//! `ubench_stream_wide` (a read-ceiling probe, not served) is held exact as integers.
//! Known issues: e4m3_nan_code (the NVFP4 kernels whose x feeds every output), act2_drops_nan, limit_drops_nan.

const std = @import("std");
const mlx = @import("../../../mlx/ffi.zig");
const ops = @import("../../../mlx/ops.zig");
const kc = @import("../../testing/kernel_contract/root.zig");
const gq = @import("gather_qmv.zig");
const gqr = @import("gather_qmv_rows.zig");
const dr = @import("down_reduce.zig");
const fa = @import("../mlp/fused_activations.zig");
const activations = @import("../mlp/activations.zig");
const device_mod = @import("../../../mlx/device.zig");
const moe_fp4 = @import("moe_fp4.zig");
const rk = @import("router_kernel.zig");
const routing = @import("routing.zig");
const vk = @import("verify_kernels.zig");
const experts = @import("experts.zig");
const vp = @import("../linear/verify_projection.zig");
const moe_affine4 = @import("moe_affine4.zig");
const QuantMode = @import("../../quant/params.zig").QuantMode;

const spec = kc.spec;
const metal = kc.metal;
const host = kc.host;
const Shape = spec.Shape;

/// Experts in a bank unless a registration needs more: at least the widest top-k a gate draws.
const E: usize = 8;
/// Top-k of the rows kernels.
const TOPK: usize = 8;
/// GLM-5-Next's SwiGLU clamp.
const LIMIT: u32 = 10;

const Fmt = enum { affine, nvfp4, mxfp4 };
const Kind = enum { qmv, gateup, gateup_rows, down, down_rows };

const aff_in = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "K_size", "N_size" };
const nv_in = [_][*:0]const u8{ "x", "w_q", "scales", "inds", "K_size", "N_size" };
const gu_aff_in = [_][*:0]const u8{ "x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "K_size", "N_size" };
const gu_nv_in = [_][*:0]const u8{ "x", "wg_q", "g_scales", "wu_q", "u_scales", "inds", "sigtab", "K_size", "N_size" };
const gur_nv_in = gu_nv_in ++ [_][*:0]const u8{"gsc"};
const dr_aff_in = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "inds", "scores", "K_size", "N_size" };
const dr_nv_in = [_][*:0]const u8{ "x", "w_q", "scales", "inds", "scores", "K_size", "N_size" };

fn header(nv: bool) [:0]const u8 {
    return if (nv) gq.GQMV_NVFP4_HEADER else gq.GQMV_AFFINE_HEADER;
}

/// Indexed [nvfp4][x per expert].
const qmv_launches = [2][2]metal.Launch{
    .{
        .{ .name = "mlxserve_moe_gather_qmv", .inputs = &aff_in, .outputs = &.{"y"}, .source = gq.GQMV_SOURCES[0][0], .header = header(false), .tg_bytes = 0, .m1 = 928, .m2 = 928 },
        .{ .name = "mlxserve_moe_gather_qmv_px", .inputs = &aff_in, .outputs = &.{"y"}, .source = gq.GQMV_SOURCES[0][1], .header = header(false), .tg_bytes = 0, .m1 = 928, .m2 = 928 },
    },
    .{
        .{ .name = "mlxserve_moe_gather_qmv_nvfp4", .inputs = &nv_in, .outputs = &.{"y"}, .source = gq.GQMV_SOURCES[1][0], .header = header(true), .tg_bytes = 0, .m1 = 1024, .m2 = 1024 },
        .{ .name = "mlxserve_moe_gather_qmv_nvfp4_px", .inputs = &nv_in, .outputs = &.{"y"}, .source = gq.GQMV_SOURCES[1][1], .header = header(true), .tg_bytes = 0, .m1 = 928, .m2 = 928 },
    },
};
const gateup_launches = [2]metal.Launch{
    .{ .name = "mlxserve_moe_gather_qmv_gateup", .inputs = &gu_aff_in, .outputs = &.{"y"}, .source = gq.GQMV_GATEUP_SOURCES[0], .header = header(false), .tg_bytes = 0, .m1 = 576, .m2 = 576 },
    .{ .name = "mlxserve_moe_gather_qmv_gateup_nvfp4", .inputs = &gu_nv_in, .outputs = &.{"y"}, .source = gq.GQMV_GATEUP_SOURCES[1], .header = header(true), .tg_bytes = 0, .m1 = 512, .m2 = 512 },
};
const gateup_rows_launches = [2]metal.Launch{
    .{ .name = "mlxserve_moe_gather_qmv_gateup_rows", .inputs = &gu_aff_in, .outputs = &.{"y"}, .source = gqr.GQMV_GATEUP_ROWS_SOURCES[0], .header = header(false), .tg_bytes = 0, .m1 = 576, .m2 = 576 },
    .{ .name = "mlxserve_moe_gather_qmv_gateup_rows_nvfp4", .inputs = &gur_nv_in, .outputs = &.{"y"}, .source = gqr.GQMV_GATEUP_ROWS_SOURCES[1], .header = header(true), .tg_bytes = 0, .m1 = 928, .m2 = 928 },
};
/// `slot_vals[TOPK * ROWS]` T: 32 slots of 4 rows at most.
const down_tg: u32 = 32 * 4 * 2;
const down_launches = [2]metal.Launch{
    .{ .name = "mlxserve_moe_gather_qmv_downred", .inputs = &dr_aff_in, .outputs = &.{"y"}, .source = dr.GQMV_DOWNRED_SOURCES[0], .header = header(false), .tg_bytes = down_tg, .m1 = 512, .m2 = 512 },
    .{ .name = "mlxserve_moe_gather_qmv_downred_nvfp4", .inputs = &dr_nv_in, .outputs = &.{"y"}, .source = dr.GQMV_DOWNRED_SOURCES[1], .header = header(true), .tg_bytes = down_tg, .m1 = 512, .m2 = 512 },
};
const down_rows_launches = [2]metal.Launch{
    .{ .name = "mlxserve_moe_gather_qmv_downred_rows", .inputs = &dr_aff_in, .outputs = &.{"y"}, .source = dr.GQMV_DOWNRED_ROWS_SOURCES[0], .header = header(false), .tg_bytes = down_tg, .m1 = 512, .m2 = 512 },
    .{ .name = "mlxserve_moe_gather_qmv_downred_rows_nvfp4", .inputs = &dr_nv_in, .outputs = &.{"y"}, .source = dr.GQMV_DOWNRED_ROWS_SOURCES[1], .header = header(true), .tg_bytes = down_tg, .m1 = 512, .m2 = 512 },
};

/// An expert bank [E, N, K] as its checkpoint stores it: affine (MLX's quantize), NVFP4 (e2m1 nibbles, one e4m3 scale
/// a 16-group, no biases) or MXFP4 (one e8m0 scale a 32-group).
const Bank = struct {
    fmt: Fmt,
    aff: host.Affine,
    codes: []u8 = &.{},
    sc: []u8 = &.{},
    words: []f64 = &.{},
    scf: []f64 = &.{},

    fn init(fmt: Fmt, bits: u32, gs: usize) Bank {
        return .{ .fmt = fmt, .aff = .{ .bits = bits, .gs = gs, .n = 0, .k = 0 } };
    }

    fn deinit(b: *Bank, a: std.mem.Allocator) void {
        b.aff.deinit(a);
        for ([_][]u8{ b.codes, b.sc }) |s| a.free(s);
        for ([_][]f64{ b.words, b.scf }) |s| a.free(s);
        b.codes = &.{};
        b.sc = &.{};
        b.words = &.{};
        b.scf = &.{};
    }

    fn fill(b: *Bank, a: std.mem.Allocator, rng: std.Random, rows: usize, k: usize, dt: mlx.mlx_dtype, stress: bool) !void {
        if (b.fmt == .affine) {
            try b.aff.fill(a, rng, rows, k, dt, stress);
            if (!stress) return;
            reflect(b.aff.codes, rows, k);
            a.free(b.aff.words);
            b.aff.words = try host.packAffine(a, b.aff.codes, rows, k, b.aff.bits);
            return;
        }
        b.deinit(a);
        b.aff.n = rows;
        b.aff.k = k;
        b.codes = try a.alloc(u8, rows * k);
        b.sc = try a.alloc(u8, rows * k / b.aff.gs);
        rng.bytes(b.codes);
        for (b.codes) |*c| c.* &= 15;
        const nv = b.fmt == .nvfp4;
        if (stress) {
            // Positive codes, scale 1.0, reflected as `stressX` reflects x.
            for (b.codes) |*c| c.* &= 7;
            reflect(b.codes, rows, k);
            @memset(b.sc, if (nv) 0x38 else 127);
        } else for (b.sc) |*s| {
            s.* = if (nv) host.e4m3Code(rng, -4, 0, false) else 121 + rng.uintLessThan(u8, 7);
        }
        b.words = try host.packNibbles(a, b.codes);
        b.scf = try a.alloc(f64, b.sc.len);
        for (b.scf, b.sc) |*o, s| o.* = @floatFromInt(s);
    }

    /// Element (r, i) dequantized as MLX reads the format (an e4m3 code 0x7f / 0xff is NaN).
    fn w(b: *const Bank, r: usize, i: usize) f64 {
        const at = r * b.aff.k + i;
        if (b.fmt == .affine) {
            const g = at / b.aff.gs;
            return @as(f64, @floatFromInt(b.aff.codes[at])) * b.aff.scales[g] + b.aff.biases[g];
        }
        const s = b.sc[at / b.aff.gs];
        if (b.fmt == .mxfp4) return host.e2m1(b.codes[at]) * host.e8m0(s);
        if (s & 0x7f == 0x7f) return std.math.nan(f64);
        return host.e2m1(b.codes[at]) * host.e4m3(s);
    }

    /// |q s| + |b| (affine) or |w|: the term's magnitude through its dequant.
    fn mag(b: *const Bank, r: usize, i: usize) f64 {
        const at = r * b.aff.k + i;
        if (b.fmt == .affine) {
            const g = at / b.aff.gs;
            return @abs(@as(f64, @floatFromInt(b.aff.codes[at])) * b.aff.scales[g]) + @abs(b.aff.biases[g]);
        }
        return @abs(b.w(r, i));
    }

    /// Weights [E, N, words], scales [E, N, groups] (the activation dtype, or the uint8 codes), biases (affine).
    fn bind(b: *const Bank, rig: *metal.Rig, dt: mlx.mlx_dtype, ne: usize, n: usize, k: usize) ![3]mlx.mlx_array {
        const e: c_int = @intCast(ne);
        const nn: c_int = @intCast(n);
        if (b.fmt == .affine) {
            const g: c_int = @intCast(k / b.aff.gs);
            return .{
                try rig.input(.uint32, &.{ e, nn, @intCast(k * b.aff.bits / 32) }, b.aff.words),
                try rig.input(dt, &.{ e, nn, g }, b.aff.scales),
                try rig.input(dt, &.{ e, nn, g }, b.aff.biases),
            };
        }
        return .{
            try rig.input(.uint32, &.{ e, nn, @intCast(k / 8) }, b.words),
            try rig.input(.uint8, &.{ e, nn, @intCast(k / b.aff.gs) }, b.scf),
            .{ .ctx = null },
        };
    }
};

/// Stress codes: each row's code k-1-j is code j, and the codes x's +1 offsets meet (32..39) are 0 or 1, so a row's
/// sum is at most 8 (under GLM's clamp).
fn reflect(codes: []u8, rows: usize, k: usize) void {
    for (0..rows) |r| {
        const c = codes[r * k ..][0..k];
        for (32..40) |j| c[j] &= 1;
        for (0..k / 2) |j| c[k - 1 - j] = c[j];
    }
}

/// Stress x: `host.mirroredWide`'s values with each row's second half reversed, so x[k-1-j] = -x[j] (+1 at j in
/// 32..39). Against reflected codes the row's terms cancel to at most 8; a lower-precision sum meets each term's
/// negative in another order, so its rounding errors do not cancel.
fn stressX(rng: std.Random, v: []f64, k: usize) void {
    host.mirroredWide(rng, v, k);
    const h = k / 2;
    for (0..v.len / k) |r| std.mem.reverse(f64, v[r * k + h ..][0 .. k - h]);
}

fn modeOf(f: Fmt) QuantMode {
    return switch (f) {
        .affine => .affine,
        .nvfp4 => .nvfp4,
        .mxfp4 => .mxfp4,
    };
}

fn gamma32(d: usize) f64 {
    return spec.gamma(@floatFromInt(d), 0x1p-24);
}

/// MLX's Minimum / Clip on a rounded value: a NaN stays NaN.
fn clampNaN(v: f64, lo: f64, hi: f64) f64 {
    if (std.math.isNan(v)) return v;
    return @max(@min(v, hi), lo);
}

const Gather = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kind: Kind,
    dt: mlx.mlx_dtype = .bfloat16,
    px: bool = false,
    act2: bool = false,
    limit: u32 = 0,
    gscale: bool = false,
    lpr: usize = 8,
    /// Experts in the bank.
    ne: usize = E,
    sp: spec.Spec = undefined,
    banks: [2]Bank,
    shape: Shape = undefined,
    x: []f64 = &.{},
    inds: []f64 = &.{},
    scores: []f64 = &.{},
    gsc: []f64 = &.{},
    table: []f64 = &.{},
    /// Exact dot products [bank][slot * N + n] and their magnitudes, computed once a prepare.
    dv: []f64 = &.{},
    dm: []f64 = &.{},
    args: [12]mlx.mlx_array = undefined,
    n_args: usize = 0,
    msg: [256]u8 = undefined,
    /// The rows down+reduce's baseline: the solo kernel's launches and their bytes.
    base: metal.Rig,
    base_bytes: std.ArrayList(u8) = .empty,

    fn init(a: std.mem.Allocator, kind: Kind, f: Fmt, b: u32, g: usize) Gather {
        const nv = f == .nvfp4;
        const l: *const metal.Launch = switch (kind) {
            .qmv => &qmv_launches[@intFromBool(nv)][0],
            .gateup => &gateup_launches[@intFromBool(nv)],
            .gateup_rows => &gateup_rows_launches[@intFromBool(nv)],
            .down => &down_launches[@intFromBool(nv)],
            .down_rows => &down_rows_launches[@intFromBool(nv)],
        };
        return .{ .rig = metal.Rig.init(a), .base = metal.Rig.init(a), .launch = l, .kind = kind, .banks = .{ Bank.init(f, b, g), Bank.init(f, b, g) } };
    }

    pub fn deinit(self: *Gather) void {
        const a = self.rig.a;
        self.rig.deinit();
        self.base.deinit();
        self.base_bytes.deinit(a);
        for (&self.banks) |*b| b.deinit(a);
        for ([_][]f64{ self.x, self.inds, self.scores, self.gsc, self.table, self.dv, self.dm }) |s| a.free(s);
    }

    fn fmt(self: *const Gather) Fmt {
        return self.banks[0].fmt;
    }

    fn bits(self: *const Gather) u32 {
        return self.banks[0].aff.bits;
    }

    fn gs(self: *const Gather) usize {
        return self.banks[0].aff.gs;
    }

    fn twoBanks(self: *const Gather) bool {
        return self.kind == .gateup or self.kind == .gateup_rows;
    }

    fn rowsKind(self: *const Gather) bool {
        return self.kind == .gateup_rows or self.kind == .down_rows;
    }

    fn slots(self: *const Gather) usize {
        return if (self.rowsKind()) self.shape.rows * TOPK else self.shape.rows;
    }

    /// The x row slot `s` reads.
    fn xrow(self: *const Gather, s: usize) usize {
        return switch (self.kind) {
            .qmv => if (self.px) s else 0,
            .gateup => 0,
            .gateup_rows => s / TOPK,
            .down, .down_rows => s,
        };
    }

    fn xrows(self: *const Gather) usize {
        return self.xrow(self.slots() - 1) + 1;
    }

    /// Output rows of the reducing kernels, and the slots each sums.
    fn perGroup(self: *const Gather) usize {
        return if (self.kind == .down) self.shape.rows else TOPK;
    }

    fn rowsPerSg(self: *const Gather) usize {
        return 32 / self.lpr;
    }

    pub fn geometry(self: *const Gather, shape: Shape) ?metal.Geometry {
        const n: c_int = @intCast(shape.n);
        const k = shape.k;
        if (shape.rows == 0 or shape.n == 0 or k == 0 or k % self.gs() != 0) return null;
        const kp: c_int = @intCast(if (self.bits() == 3) k / 8 else k * self.bits() / 32);
        var g: metal.Geometry = undefined;
        switch (self.kind) {
            .qmv, .gateup, .gateup_rows => {
                if (shape.n % 8 != 0) return null;
                const z: c_int = @intCast(if (self.kind == .gateup_rows) shape.rows * TOPK else shape.rows);
                if (self.kind == .gateup_rows and shape.rows > 8) return null;
                if (self.kind != .gateup_rows and shape.rows > self.ne) return null;
                g = .{ .grid = .{ 32, n, z }, .tg = .{ 32, 8, 1 } };
                if (self.kind == .gateup_rows) g.out(&.{ @intCast(shape.rows), @intCast(TOPK), n }, self.dt) else g.out(&.{ z, n }, self.dt);
                g.dtype("T", self.dt);
                g.int("GS", self.gs());
                g.int("BITS", self.bits());
                switch (self.kind) {
                    .qmv => g.flag("ACT2", self.act2),
                    .gateup => {
                        g.int("KP", kp);
                        g.int("LIMIT", self.limit);
                    },
                    else => {
                        g.int("KP", kp);
                        g.int("TOPK", TOPK);
                        if (self.fmt() == .nvfp4) g.int("GSCALE", @intFromBool(self.gscale));
                    },
                }
            },
            .down, .down_rows => {
                const rows = self.rowsPerSg();
                const topk: c_int = @intCast(if (self.kind == .down) shape.rows else TOPK);
                if (shape.n % rows != 0 or k >= 2048 or topk > 32) return null;
                if (self.kind == .down and shape.rows > self.ne) return null;
                if (self.kind == .down_rows and shape.rows > 8) return null;
                const tiles: c_int = @intCast(shape.n / rows);
                g = .{ .grid = .{ tiles * topk * 32, if (self.kind == .down_rows) @intCast(shape.rows) else 1, 1 }, .tg = .{ topk * 32, 1, 1 } };
                if (self.kind == .down_rows) g.out(&.{ @intCast(shape.rows), n }, self.dt) else g.out(&.{n}, self.dt);
                g.dtype("T", self.dt);
                g.int("GS", self.gs());
                g.int("BITS", self.bits());
                g.int("TOPK", topk);
                g.int("ROWS", rows);
                g.int("LPR", self.lpr);
                g.int("KP", kp);
            },
        }
        return g;
    }

    pub fn prepare(self: *Gather, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        for ([_]*[]f64{ &self.x, &self.inds, &self.scores, &self.gsc, &self.dv, &self.dm }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        if (self.table.len == 0 and self.twoBanks()) self.table = try host.read(a, try fa.swigluSigTable(self.rig.stream(), self.dt, std.heap.c_allocator), self.rig.stream());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = shape.k;
        const stress = fill == .stress;
        for (self.banks[0..if (self.twoBanks()) 2 else 1]) |*b| try b.fill(a, rng, self.ne * shape.n, k, self.dt, stress);
        self.x = try a.alloc(f64, self.xrows() * k);
        if (stress) stressX(rng, self.x, k) else host.uniformInto(rng, self.x, -2, 2, self.dt);
        // Each row of slots picks distinct experts.
        self.inds = try a.alloc(f64, self.slots());
        const per = if (self.rowsKind()) TOPK else shape.rows;
        var buf: [32]u8 = undefined;
        const perm = buf[0..self.ne];
        for (0..self.slots() / per) |r| {
            for (perm, 0..) |*p, i| p.* = @intCast(i);
            rng.shuffle(u8, perm);
            for (0..per) |j| self.inds[r * per + j] = @floatFromInt(perm[j]);
        }
        self.scores = try a.alloc(f64, self.slots());
        for (self.scores) |*s| s.* = if (stress) 1 else host.rt(self.dt, 0.05 + 0.25 * rng.float(f64));
        self.gsc = try a.alloc(f64, self.ne);
        for (self.gsc) |*s| s.* = if (stress) 1 else host.rt(self.dt, 0.5 + 1.5 * rng.float(f64));
        if (fill == .nonfinite) self.plant(k);
        try self.bind();
        return true;
    }

    /// NaN and both infinities into x rows that feed only some outputs, else into the first bank's scales.
    fn plant(self: *Gather, k: usize) void {
        const nan = std.math.nan(f64);
        const inf = std.math.inf(f64);
        const step: usize = if (self.kind == .down_rows) TOPK else 1;
        const by_x = switch (self.kind) {
            .qmv => self.px,
            .gateup_rows, .down_rows => true,
            .gateup, .down => false,
        };
        if (by_x) {
            const at = [_]struct { usize, f64 }{ .{ k / 3, nan }, .{ k / 2, inf }, .{ k - 1, -inf } };
            for (at, 0..) |p, i| if (i * step < self.slots()) {
                self.x[self.xrow(i * step) * k + p[0]] = p[1];
            };
            return;
        }
        const b = &self.banks[0];
        const n = self.shape.n;
        const row = [3]usize{ @as(usize, @intFromFloat(self.inds[0])) * n + 1, @as(usize, @intFromFloat(self.inds[1])) * n + 2, @as(usize, @intFromFloat(self.inds[2])) * n + 3 };
        if (b.fmt == .nvfp4) {
            b.sc[row[0] * k / 16] = 0x7f;
            b.scf[row[0] * k / 16] = 0x7f;
            return;
        }
        const groups = k / b.aff.gs;
        b.aff.scales[row[0] * groups] = nan;
        b.aff.scales[row[1] * groups + 1] = inf;
        b.aff.biases[row[2] * groups] = -inf;
    }

    const Args = struct { x: mlx.mlx_array, b: [2][3]mlx.mlx_array, inds: mlx.mlx_array, sig: mlx.mlx_array, scores: mlx.mlx_array, gsc: mlx.mlx_array, kk: mlx.mlx_array, nn: mlx.mlx_array };

    fn bind(self: *Gather) !void {
        const rig = &self.rig;
        rig.reset();
        const k: c_int = @intCast(self.shape.k);
        const n = self.shape.n;
        const ns: c_int = @intCast(self.slots());
        var h: Args = undefined;
        h.x = try rig.input(self.dt, switch (self.kind) {
            .qmv => if (self.px) &.{ ns, k } else &.{k},
            .gateup => &.{k},
            .gateup_rows => &.{ @intCast(self.shape.rows), k },
            .down => &.{ ns, k },
            .down_rows => &.{ @intCast(self.shape.rows), @intCast(TOPK), k },
        }, self.x);
        h.b[0] = try self.banks[0].bind(rig, self.dt, self.ne, n, self.shape.k);
        if (self.twoBanks()) h.b[1] = try self.banks[1].bind(rig, self.dt, self.ne, n, self.shape.k);
        h.inds = try rig.input(.uint32, if (self.rowsKind()) &.{ @intCast(self.shape.rows), @intCast(TOPK) } else &.{ns}, self.inds);
        if (self.twoBanks()) h.sig = try rig.input(self.dt, &.{65536}, self.table);
        if (self.kind == .down or self.kind == .down_rows) h.scores = try rig.input(self.dt, if (self.rowsKind()) &.{ @intCast(self.shape.rows), @intCast(TOPK) } else &.{ns}, self.scores);
        if (self.gscale) h.gsc = try rig.input(self.dt, &.{@intCast(self.ne)}, self.gsc);
        h.kk = try rig.input(.int32, &.{}, &.{@floatFromInt(self.shape.k)});
        h.nn = try rig.input(.int32, &.{}, &.{@floatFromInt(n)});
        const nv = self.fmt() == .nvfp4;
        var list: std.ArrayList(mlx.mlx_array) = .initBuffer(&self.args);
        list.appendAssumeCapacity(h.x);
        switch (self.kind) {
            .qmv, .down, .down_rows => {
                list.appendSliceAssumeCapacity(h.b[0][0..if (nv) 2 else 3]);
                list.appendAssumeCapacity(h.inds);
                if (self.kind != .qmv) list.appendAssumeCapacity(h.scores);
            },
            .gateup, .gateup_rows => {
                list.appendSliceAssumeCapacity(h.b[0][0..if (nv) 2 else 3]);
                list.appendSliceAssumeCapacity(h.b[1][0..if (nv) 2 else 3]);
                list.appendAssumeCapacity(h.inds);
                list.appendAssumeCapacity(h.sig);
            },
        }
        list.appendAssumeCapacity(h.kk);
        list.appendAssumeCapacity(h.nn);
        // Without a gate scale the rows kernel never reads `gsc`; any array binds, as in the wrapper.
        if (self.kind == .gateup_rows and nv) list.appendAssumeCapacity(if (self.gscale) h.gsc else h.sig);
        self.n_args = list.items.len;
    }

    pub fn run(self: *Gather) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Gather) usize {
        return switch (self.kind) {
            .down => self.shape.n,
            .down_rows => self.shape.rows * self.shape.n,
            else => self.slots() * self.shape.n,
        };
    }

    fn expert(self: *const Gather, s: usize) usize {
        return @intFromFloat(self.inds[s]);
    }

    /// Every slot's exact dot products and magnitudes, once a prepare.
    fn dots(self: *Gather) !void {
        if (self.dv.len > 0) return;
        const a = self.rig.a;
        const n = self.shape.n;
        const k = self.shape.k;
        const nb: usize = if (self.twoBanks()) 2 else 1;
        self.dv = try a.alloc(f64, nb * self.slots() * n);
        self.dm = try a.alloc(f64, nb * self.slots() * n);
        const w = try a.alloc(f64, 3 * k);
        defer a.free(w);
        for (0..nb) |bi| for (0..self.slots()) |s| for (0..n) |j| {
            const r = self.expert(s) * n + j;
            for (w[0..k], w[k .. 2 * k], 0..) |*o, *m, i| {
                o.* = self.banks[bi].w(r, i);
                m.* = self.banks[bi].mag(r, i);
            }
            const xr = self.x[self.xrow(s) * k ..][0..k];
            for (w[2 * k ..], xr) |*o, v| o.* = @abs(v);
            const at = (bi * self.slots() + s) * n + j;
            self.dv[at] = host.dot(xr, w[0..k])[0];
            self.dm[at] = host.dot(w[2 * k ..], w[k .. 2 * k])[0];
        };
    }

    fn pattern(self: *const Gather, v: f64) usize {
        return if (self.dt == .bfloat16) spec.toBf16(@floatCast(v)) else @as(u16, @bitCast(@as(f16, @floatCast(v))));
    }

    /// The kernel's arithmetic after its sums on dot products `dv` (magnitudes `dm`): the float64 outputs and, for
    /// the kernels held to float64, each output's allowance.
    fn epilogue(self: *const Gather, dv: []const f64, dm: []const f64, want: []f64, allow: []f64) void {
        const n = self.shape.n;
        const sn = self.slots() * n;
        const dt = self.dt;
        const ut = 0x1p-8 * @as(f64, if (dt == .bfloat16) 1 else 0x1p-3);
        switch (self.kind) {
            .qmv => for (want, allow, dv[0..sn], dm[0..sn]) |*y, *l, d, m| {
                if (self.act2) {
                    const o = @max(d, 0);
                    y.* = if (std.math.isNan(d)) d else o * o;
                    l.* = spec.bound(self.sp, self.sp.out, self.shape, y.*, 2 * @abs(d) * m);
                } else {
                    y.* = d;
                    l.* = spec.bound(self.sp, self.sp.out, self.shape, d, m);
                }
            },
            .gateup, .gateup_rows => for (want, allow, 0..) |*y, *l, at| {
                const s = at / n;
                var gt = host.rt(dt, dv[at]);
                var u = host.rt(dt, dv[sn + at]);
                if (self.gscale) gt = host.rt(dt, gt * self.gsc[self.expert(s)]);
                if (self.limit > 0) {
                    const lim: f64 = @floatFromInt(self.limit);
                    gt = clampNaN(gt, -std.math.inf(f64), lim);
                    u = clampNaN(u, -lim, lim);
                }
                y.* = host.rt(dt, gt * self.table[self.pattern(gt)]) * u;
                l.* = 0; // replaced: gate 3 compares bits
            },
            .down, .down_rows => for (want, allow, 0..) |*y, *l, at| {
                const row = at / n;
                const j = at % n;
                var sum: f64 = 0;
                var big: f64 = 0;
                var err: f64 = 0;
                for (0..self.perGroup()) |e| {
                    const s = row * self.perGroup() + e;
                    const sc = self.scores[s];
                    sum += sc * dv[s * n + j];
                    big += @abs(sc * dv[s * n + j]);
                    err += @abs(sc) * gamma32(self.shape.k + 4) * dm[s * n + j];
                }
                y.* = sum;
                // Each slot's rounding to T, its product, and the bf16 chain over the slots: perGroup + 2 roundings.
                l.* = err * (1 + ut) + spec.gamma(@floatFromInt(self.perGroup() + 2), ut) * (big + err) + 0x1p-133 * @as(f64, @floatFromInt(self.perGroup()));
            },
        }
    }

    pub fn reference(self: *Gather, want: []f64, mag: []f64) !void {
        try self.dots();
        self.epilogue(self.dv, self.dm, want, mag);
    }

    /// fp32 sums of the integer stress terms (under 2^24) are exact in any order; fp16 and bf16 sums are run for the
    /// first 32 outputs of every slot: a lower bound on the miss.
    pub fn simulate(self: *Gather, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.dots();
        const a = self.rig.a;
        const scratch = try a.alloc(f64, out.len);
        defer a.free(scratch);
        if (acc == .f32) {
            for (self.dm) |m| if (m >= 0x1p24) return error.StressPastExact;
            return self.epilogue(self.dv, self.dm, out, scratch);
        }
        const dv = try a.dupe(f64, self.dv);
        defer a.free(dv);
        const n = self.shape.n;
        const k = self.shape.k;
        const t = try a.alloc(f64, k);
        defer a.free(t);
        const nb: usize = if (self.twoBanks()) 2 else 1;
        for (0..nb) |bi| for (0..self.slots()) |s| for (0..@min(32, n)) |j| {
            const r = self.expert(s) * n + j;
            for (t, 0..) |*o, i| o.* = self.x[self.xrow(s) * k + i] * self.banks[bi].w(r, i);
            dv[(bi * self.slots() + s) * n + j] = spec.sumIn(acc, order, t);
        };
        self.epilogue(dv, self.dm, out, scratch);
    }

    /// Gates 1 and 3 against the float64 outputs and their allowances.
    pub fn check(self: *Gather, at: spec.Gate) !?[]const u8 {
        try self.dots();
        const a = self.rig.a;
        const n = self.count();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        self.epilogue(self.dv, self.dm, buf[0..n], buf[n .. 2 * n]);
        const out = self.rig.outputBytes();
        for (buf[2 * n ..], 0..) |*g, i| g.* = spec.read(self.sp.out, out, i);
        return host.within(&self.msg, at, buf[2 * n ..], buf[0..n], buf[n .. 2 * n]);
    }

    fn mode(self: *const Gather) QuantMode {
        return modeOf(self.fmt());
    }

    fn bankArgs(self: *const Gather, bi: usize) [3]mlx.mlx_array {
        const nv = self.fmt() == .nvfp4;
        const base: usize = 1 + bi * @as(usize, if (nv) 2 else 3);
        return .{ self.args[base], self.args[base + 1], if (nv) .{ .ctx = null } else self.args[base + 2] };
    }

    fn indsArg(self: *const Gather) mlx.mlx_array {
        const nv = self.fmt() == .nvfp4;
        const per: usize = if (nv) 2 else 3;
        return self.args[1 + per * @as(usize, if (self.twoBanks()) 2 else 1)];
    }

    /// The composed chain each output must equal: two gathers and MLX's SwiGLU (per row for the rows kernel, the
    /// gate scale as MLX's multiply), or the solo down+reduce per row.
    pub fn replaced(self: *Gather) ![]const u8 {
        const s = self.rig.stream();
        var g = ops.Graph.init(self.rig.a, s);
        defer g.deinit();
        const x = self.args[0];
        const inds = self.indsArg();
        const rows: usize = if (self.rowsKind()) self.shape.rows else 1;
        var outs: [8]mlx.mlx_array = undefined;
        self.base_bytes.clearRetainingCapacity();
        for (0..rows) |r| {
            const ri: c_int = @intCast(r);
            const topk: c_int = @intCast(TOPK);
            const xr = if (self.rowsKind()) try g.sliceAxis(x, 0, ri, ri + 1) else x;
            const ir = if (self.rowsKind()) try g.reshape(try g.sliceAxis(inds, 0, ri, ri + 1), &.{topk}) else inds;
            if (self.kind == .down_rows) {
                // The solo kernel through its declared launch: the shipped solo wrapper caches its config without K.
                var solo = self.*;
                solo.kind = .down;
                solo.shape.rows = TOPK;
                const b = self.bankArgs(0);
                const sc = try g.reshape(try g.sliceAxis(self.args[self.n_args - 3], 0, ri, ri + 1), &.{topk});
                var list: [8]mlx.mlx_array = undefined;
                var len: usize = 0;
                for ([_]mlx.mlx_array{ try g.reshape(xr, &.{ topk, @intCast(self.shape.k) }), b[0], b[1], b[2], ir, sc, self.args[self.n_args - 2], self.args[self.n_args - 1] }, 0..) |v, i| {
                    if (i == 3 and self.fmt() == .nvfp4) continue;
                    list[len] = v;
                    len += 1;
                }
                try self.base.launch(&down_launches[@intFromBool(self.fmt() == .nvfp4)], solo.geometry(solo.shape).?, list[0..len]);
                try self.base_bytes.appendSlice(self.rig.a, self.base.outputBytes());
                continue;
            }
            var pair: [2]mlx.mlx_array = undefined;
            for (&pair, 0..) |*p, bi| {
                const b = self.bankArgs(bi);
                p.* = try g.adopt((try gq.gatherQmv(s, xr, b[0], b[1], b[2], ir, self.bits(), @intCast(self.gs()), self.mode(), false)) orelse return error.Declined);
            }
            if (self.gscale) {
                const per = try g.reshape(try g.take(self.args[self.n_args - 1], ir, 0), &.{ topk, 1 });
                pair[0] = try g.mul(pair[0], per);
            }
            outs[r] = if (self.limit > 0)
                try g.adopt(try activations.clampedSwiGLU(s, pair[0], pair[1], @floatFromInt(self.limit)))
            else
                try g.mul(try g.mul(pair[0], try g.sigmoid(pair[0])), pair[1]);
        }
        if (self.kind == .down_rows) return self.base_bytes.items;
        return self.rig.bytesOf(outs[0..rows]);
    }

    /// The shipped wrapper on the same inputs.
    fn wrapper(self: *Gather) ![]const u8 {
        const s = self.rig.stream();
        const x = self.args[0];
        const inds = self.indsArg();
        const b0 = self.bankArgs(0);
        const nb = self.bits();
        const gsz: u32 = @intCast(self.gs());
        const y = switch (self.kind) {
            .qmv => try gq.gatherQmvAct(s, x, b0[0], b0[1], b0[2], inds, nb, gsz, self.mode(), self.px, self.act2),
            .gateup => blk: {
                const b1 = self.bankArgs(1);
                break :blk try gq.gatherQmvGateUp(s, x, b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], inds, nb, gsz, self.mode(), self.limit);
            },
            .gateup_rows => blk: {
                const b1 = self.bankArgs(1);
                break :blk try gqr.gatherQmvGateUpRows(s, x, b0[0], b0[1], b0[2], b1[0], b1[1], b1[2], inds, nb, gsz, self.mode(), if (self.gscale) self.args[self.n_args - 1] else null);
            },
            .down => try dr.gatherQmvDownReduce(s, x, b0[0], b0[1], b0[2], inds, self.args[self.n_args - 3], nb, gsz, self.mode()),
            .down_rows => try dr.gatherQmvDownReduceRows(s, x, b0[0], b0[1], b0[2], inds, self.args[self.n_args - 3], nb, gsz, self.mode()),
        } orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

fn depthK(shape: Shape) usize {
    return shape.k + 4; // a row's sum, behind the dequant (q * s + b) and the product
}

const e4m3_nan_code: spec.Known = .{ .gate = .nonfinite, .finding = "an NVFP4 scale decodes by bit placement, so e4m3's NaN code 0x7f reads as 480 where MLX's fp8_e4m3 gives NaN; with one x for every output, a NaN code is the only non-finite input that leaves some outputs finite, and it comes out finite" };
const act2_drops_nan: spec.Known = .{ .gate = .nonfinite, .finding = "ReLU^2 runs metal::max(o, 0), which returns 0 for a NaN, where MLX's maximum (the chain it stands for) keeps the NaN: a NaN sum gives 0" };
const limit_drops_nan: spec.Known = .{ .gate = .nonfinite, .finding = "metal::min / max return the other operand for a NaN, where MLX's Minimum and Clip (the clampedSwiGLU chain it replaces) keep the NaN: a NaN gate or up gives a finite output" };

fn gatherSpec(comptime name: []const u8, g: Gather, served: []const Shape, replaces: ?[]const u8, known: ?spec.Known) spec.Spec {
    const vpw: usize = if (g.bits() == 3) 8 else 32 / g.bits();
    const lanes: usize = if (g.kind == .down or g.kind == .down_rows) g.lpr else 32;
    const n_tile: spec.Tile = if (g.kind == .down or g.kind == .down_rows) .{ .size = 8, .step = 4, .min = 4 } else .{ .size = 16, .step = 8, .min = 8 };
    return .{
        .name = name,
        .accum = .f32,
        .depth = depthK,
        .ops = if (g.act2) 1 else 0,
        .mids = if (g.act2) 2 else 0,
        .out = host.elemOf(g.dt),
        .replaces = replaces,
        // n: the 8-simdgroup (or ROWS-row) tile; k: one pass of the lanes over their packs.
        .tiles = .{ .{}, n_tile, .{ .size = lanes * vpw, .step = g.gs(), .min = g.gs() } },
        .served = served,
        .known = known,
    };
}

fn add(set: *kc.Set, g0: Gather, comptime name: []const u8, served: []const Shape, replaces: ?[]const u8, known: ?spec.Known) !void {
    var g = g0;
    g.sp = gatherSpec(name, g, served, replaces, known);
    try set.add(Gather, g, g.sp, null);
}

// ── fp4 banks: moe_fp4's gate+up+SwiGLU and weighted down ──

const fp4_gateup_in = [_][*:0]const u8{ "x", "wg_q", "g_scales", "wu_q", "u_scales", "inds", "sigtab", "gsc", "usc" };
const fp4_downred_in = [_][*:0]const u8{ "act", "wd_q", "d_scales", "inds", "scores", "sdown", "glog" };
const fp4_headers = [2][:0]const u8{ moe_fp4.E2M1_HEADER ++ moe_fp4.NV_SCALE ++ moe_fp4.DOT16, moe_fp4.E2M1_HEADER ++ moe_fp4.MX_SCALE ++ moe_fp4.DOT16 };
/// Indexed [part][nv, mx].
const fp4_launches = [2][2]metal.Launch{
    .{
        .{ .name = "mlxserve_moe_fp4_gateup_nv", .inputs = &fp4_gateup_in, .outputs = &.{"y"}, .source = moe_fp4.GATEUP_SOURCE, .header = fp4_headers[0], .tg_bytes = 0, .m1 = 1024, .m2 = 1024 },
        .{ .name = "mlxserve_moe_fp4_gateup_mx", .inputs = &fp4_gateup_in, .outputs = &.{"y"}, .source = moe_fp4.GATEUP_SOURCE, .header = fp4_headers[1], .tg_bytes = 0, .m1 = 1024, .m2 = 1024 },
    },
    .{
        .{ .name = "mlxserve_moe_fp4_downred_nv", .inputs = &fp4_downred_in, .outputs = &.{"y"}, .source = moe_fp4.DOWNRED_SOURCE, .header = fp4_headers[0], .tg_bytes = 0, .m1 = 1024, .m2 = 1024 },
        .{ .name = "mlxserve_moe_fp4_downred_mx", .inputs = &fp4_downred_in, .outputs = &.{"y"}, .source = moe_fp4.DOWNRED_SOURCE, .header = fp4_headers[1], .tg_bytes = 0, .m1 = 1024, .m2 = 1024 },
    },
};

const Part = enum { gateup, downred };

/// One of `moe_fp4.decode`'s two dispatches. Rows are tokens (each with its own TOPK experts out of a bank of TOPK);
/// gate+up: n = the expert width I, k = hidden; down: n = hidden, k = I.
const Fp4 = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    part: Part,
    dt: mlx.mlx_dtype = .bfloat16,
    gscale: bool = false,
    uscale: bool = false,
    clamp: u32 = 0,
    shared: bool = false,
    /// Fill every bank (the drift guard chains both dispatches).
    both: bool = false,
    sp: spec.Spec = undefined,
    /// gate, up, down.
    banks: [3]Bank,
    shape: Shape = undefined,
    x: []f64 = &.{},
    act: []f64 = &.{},
    inds: []f64 = &.{},
    scores: []f64 = &.{},
    gsc: []f64 = &.{},
    usc: []f64 = &.{},
    sdown: []f64 = &.{},
    glog: f64 = 0,
    table: []f64 = &.{},
    dv: []f64 = &.{},
    dm: []f64 = &.{},
    args: [9]mlx.mlx_array = undefined,
    n_args: usize = 0,
    /// The other dispatch's inputs (drift guard).
    other: [9]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator, part: Part, f: Fmt) Fp4 {
        const g: usize = if (f == .nvfp4) 16 else 32;
        const b = Bank.init(f, 4, g);
        return .{ .rig = metal.Rig.init(a), .launch = &fp4_launches[@backingInt(part)][@intFromBool(f == .mxfp4)], .part = part, .banks = .{ b, b, b } };
    }

    pub fn deinit(self: *Fp4) void {
        const a = self.rig.a;
        self.rig.deinit();
        for (&self.banks) |*b| b.deinit(a);
        for ([_][]f64{ self.x, self.act, self.inds, self.scores, self.gsc, self.usc, self.sdown, self.table, self.dv, self.dm }) |s| a.free(s);
    }

    fn fmt(self: *const Fp4) Fmt {
        return self.banks[0].fmt;
    }

    fn gs(self: *const Fp4) usize {
        return self.banks[0].aff.gs;
    }

    /// Hidden and expert widths at a shape of `part`.
    fn hiddenOf(part: Part, shape: Shape) usize {
        return if (part == .gateup) shape.k else shape.n;
    }

    fn interOf(part: Part, shape: Shape) usize {
        return if (part == .gateup) shape.n else shape.k;
    }

    fn hidden(self: *const Fp4, shape: Shape) usize {
        return hiddenOf(self.part, shape);
    }

    fn inter(self: *const Fp4, shape: Shape) usize {
        return interOf(self.part, shape);
    }

    fn geometryOf(self: *const Fp4, part: Part, shape: Shape) ?metal.Geometry {
        const h = hiddenOf(part, shape);
        const i = interOf(part, shape);
        const rows: usize = @intCast(moe_fp4.ROWS * moe_fp4.SGS);
        if (shape.rows == 0 or shape.rows > 8 or h % 16 != 0 or i % 16 != 0 or h % self.gs() != 0 or i % self.gs() != 0 or h % rows != 0 or i % rows != 0) return null;
        const r: c_int = @intCast(shape.rows);
        const topk: c_int = @intCast(TOPK);
        const ch: c_int = @intCast(h);
        const ci: c_int = @intCast(i);
        var g: metal.Geometry = undefined;
        if (part == .gateup) {
            g = .{ .grid = .{ 32, @divExact(ci, moe_fp4.ROWS), r * topk }, .tg = .{ 32, moe_fp4.SGS, 1 } };
            g.out(&.{ r * topk, ci }, self.dt);
        } else {
            g = .{ .grid = .{ 32, @divExact(ch, moe_fp4.ROWS), r }, .tg = .{ 32, moe_fp4.SGS, 1 } };
            g.out(&.{ r, ch }, self.dt);
        }
        g.dtype("T", self.dt);
        g.int("ROWS", moe_fp4.ROWS);
        if (part == .gateup) {
            g.int("K", ch);
            g.int("N", ci);
            g.int("GS", self.gs());
            g.int("TOPK", topk);
            g.int("GSCALE", @intFromBool(self.gscale));
            g.int("USCALE", @intFromBool(self.uscale));
            g.int("CLAMP", self.clamp);
        } else {
            g.int("I", ci);
            g.int("H", ch);
            g.int("GS", self.gs());
            g.int("TOPK", topk);
            g.int("SHARED", @intFromBool(self.shared));
        }
        return g;
    }

    pub fn geometry(self: *const Fp4, shape: Shape) ?metal.Geometry {
        return self.geometryOf(self.part, shape);
    }

    fn slots(self: *const Fp4) usize {
        return self.shape.rows * TOPK;
    }

    pub fn prepare(self: *Fp4, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        for ([_]*[]f64{ &self.x, &self.act, &self.inds, &self.scores, &self.gsc, &self.usc, &self.sdown, &self.dv, &self.dm }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        if (self.table.len == 0) self.table = try host.read(a, try fa.swigluSigTable(self.rig.stream(), self.dt, std.heap.c_allocator), self.rig.stream());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const h = self.hidden(shape);
        const i = self.inter(shape);
        const stress = fill == .stress;
        if (self.part == .gateup or self.both) for (self.banks[0..2]) |*b| try b.fill(a, rng, TOPK * i, h, self.dt, stress);
        if (self.part == .downred or self.both) try self.banks[2].fill(a, rng, TOPK * h, i, self.dt, stress);
        self.x = try a.alloc(f64, shape.rows * h);
        self.act = try a.alloc(f64, self.slots() * i);
        if (stress) {
            stressX(rng, self.x, h);
            stressX(rng, self.act, i);
        } else {
            host.uniformInto(rng, self.x, -2, 2, self.dt);
            host.uniformInto(rng, self.act, -2, 2, self.dt);
        }
        self.inds = try a.alloc(f64, self.slots());
        var perm: [TOPK]u8 = undefined;
        for (0..shape.rows) |r| {
            for (&perm, 0..) |*p, j| p.* = @intCast(j);
            rng.shuffle(u8, &perm);
            for (perm, 0..) |p, j| self.inds[r * TOPK + j] = @floatFromInt(p);
        }
        self.scores = try a.alloc(f64, self.slots());
        for (self.scores) |*v| v.* = if (stress) 1 else host.rt(self.dt, 0.05 + 0.25 * rng.float(f64));
        self.gsc = try a.alloc(f64, TOPK);
        self.usc = try a.alloc(f64, TOPK);
        for (self.gsc, self.usc) |*g, *u| {
            g.* = if (stress) 1 else host.rt(self.dt, 0.5 + 1.5 * rng.float(f64));
            u.* = if (stress) 1 else host.rt(self.dt, 0.5 + 1.5 * rng.float(f64));
        }
        self.sdown = try a.alloc(f64, h);
        for (self.sdown) |*v| v.* = if (stress) 0 else host.rt(self.dt, -1 + 2 * rng.float(f64));
        self.glog = host.rt(self.dt, -3 + 6 * rng.float(f64));
        if (fill == .nonfinite) {
            // Token 0's x (or its first expert's activation) takes NaN, token 1's +Inf, token 2's -Inf.
            const v = if (self.part == .gateup) self.x else self.act;
            const w = if (self.part == .gateup) h else i;
            const per: usize = if (self.part == .gateup) 1 else TOPK;
            const at = [_]struct { usize, f64 }{ .{ w / 3, std.math.nan(f64) }, .{ w / 2, std.math.inf(f64) }, .{ w - 1, -std.math.inf(f64) } };
            for (at, 0..) |p, r| v[r * per * w + p[0]] = p[1];
        }
        try self.bind();
        return true;
    }

    fn bind(self: *Fp4) !void {
        const rig = &self.rig;
        rig.reset();
        const r: c_int = @intCast(self.shape.rows);
        const topk: c_int = @intCast(TOPK);
        const h = self.hidden(self.shape);
        const i = self.inter(self.shape);
        const ind = try rig.input(.uint32, &.{ r, topk }, self.inds);
        const gsc = try rig.input(self.dt, &.{topk}, self.gsc);
        const usc = try rig.input(self.dt, &.{topk}, self.usc);
        var gu: [9]mlx.mlx_array = undefined;
        var dn: [7]mlx.mlx_array = undefined;
        if (self.part == .gateup or self.both) {
            const g = try self.banks[0].bind(rig, self.dt, TOPK, i, h);
            const u = try self.banks[1].bind(rig, self.dt, TOPK, i, h);
            gu = .{ try rig.input(self.dt, &.{ r, @intCast(h) }, self.x), g[0], g[1], u[0], u[1], ind, try rig.input(self.dt, &.{65536}, self.table), gsc, usc };
        }
        if (self.part == .downred or self.both) {
            const d = try self.banks[2].bind(rig, self.dt, TOPK, h, i);
            dn = .{ try rig.input(self.dt, &.{ r * topk, @intCast(i) }, self.act), d[0], d[1], ind, try rig.input(self.dt, &.{ r, topk }, self.scores), try rig.input(self.dt, &.{@intCast(h)}, self.sdown), try rig.input(self.dt, &.{1}, &.{self.glog}) };
        }
        if (self.part == .gateup) {
            self.args = gu;
            self.n_args = 9;
            @memcpy(self.other[0..7], &dn);
        } else {
            @memcpy(self.args[0..7], &dn);
            self.n_args = 7;
            self.other = gu;
        }
    }

    pub fn run(self: *Fp4) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Fp4) usize {
        return if (self.part == .gateup) self.slots() * self.shape.n else self.shape.rows * self.shape.n;
    }

    fn expert(self: *const Fp4, s: usize) usize {
        return @intFromFloat(self.inds[s]);
    }

    /// Every slot's exact dot products (gate and up, or down) and magnitudes, once a prepare.
    fn dots(self: *Fp4) !void {
        if (self.dv.len > 0) return;
        const a = self.rig.a;
        const n = self.shape.n;
        const k = self.shape.k;
        const nb: usize = if (self.part == .gateup) 2 else 1;
        self.dv = try a.alloc(f64, nb * self.slots() * n);
        self.dm = try a.alloc(f64, nb * self.slots() * n);
        const w = try a.alloc(f64, 3 * k);
        defer a.free(w);
        for (0..nb) |bi| for (0..self.slots()) |s| for (0..n) |j| {
            const bank = &self.banks[if (self.part == .gateup) bi else 2];
            const r = self.expert(s) * n + j;
            for (w[0..k], w[k .. 2 * k], 0..) |*o, *m, i| {
                o.* = bank.w(r, i);
                m.* = bank.mag(r, i);
            }
            const xr = if (self.part == .gateup) self.x[(s / TOPK) * k ..][0..k] else self.act[s * k ..][0..k];
            for (w[2 * k ..], xr) |*o, v| o.* = @abs(v);
            const at = (bi * self.slots() + s) * n + j;
            self.dv[at] = host.dot(xr, w[0..k])[0];
            self.dm[at] = host.dot(w[2 * k ..], w[k .. 2 * k])[0];
        };
    }

    fn pattern(self: *const Fp4, v: f64) usize {
        return if (self.dt == .bfloat16) spec.toBf16(@floatCast(v)) else @as(u16, @bitCast(@as(f16, @floatCast(v))));
    }

    fn silu(v: f64) f64 {
        return v / (1 + @exp(-v));
    }

    /// The outputs on dot products `dv`: with `exact`, the float64 function and each output's allowance (gate 3);
    /// else the kernel's own chain of roundings after its sums (gates 1 and 4).
    fn epilogue(self: *const Fp4, dv: []const f64, dm: []const f64, exact: bool, want: []f64, allow: []f64) void {
        if (self.part == .gateup) return self.gateupOut(dv, dm, exact, want, allow);
        const n = self.shape.n;
        const dt = self.dt;
        const ut: f64 = if (dt == .bfloat16) 0x1p-8 else 0x1p-11;
        const floor: f64 = if (dt == .bfloat16) 0x1p-133 else 0x1p-24;
        const depth = self.sp.depth(self.shape);
        for (want, allow, 0..) |*y, *l, at| {
            const r = at / n;
            const j = at % n;
            var v: f64 = 0;
            var mag: f64 = 0;
            for (0..TOPK) |e| {
                const s = r * TOPK + e;
                v += self.scores[s] * dv[s * n + j];
                mag += @abs(self.scores[s]) * dm[s * n + j];
            }
            const ev = gamma32(depth) * mag;
            if (!self.shared) {
                y.* = v;
                l.* = ev + ut * (@abs(v) + ev) + floor;
                continue;
            }
            const sig = 1 / (1 + @exp(-self.glog));
            const sd = self.sdown[j];
            if (!exact) {
                y.* = host.rt(dt, v) + host.rt(dt, host.rt(dt, sig) * sd);
                l.* = 0;
                continue;
            }
            y.* = v + sig * sd;
            // T(v), the sigmoid (precise exp, then T), its product with the shared row, the f32 add, the output.
            const dsig = ut * sig + 0x1p-20;
            const work = ev + ut * (@abs(v) + ev) + @abs(sd) * dsig + ut * @abs(sig * sd) + 0x1p-24 * (@abs(v) + @abs(sig * sd));
            l.* = work + ut * (@abs(y.*) + work) + floor;
        }
    }

    fn gateupOut(self: *const Fp4, dv: []const f64, dm: []const f64, exact: bool, want: []f64, allow: []f64) void {
        const n = self.shape.n;
        const sn = self.slots() * n;
        const dt = self.dt;
        const ut: f64 = if (dt == .bfloat16) 0x1p-8 else 0x1p-11;
        const floor: f64 = if (dt == .bfloat16) 0x1p-133 else 0x1p-24;
        const depth = self.sp.depth(self.shape);
        const lim: f64 = @floatFromInt(self.clamp);
        for (want, allow, 0..) |*y, *l, at| {
            const e = self.expert(at / n);
            const gm: f64 = if (self.gscale) self.gsc[e] else 1;
            const um: f64 = if (self.uscale) self.usc[e] else 1;
            if (!exact) {
                var gt = host.rt(dt, dv[at]);
                var u = host.rt(dt, dv[sn + at]);
                if (self.gscale) gt = host.rt(dt, gt * gm);
                if (self.uscale) u = host.rt(dt, u * um);
                if (self.clamp > 0) {
                    gt = clampNaN(gt, -std.math.inf(f64), lim);
                    u = clampNaN(u, -lim, lim);
                }
                y.* = host.rt(dt, gt * self.table[self.pattern(gt)]) * u;
                l.* = 0;
                continue;
            }
            var g = dv[at] * gm;
            var u = dv[sn + at] * um;
            if (self.clamp > 0) {
                g = clampNaN(g, -std.math.inf(f64), lim);
                u = clampNaN(u, -lim, lim);
            }
            const sg = silu(g);
            y.* = sg * u;
            // Each sum's error and its rounding to T (and the scale's product); SiLU's slope is at most 1.1; the
            // table's rounding and the product's; the final multiply's.
            const ng: f64 = if (self.gscale) 2 else 1;
            const nu: f64 = if (self.uscale) 2 else 1;
            const eg = gm * gamma32(depth) * dm[at];
            const eu = um * gamma32(depth) * dm[sn + at];
            const dg = eg + ng * ut * (@abs(g) + eg);
            const du = eu + nu * ut * (@abs(u) + eu);
            const ds = 1.1 * dg + 2 * ut * (@abs(sg) + 1.1 * dg);
            const work = @abs(u) * ds + (@abs(sg) + ds) * du;
            l.* = work + ut * (@abs(y.*) + work) + floor;
        }
    }

    /// The kernel's rounding chain on exact sums (gate 4).
    pub fn reference(self: *Fp4, want: []f64, mag: []f64) !void {
        try self.dots();
        self.epilogue(self.dv, self.dm, false, want, mag);
    }

    /// fp32 sums of the integer stress terms (under 2^24) are exact in any order; fp16 and bf16 sums are run for the
    /// first 32 outputs of every slot: a lower bound on the miss.
    pub fn simulate(self: *Fp4, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.dots();
        const a = self.rig.a;
        const scratch = try a.alloc(f64, out.len);
        defer a.free(scratch);
        if (acc == .f32) {
            for (self.dm) |m| if (m >= 0x1p24) return error.StressPastExact;
            return self.epilogue(self.dv, self.dm, false, out, scratch);
        }
        const dv = try a.dupe(f64, self.dv);
        defer a.free(dv);
        const n = self.shape.n;
        const k = self.shape.k;
        const t = try a.alloc(f64, k);
        defer a.free(t);
        const nb: usize = if (self.part == .gateup) 2 else 1;
        for (0..nb) |bi| for (0..self.slots()) |s| for (0..@min(32, n)) |j| {
            const bank = &self.banks[if (self.part == .gateup) bi else 2];
            const r = self.expert(s) * n + j;
            const xr = if (self.part == .gateup) self.x[(s / TOPK) * k ..][0..k] else self.act[s * k ..][0..k];
            for (t, xr, 0..) |*o, v, i| o.* = v * bank.w(r, i);
            dv[(bi * self.slots() + s) * n + j] = spec.sumIn(acc, order, t);
        };
        self.epilogue(dv, self.dm, false, out, scratch);
    }

    /// Gate 1 on the kernel's chain, gate 3 against the float64 function within each output's allowance.
    pub fn check(self: *Fp4, at: spec.Gate) !?[]const u8 {
        try self.dots();
        const a = self.rig.a;
        const n = self.count();
        const buf = try a.alloc(f64, 3 * n);
        defer a.free(buf);
        self.epilogue(self.dv, self.dm, at == .shapes, buf[0..n], buf[n .. 2 * n]);
        const out = self.rig.outputBytes();
        for (buf[2 * n ..], 0..) |*g, i| g.* = spec.read(self.sp.out, out, i);
        return host.within(&self.msg, at, buf[2 * n ..], buf[0..n], buf[n .. 2 * n]);
    }

    /// Both dispatches through the declared launches (this one and its twin), then `moe_fp4.decode` on the same
    /// inputs: the bytes of the weighted down sum.
    fn chained(self: *Fp4) ![2][]const u8 {
        const a = self.rig.a;
        const gu = if (self.part == .gateup) self.args else self.other;
        var dn: [7]mlx.mlx_array = undefined;
        @memcpy(&dn, (if (self.part == .gateup) self.other else self.args)[0..7]);
        try self.rig.launch(&fp4_launches[0][@intFromBool(self.fmt() == .mxfp4)], self.geometryOf(.gateup, self.gateupShape()).?, &gu);
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        try mlx.check(mlx.mlx_array_set(&act, self.rig.output(0)));
        dn[0] = act;
        try self.rig.launch(&fp4_launches[1][@intFromBool(self.fmt() == .mxfp4)], self.geometryOf(.downred, self.downShape()).?, &dn);
        const ours = try a.dupe(u8, self.rig.outputBytes());
        errdefer a.free(ours);
        const B = moe_fp4.Bank;
        const scales: moe_fp4.Scales = .{ .gate = if (self.gscale) gu[7] else null, .up = if (self.uscale) gu[8] else null, .clamp = self.clamp };
        const sh: ?moe_affine4.Shared = if (self.shared) .{ .down = dn[5], .logit = dn[6] } else null;
        const y = (try moe_fp4.decode(self.rig.stream(), gu[0], B{ .w = gu[1], .s = gu[2] }, B{ .w = gu[3], .s = gu[4] }, B{ .w = dn[1], .s = dn[2] }, null, scales, sh, gu[5], dn[4], gu[6], modeOf(self.fmt()), @intCast(self.gs()))) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return .{ ours, try self.rig.bytesOf(&.{y}) };
    }

    fn gateupShape(self: *const Fp4) Shape {
        return .{ .rows = self.shape.rows, .n = self.inter(self.shape), .k = self.hidden(self.shape) };
    }

    fn downShape(self: *const Fp4) Shape {
        return .{ .rows = self.shape.rows, .n = self.hidden(self.shape), .k = self.inter(self.shape) };
    }
};

fn depthFp4(shape: Shape) usize {
    // A lane's chain: per 512-wide pass a 16-value dot (two chains of 8 and their join), its scale, its add; then the
    // 32-lane reduction (and the scores, one product each, ahead of every pass of the down).
    return 12 * TOPK * ((shape.k + 511) / 512) + 8;
}

fn fp4Spec(comptime name: []const u8, f: Fp4, served: []const Shape, known: ?spec.Known) spec.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = depthFp4,
        .out = host.elemOf(f.dt),
        // n: two simdgroups of four rows; k: one pass of the 32 lanes' 16 values.
        .tiles = .{ .{}, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 512, .step = f.gs(), .min = f.gs() } },
        .served = served,
        .known = known,
    };
}

fn addFp4(set: *kc.Set, f0: Fp4, comptime name: []const u8, served: []const Shape, known: ?spec.Known) !void {
    var f = f0;
    f.sp = fp4Spec(name, f, served, known);
    try set.add(Fp4, f, f.sp, null);
}

// ── the fused router ──

const RouterMode = rk.RouterMode;

fn routerInputs(mode: RouterMode) []const [*:0]const u8 {
    return switch (mode) {
        .softmax => &.{"logits"},
        .softmax_gate => &.{ "logits", "gx", "gw" },
        .sigmoid_bias, .sigmoid_bias_grouped, .logit_bias => &.{ "logits", "bias", "scale" },
    };
}

/// rk[NE] and (sigmoid arms) rw[NE] f32 at the largest expert count, gs[NG], sel[NK], gp[TG / 32].
const router_tg: u32 = 2 * 2048 * 4 + 32 * 4 + 32 * 4 + 8 * 4;
const router_launches = blk: {
    var out: [5]metal.Launch = undefined;
    for (&out, 0..) |*l, i| {
        const mode: RouterMode = @fromBackingInt(@intCast(i));
        l.* = .{
            .name = std.fmt.comptimePrint("mlxserve_moe_router_{t}", .{mode}),
            .inputs = routerInputs(mode),
            .outputs = if (mode == .softmax_gate) &.{ "inds", "scores", "glogit" } else &.{ "inds", "scores" },
            .source = rk.ROUTER_SOURCES[i],
            .tg_bytes = router_tg,
            // M1/M2 compiled limits (`offline.zig`): 928 for the gate, grouped and logit-select arms.
            .m1 = if (mode == .softmax or mode == .sigmoid_bias) 1024 else 928,
            .m2 = if (mode == .softmax or mode == .sigmoid_bias) 1024 else 928,
        };
    }
    break :blk out;
};

/// The fused top-k router. Rows are router rows, n is the top-k, k the expert count.
const Router = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    mode: RouterMode,
    /// Logits and output dtypes.
    dt: mlx.mlx_dtype = .bfloat16,
    out_dt: mlx.mlx_dtype = .bfloat16,
    norm: bool = true,
    scale: f32 = 1,
    ng: usize = 0,
    topkg: usize = 0,
    /// The shared-expert gate's width.
    gk: usize = 0,
    sp: spec.Spec = undefined,
    shape: Shape = undefined,
    logits: []f64 = &.{},
    bias: []f64 = &.{},
    gx: []f64 = &.{},
    gw: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,
    n_args: usize = 0,
    /// Planted experts per row (gate 1): NaN, +Inf, -Inf.
    nan_at: usize = 0,
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator, mode: RouterMode) Router {
        return .{ .rig = metal.Rig.init(a), .launch = &router_launches[@backingInt(mode)], .mode = mode };
    }

    pub fn deinit(self: *Router) void {
        const a = self.rig.a;
        self.rig.deinit();
        for ([_][]f64{ self.logits, self.bias, self.gx, self.gw }) |s| a.free(s);
    }

    fn sigmoidArm(self: *const Router) bool {
        return self.mode == .sigmoid_bias or self.mode == .sigmoid_bias_grouped or self.mode == .logit_bias;
    }

    pub fn geometry(self: *const Router, shape: Shape) ?metal.Geometry {
        const e: c_int = @intCast(shape.k);
        const k: c_int = @intCast(shape.n);
        if (shape.rows == 0 or k < 1 or k > 32 or e < k or e > 2048) return null;
        if (self.mode == .sigmoid_bias_grouped and (shape.k % self.ng != 0 or shape.k / self.ng < 2)) return null;
        const tg = rk.moeRouterThreadGroup(e);
        if (self.mode == .softmax_gate and tg < 64) return null;
        const rows: c_int = @intCast(shape.rows);
        var g = metal.Geometry{ .grid = .{ tg, rows, 1 }, .tg = .{ tg, 1, 1 } };
        g.out(&.{ rows, k }, .uint32);
        g.out(&.{ rows, k }, self.out_dt);
        if (self.mode == .softmax_gate) {
            g.out(&.{rows}, self.out_dt);
            g.int("GK", self.gk);
        }
        g.dtype("TOUT", self.out_dt);
        g.int("NE", e);
        g.int("NK", k);
        g.int("TG", tg);
        g.int("NORM", @intFromBool(self.norm));
        if (self.mode == .sigmoid_bias_grouped) {
            g.int("NG", self.ng);
            g.int("GSZ", shape.k / self.ng);
            g.int("TOPKG", self.topkg);
        }
        return g;
    }

    pub fn prepare(self: *Router, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        for ([_]*[]f64{ &self.logits, &self.bias, &self.gx, &self.gw }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const e = shape.k;
        self.logits = try a.alloc(f64, shape.rows * e);
        // Spread wide enough that the top-k boundary is never an exact tie (the chain's set is then defined).
        for (self.logits, 0..) |*v, i| v.* = host.rt(self.dt, (rng.float(f64) - 0.5) * 8 + @as(f64, @floatFromInt(i % 7)) * 0.011);
        self.bias = try a.alloc(f64, e);
        for (self.bias) |*v| v.* = host.rt(.float32, (rng.float(f64) - 0.5) * 0.2);
        if (self.mode == .softmax_gate) {
            self.gx = try a.alloc(f64, shape.rows * self.gk);
            self.gw = try a.alloc(f64, self.gk);
            host.uniformInto(rng, self.gx, -2, 2, .bfloat16);
            host.uniformInto(rng, self.gw, -0.2, 0.2, .bfloat16);
        }
        if (fill == .nonfinite) {
            self.nan_at = e / 3;
            const inf = std.math.inf(f64);
            self.logits[0 * e + e / 3] = std.math.nan(f64);
            self.logits[1 * e + e / 2] = inf;
            self.logits[2 * e + e - 1] = -inf;
            self.logits[3 * e + 1] = inf;
            self.logits[3 * e + 2] = -inf;
        }
        const rig = &self.rig;
        rig.reset();
        const rows: c_int = @intCast(shape.rows);
        self.args[0] = try rig.input(self.dt, &.{ rows, @intCast(e) }, self.logits);
        self.n_args = 1;
        if (self.mode == .softmax_gate) {
            self.args[1] = try rig.input(.bfloat16, &.{ rows, @intCast(self.gk) }, self.gx);
            self.args[2] = try rig.input(.bfloat16, &.{@intCast(self.gk)}, self.gw);
            self.n_args = 3;
        } else if (self.sigmoidArm()) {
            self.args[1] = try rig.input(.float32, &.{@intCast(e)}, self.bias);
            self.args[2] = try rig.scalar(self.scale);
            self.n_args = 3;
        }
        return true;
    }

    pub fn run(self: *Router) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Router) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Never asked: gate 1 runs the stated rule, gate 3 the chain's bits (or `check`), gate 4 is N/A.
    pub fn reference(_: *Router, _: []f64, _: []f64) !void {
        return error.RouterHasNoFloat64Outputs;
    }

    /// The composed chain this kernel replaces, on the same inputs: (ids [rows, k] u32, weights) bytes.
    fn chain(self: *Router) ![]const u8 {
        const s = self.rig.stream();
        const k: c_int = @intCast(self.shape.n);
        const x = self.args;
        const outer = rk.moe_router_fused_override;
        rk.moe_router_fused_override = false;
        defer rk.moe_router_fused_override = outer;
        const r = switch (self.mode) {
            .softmax, .softmax_gate => try routing.moeRoutingChain(x[0], k, s),
            .sigmoid_bias => try routing.hy3RoutingChain(x[0], x[1], k, self.norm, self.scale, s),
            .sigmoid_bias_grouped => try routing.groupLimitedRouting(x[0], x[1], k, @intCast(self.ng), @intCast(self.topkg), self.norm, self.scale, s),
            .logit_bias => try routing.kolibriRoutingChain(x[0], x[1], k, s),
        };
        defer _ = mlx.mlx_array_free(r.inds);
        defer _ = mlx.mlx_array_free(r.norm_scores);
        var g = ops.Graph.init(self.rig.a, s);
        defer g.deinit();
        return self.rig.bytesOf(&.{ try g.astype(r.inds, .uint32), try g.astype(r.norm_scores, self.out_dt) });
    }

    /// The chain's ids and weights laid out in the kernel's order: its top-k is a set (argpartition has no order), so
    /// row by row each of the kernel's ids takes the chain's weight for that id; a set that differs keeps the chain's
    /// own ids, and the bytes then differ.
    pub fn replaced(self: *Router) ![]const u8 {
        const a = self.rig.a;
        const theirs = try a.dupe(u8, try self.chain());
        defer a.free(theirs);
        const ours = self.rig.outputBytes();
        const nk = self.count();
        const k = self.shape.n;
        const wsz = spec.size(host.elemOf(self.out_dt));
        const out = &self.rig.scratch;
        out.clearRetainingCapacity();
        try out.appendSlice(a, theirs[0 .. 4 * nk + wsz * nk]);
        for (0..self.shape.rows) |r| {
            var same = true;
            for (0..k) |j| {
                if (self.slotOf(theirs, r, self.idAt(ours, r * k + j)) == null) same = false;
            }
            if (!same) continue;
            for (0..k) |j| {
                const id = self.idAt(ours, r * k + j);
                const s = self.slotOf(theirs, r, id).?;
                std.mem.writeInt(u32, out.items[4 * (r * k + j) ..][0..4], id, .little);
                @memcpy(out.items[4 * nk + wsz * (r * k + j) ..][0..wsz], theirs[4 * nk + wsz * s ..][0..wsz]);
            }
        }
        return out.items;
    }

    fn idAt(_: *const Router, bytes: []const u8, i: usize) u32 {
        return std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
    }

    fn slotOf(self: *const Router, bytes: []const u8, r: usize, id: u32) ?usize {
        const k = self.shape.n;
        for (0..k) |j| if (self.idAt(bytes, r * k + j) == id) return r * k + j;
        return null;
    }

    fn weightAt(self: *const Router, bytes: []const u8, i: usize) f64 {
        return spec.read(host.elemOf(self.out_dt), bytes[4 * self.count() ..], i);
    }

    fn fail(self: *Router, comptime f: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&self.msg, f, args) catch &self.msg;
    }

    /// Gate 1: the stated rule on the planted rows (and distinct in-range ids everywhere). Gate 3 (the gate arm, whose
    /// logit no chain reproduces): the chain's ids and weights per id, the logit within its float64 bound.
    pub fn check(self: *Router, at: spec.Gate) !?[]const u8 {
        const out = self.rig.outputBytes();
        const k = self.shape.n;
        const e = self.shape.k;
        for (0..self.shape.rows) |r| for (0..k) |j| {
            const id = self.idAt(out, r * k + j);
            if (id >= e) return self.fail("row {d}: id {d} out of range", .{ r, id });
            for (0..j) |i| if (self.idAt(out, r * k + i) == id) return self.fail("row {d}: id {d} picked twice", .{ r, id });
        };
        if (at == .nonfinite) {
            const softmax = !self.sigmoidArm();
            for (0..self.shape.rows) |r| {
                for (0..k) |j| {
                    const w = self.weightAt(out, r * k + j);
                    if (r == 0 and self.idAt(out, r * k + j) == self.nan_at) return self.fail("row 0 picked its NaN expert", .{});
                    const nan_row = softmax and (r == 0 or r == 1 or r == 3);
                    if (nan_row != std.math.isNan(w)) return self.fail("row {d} weight {d} is {e}", .{ r, j, w });
                    if (!nan_row and !std.math.isFinite(w)) return self.fail("row {d} weight {d} is {e}", .{ r, j, w });
                }
                // An +Inf key is the first pick (softmax and the raw-logit keys of kolibri1).
                if ((softmax or self.mode == .logit_bias) and r == 1 and self.idAt(out, r * k) != e / 2) return self.fail("row 1 did not pick its +Inf expert first", .{});
            }
            if (self.mode == .softmax_gate) for (0..self.shape.rows) |r| {
                const g = spec.read(host.elemOf(self.out_dt), out[(4 + spec.size(host.elemOf(self.out_dt))) * self.count() ..], r);
                if (!std.math.isFinite(g)) return self.fail("row {d}: gate logit {e} from finite x", .{ r, g });
            };
            return null;
        }
        // The gate arm at gate 3.
        const a = self.rig.a;
        const theirs = try a.dupe(u8, try self.chain());
        defer a.free(theirs);
        for (0..self.shape.rows) |r| for (0..k) |j| {
            const id = self.idAt(out, r * k + j);
            const s = self.slotOf(theirs, r, id) orelse return self.fail("row {d}: id {d} is not in the chain's top-{d}", .{ r, id, k });
            const got = self.weightAt(out, r * k + j);
            const want = spec.read(host.elemOf(self.out_dt), theirs[4 * self.count() ..], s);
            if (@as(u64, @bitCast(got)) != @as(u64, @bitCast(want))) return self.fail("row {d} id {d}: weight {e} where the chain gives {e}", .{ r, id, got, want });
        };
        const gl = out[(4 + spec.size(host.elemOf(self.out_dt))) * self.count() ..];
        const tg: usize = @intCast(rk.moeRouterThreadGroup(@intCast(e)));
        // A thread's strided chain over the gate width, the 32-lane sum, the simdgroups' sum.
        const depth = (self.gk + tg - 33) / (tg - 32) + 5 + tg / 32;
        for (0..self.shape.rows) |r| {
            const d = host.dot(self.gx[r * self.gk ..][0..self.gk], self.gw);
            var mag: f64 = 0;
            for (self.gx[r * self.gk ..][0..self.gk], self.gw) |x, w| mag += @abs(x * w);
            const allow = spec.gamma(@floatFromInt(depth), 0x1p-24) * mag + 0x1p-8 * (@abs(d[0]) + 1e-30);
            const got = spec.read(host.elemOf(self.out_dt), gl, r);
            if (!(@abs(got - d[0]) <= allow)) return self.fail("row {d}: gate logit {e} where float64 gives {e} ({d:.2} of its bound)", .{ r, got, d[0], @abs(got - d[0]) / allow });
        }
        return null;
    }

    fn wrapper(self: *Router) ![]const u8 {
        const s = self.rig.stream();
        const x = self.args;
        const outer = rk.moe_router_fused_override;
        rk.moe_router_fused_override = true;
        defer rk.moe_router_fused_override = outer;
        const mode: RouterMode = if (self.mode == .softmax_gate) .softmax else self.mode;
        const gate: ?rk.RouterGate = if (self.mode == .softmax_gate) .{ .x = x[1], .w = x[2] } else null;
        const bias = if (self.sigmoidArm()) x[1] else mlx.mlx_array{ .ctx = null };
        const r = (try rk.moeRouterTopK(s, x[0], bias, @intCast(self.shape.n), mode, self.norm, self.scale, self.out_dt, @intCast(self.ng), @intCast(self.topkg), gate)) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(r.inds);
        defer _ = mlx.mlx_array_free(r.norm_scores);
        defer if (r.gate_logit.ctx != null) {
            _ = mlx.mlx_array_free(r.gate_logit);
        };
        return if (r.gate_logit.ctx != null) self.rig.bytesOf(&.{ r.inds, r.norm_scores, r.gate_logit }) else self.rig.bytesOf(&.{ r.inds, r.norm_scores });
    }
};

fn routerDepth(_: Shape) usize {
    return 0;
}

const softmax_rule = "ids distinct and in range; a NaN expert is never picked, an +Inf key is picked first; a row holding NaN or +Inf gives NaN weights (MLX's softmax does), every other row finite weights";
const sigmoid_rule = "ids distinct and in range; a NaN expert is never picked (its key fails every compare); the sigmoid saturates +-Inf to 1 and 0, so every weight is finite";

fn routerSpec(comptime name: []const u8, r: Router, served: []const Shape, replaces: ?[]const u8) spec.Spec {
    const step: usize = if (r.mode == .sigmoid_bias_grouped) 64 else 32;
    return .{
        .name = name,
        // Selection is compares; the weights' sums are held to the chain's bits (gate 3) and the outputs hold
        // uint32 ids, which gate 4 cannot read as weights.
        .accum = .none,
        .depth = routerDepth,
        .out = host.elemOf(r.out_dt),
        .replaces = replaces,
        .nan = .{ .stated = if (r.sigmoidArm()) sigmoid_rule else softmax_rule },
        // k: the experts against the threadgroup (one thread an expert up to 256, then strided).
        .tiles = .{ .{}, .{}, .{ .size = 256, .step = step, .min = step } },
        .served = served,
    };
}

fn addRouter(set: *kc.Set, r0: Router, comptime name: []const u8, served: []const Shape, replaces: ?[]const u8) !void {
    var r = r0;
    r.sp = routerSpec(name, r, served, replaces);
    try set.add(Router, r, r.sp, null);
}

fn registerRouter(set: *kc.Set) !void {
    const a = set.a;
    const by_id = " (ids as a set, weights per id)";
    // Laguna / Qwen3 MoE decode and a prefill chunk.
    try addRouter(set, .init(a, .softmax), "router bf16 (softmax)", &.{ .{ .rows = 1, .n = 8, .k = 256 }, .{ .rows = 16, .n = 8, .k = 128 }, .{ .rows = 256, .n = 8, .k = 256 } }, "moeRoutingChain" ++ by_id);
    {
        // Qwen3.5 MoE: 512 experts, top-10, the shared expert's gate over a 2048 hidden.
        var r = Router.init(a, .softmax_gate);
        r.gk = 2048;
        try addRouter(set, r, "router bf16 (softmax, shared-expert gate logit)", &.{ .{ .rows = 1, .n = 10, .k = 512 }, .{ .rows = 64, .n = 10, .k = 512 } }, null);
        r = Router.init(a, .sigmoid_bias);
        r.scale = 2.5;
        try addRouter(set, r, "router bf16 (hy3 sigmoid + bias, normed, scale 2.5)", &.{ .{ .rows = 1, .n = 8, .k = 192 }, .{ .rows = 256, .n = 8, .k = 192 } }, "hy3RoutingChain" ++ by_id);
        r = Router.init(a, .sigmoid_bias);
        r.norm = false;
        try addRouter(set, r, "router bf16 (hy3 sigmoid + bias, unnormed)", &.{.{ .rows = 4, .n = 8, .k = 256 }}, "hy3RoutingChain" ++ by_id);
        // DeepSeek-style group limit at the shipped 512 / 8 groups / 4 kept / top-8, f32 logits.
        r = Router.init(a, .sigmoid_bias_grouped);
        r.dt = .float32;
        r.out_dt = .float32;
        r.scale = 2.5;
        r.ng = 8;
        r.topkg = 4;
        try addRouter(set, r, "router f32 (grouped sigmoid + bias, 8 groups keep 4)", &.{ .{ .rows = 1, .n = 8, .k = 512 }, .{ .rows = 256, .n = 8, .k = 512 }, .{ .rows = 8, .n = 8, .k = 256 } }, "groupLimitedRouting" ++ by_id);
        // Kolibri-1: 384 experts, top-6 on logit + bias, f32.
        r = Router.init(a, .logit_bias);
        r.dt = .float32;
        r.out_dt = .float32;
        r.norm = false;
        try addRouter(set, r, "router f32 (kolibri1 logit + bias)", &.{ .{ .rows = 1, .n = 6, .k = 384 }, .{ .rows = 256, .n = 6, .k = 384 } }, "kolibriRoutingChain" ++ by_id);
    }
}

// ── the verify and prefill routing kernels ──

/// MLX's affine qmv over one row of a bank: per group s * sum(x q) + b * sum(x), and its terms' magnitude.
fn factoredDot(b: *const Bank, r: usize, x: []const f64) [2]f64 {
    const k = x.len;
    const g = b.aff.gs;
    var y: f64 = 0;
    var mag: f64 = 0;
    var at: usize = 0;
    while (at < k) : (at += g) {
        var xq: f64 = 0;
        var xs: f64 = 0;
        var aq: f64 = 0;
        var ax: f64 = 0;
        for (at..at + g) |i| {
            const q: f64 = @floatFromInt(b.aff.codes[r * k + i]);
            xq += x[i] * q;
            xs += x[i];
            aq += @abs(x[i] * q);
            ax += @abs(x[i]);
        }
        const s = b.aff.scales[(r * k + at) / g];
        const bb = b.aff.biases[(r * k + at) / g];
        y += s * xq + bb * xs;
        mag += @abs(s) * aq + @abs(bb) * ax;
    }
    return .{ y, mag };
}

/// A row's x * q terms (the factored sum's, at scale 1 and bias 0) summed in `acc`.
fn simDot(b: *const Bank, r: usize, x: []const f64, acc: spec.Accum, order: spec.Order, t: []f64) f64 {
    for (t, x, 0..) |*o, v, i| o.* = v * @as(f64, @floatFromInt(b.aff.codes[r * x.len + i]));
    return spec.sumIn(acc, order, t);
}

const reuse_launch = metal.Launch{ .name = "verify_expert_reuse", .inputs = &.{ "x", "w", "sc", "bi", "ids" }, .outputs = &.{"y"}, .source = vk.VERIFY_EXPERT_REUSE_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 };

/// `verify_expert_reuse`: each routed row's expert projection, a run's rows in pairs sharing each packed word, as
/// MLX's gather_qmv sums. Rows are routed rows (sorted runs), n = N, k = K (640, the Flash Next down projection).
const Reuse = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &reuse_launch,
    bank: Bank = Bank.init(.affine, 4, 64),
    ne: usize = 0,
    shape: Shape = undefined,
    x: []f64 = &.{},
    ids: []f64 = &.{},
    dv: []f64 = &.{},
    dm: []f64 = &.{},
    args: [5]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) Reuse {
        return .{ .rig = metal.Rig.init(a) };
    }

    pub fn deinit(self: *Reuse) void {
        const a = self.rig.a;
        self.rig.deinit();
        self.bank.deinit(a);
        for ([_][]f64{ self.x, self.ids, self.dv, self.dm }) |s| a.free(s);
    }

    pub fn geometry(_: *const Reuse, shape: Shape) ?metal.Geometry {
        if (shape.rows < 2 or shape.n % 8 != 0 or shape.k % 64 != 0) return null;
        var g = metal.Geometry{ .grid = .{ 32, @intCast(shape.n / 4), @intCast(shape.rows) }, .tg = .{ 32, 2, 1 } };
        g.out(&.{ @intCast(shape.rows), 1, @intCast(shape.n) }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("K", shape.k);
        g.int("N", shape.n);
        return g;
    }

    pub fn prepare(self: *Reuse, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        for ([_]*[]f64{ &self.x, &self.ids, &self.dv, &self.dm }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        // Under four rows an expert, where MLX gathers row by row (its sorted matrix path starts at four).
        self.ne = @max(8, shape.rows / 3 + 1);
        try self.bank.fill(a, rng, self.ne * shape.n, shape.k, .bfloat16, fill == .stress);
        self.x = try a.alloc(f64, shape.rows * shape.k);
        if (fill == .stress) stressX(rng, self.x, shape.k) else host.uniformInto(rng, self.x, -2, 2, .bfloat16);
        // Sorted runs of 1..6 rows, ascending experts.
        self.ids = try a.alloc(f64, shape.rows);
        var e: usize = 0;
        var left: usize = 1 + rng.uintLessThan(usize, 6);
        for (self.ids) |*v| {
            if (left == 0) {
                e = @min(e + 1, self.ne - 1);
                left = 1 + rng.uintLessThan(usize, 6);
            }
            v.* = @floatFromInt(e);
            left -= 1;
        }
        if (fill == .nonfinite) plantRows(self.x, shape.k);
        const rig = &self.rig;
        rig.reset();
        const ne: c_int = @intCast(self.ne);
        const n: c_int = @intCast(shape.n);
        const k: c_int = @intCast(shape.k);
        self.args = .{
            try rig.input(.bfloat16, &.{ @intCast(shape.rows), 1, k }, self.x),
            try rig.input(.uint32, &.{ ne, n, @divExact(k, 8) }, self.bank.aff.words),
            try rig.input(.bfloat16, &.{ ne, n, @divExact(k, 64) }, self.bank.aff.scales),
            try rig.input(.bfloat16, &.{ ne, n, @divExact(k, 64) }, self.bank.aff.biases),
            try rig.input(.uint32, &.{@intCast(shape.rows)}, self.ids),
        };
        return true;
    }

    pub fn run(self: *Reuse) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Reuse) usize {
        return self.shape.rows * self.shape.n;
    }

    fn dots(self: *Reuse) !void {
        if (self.dv.len > 0) return;
        const a = self.rig.a;
        const n = self.shape.n;
        const k = self.shape.k;
        self.dv = try a.alloc(f64, self.count());
        self.dm = try a.alloc(f64, self.count());
        for (0..self.shape.rows) |r| for (0..n) |j| {
            const e: usize = @intFromFloat(self.ids[r]);
            const d = factoredDot(&self.bank, e * n + j, self.x[r * k ..][0..k]);
            self.dv[r * n + j] = d[0];
            self.dm[r * n + j] = d[1];
        };
    }

    pub fn reference(self: *Reuse, want: []f64, mag: []f64) !void {
        try self.dots();
        @memcpy(want, self.dv);
        @memcpy(mag, self.dm);
    }

    pub fn simulate(self: *Reuse, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.dots();
        @memcpy(out, self.dv);
        if (acc == .f32) {
            for (self.dm) |m| if (m >= 0x1p24) return error.StressPastExact;
            return;
        }
        const a = self.rig.a;
        const n = self.shape.n;
        const k = self.shape.k;
        const t = try a.alloc(f64, k);
        defer a.free(t);
        for (0..self.shape.rows) |r| for (0..@min(32, n)) |j| {
            const e: usize = @intFromFloat(self.ids[r]);
            out[r * n + j] = simDot(&self.bank, e * n + j, self.x[r * k ..][0..k], acc, order, t);
        };
    }

    pub fn replaced(self: *Reuse) ![]const u8 {
        var y = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y);
        try experts.gatherExpertMm(&y, self.args[0], self.args[1], self.args[2], self.args[3], .{ .ctx = null }, self.args[4], 4, 64, .affine, true, self.rig.stream());
        return self.rig.bytesOf(&.{y});
    }

    fn wrapper(self: *Reuse) ![]const u8 {
        const y = (try vk.verifyExpertReuse(self.rig.stream(), self.args[0], self.args[1], self.args[2], self.args[3], self.args[4])) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

/// NaN, +Inf and -Inf in the first three rows of a row-major [rows, k] input (each row feeds only its own outputs).
fn plantRows(v: []f64, k: usize) void {
    const at = [_]struct { usize, f64 }{ .{ k / 3, std.math.nan(f64) }, .{ k / 2, std.math.inf(f64) }, .{ k - 1, -std.math.inf(f64) } };
    for (at, 0..) |p, r| if ((r + 1) * k <= v.len) {
        v[r * k + p[0]] = p[1];
    };
}

const reduce_launch = metal.Launch{ .name = "verify_expert_reduce", .inputs = &.{ "down", "inverse", "scores" }, .outputs = &.{"y"}, .source = vk.VERIFY_EXPERT_REDUCE_SOURCE, .tg_bytes = 0, .m1 = 928, .m2 = 928 };

/// `verify_expert_reduce`: the routed down rows unsorted, weighted and summed over the top-k in bf16, MLX's eight
/// partial sums (col_reduce_small). Rows are tokens, n = hidden, k = top-k.
const Reduce = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &reduce_launch,
    shape: Shape = undefined,
    down: []f64 = &.{},
    inverse: []f64 = &.{},
    scores: []f64 = &.{},
    args: [3]mlx.mlx_array = undefined,

    fn init(a: std.mem.Allocator) Reduce {
        return .{ .rig = metal.Rig.init(a) };
    }

    pub fn deinit(self: *Reduce) void {
        const a = self.rig.a;
        self.rig.deinit();
        for ([_][]f64{ self.down, self.inverse, self.scores }) |s| a.free(s);
    }

    pub fn geometry(_: *const Reduce, shape: Shape) ?metal.Geometry {
        if (shape.rows < 2 or shape.n == 0 or shape.k == 0) return null;
        var g = metal.Geometry{ .grid = .{ @intCast(shape.rows * shape.n), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{ 1, @intCast(shape.rows), @intCast(shape.n) }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("TOPK", shape.k);
        g.int("H", shape.n);
        return g;
    }

    pub fn prepare(self: *Reduce, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        for ([_]*[]f64{ &self.down, &self.inverse, &self.scores }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const m = shape.rows * shape.k;
        self.down = try a.alloc(f64, m * shape.n);
        host.uniformInto(rng, self.down, -2, 2, .bfloat16);
        self.scores = try a.alloc(f64, m);
        host.uniformInto(rng, self.scores, 0, 1, .bfloat16);
        const perm = try a.alloc(u32, m);
        defer a.free(perm);
        for (perm, 0..) |*p, i| p.* = @intCast(i);
        rng.shuffle(u32, perm);
        self.inverse = try a.alloc(f64, m);
        for (self.inverse, perm) |*o, p| o.* = @floatFromInt(p);
        if (fill == .nonfinite) {
            // Token 0's first expert row takes NaN, token 1's +Inf, token 2's -Inf (one column each).
            const h = shape.n;
            const at = [_]struct { usize, f64 }{ .{ h / 3, std.math.nan(f64) }, .{ h / 2, std.math.inf(f64) }, .{ h - 1, -std.math.inf(f64) } };
            for (at, 0..) |p, t| {
                const row: usize = perm[t * shape.k];
                self.down[row * h + p[0]] = p[1];
            }
        }
        const rig = &self.rig;
        rig.reset();
        self.args = .{
            try rig.input(.bfloat16, &.{ @intCast(m), @intCast(shape.n) }, self.down),
            try rig.input(.uint32, &.{@intCast(m)}, self.inverse),
            try rig.input(.bfloat16, &.{ 1, @intCast(shape.rows), @intCast(shape.k) }, self.scores),
        };
        return true;
    }

    pub fn run(self: *Reduce) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const Reduce) usize {
        return self.shape.rows * self.shape.n;
    }

    pub fn reference(self: *Reduce, want: []f64, mag: []f64) !void {
        const h = self.shape.n;
        const k = self.shape.k;
        for (want, mag, 0..) |*y, *m, p| {
            const r = p / h;
            const c = p % h;
            y.* = 0;
            m.* = 0;
            for (0..k) |e| {
                const at = r * k + e;
                const row: usize = @intFromFloat(self.inverse[at]);
                y.* += self.down[row * h + c] * self.scores[at];
                m.* += @abs(self.down[row * h + c] * self.scores[at]);
            }
        }
    }

    /// take (unsort) -> multiply by the scores -> sum over the top-k, MLX's ops in bf16.
    pub fn replaced(self: *Reduce) ![]const u8 {
        const s = self.rig.stream();
        var g = ops.Graph.init(self.rig.a, s);
        defer g.deinit();
        const x = self.args;
        const rows = try g.reshape(try g.take(x[0], x[1], 0), &.{ 1, @intCast(self.shape.rows), @intCast(self.shape.k), @intCast(self.shape.n) });
        return self.rig.bytesOf(&.{try g.sum(try g.mul(rows, try g.expandDims(x[2], -1)), -2, false)});
    }

    fn wrapper(self: *Reduce) ![]const u8 {
        const s = self.rig.stream();
        const x = self.args;
        const y = (if (self.shape.rows <= 32) try vk.verifyExpertReduce(s, x[0], x[1], x[2]) else try vk.moePrefillExpertReduce(s, x[0], x[1], x[2])) orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

/// The stable argsort of ids, its inverse and each sorted position's row (id index / top-k), on the host.
fn stablePack(a: std.mem.Allocator, ids: []const f64, topk: usize) ![3][]u32 {
    const n = ids.len;
    const order = try a.alloc(u32, n);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    const Ctx = struct {
        ids: []const f64,
        fn less(c: @This(), x: u32, y: u32) bool {
            return c.ids[x] < c.ids[y] or (c.ids[x] == c.ids[y] and x < y);
        }
    };
    std.mem.sort(u32, order, Ctx{ .ids = ids }, Ctx.less);
    const inverse = try a.alloc(u32, n);
    const sorted = try a.alloc(u32, n);
    const lhs = try a.alloc(u32, n);
    for (order, 0..) |o, pos| {
        inverse[o] = @intCast(pos);
        sorted[pos] = @intFromFloat(ids[o]);
        lhs[pos] = @intCast(o / topk);
    }
    a.free(order);
    return .{ inverse, sorted, lhs };
}

const RouteKind = enum { pack, hist, scan, scatter };

const route_launches = [4]metal.Launch{
    .{ .name = "verify_route_pack", .inputs = &.{"ids"}, .outputs = &.{ "inverse", "sorted", "lhs" }, .source = vk.VERIFY_ROUTE_PACK_SOURCE, .tg_bytes = 320 * 4, .m1 = 1024, .m2 = 1024 },
    .{ .name = "prefill_route_hist", .inputs = &.{"ids"}, .outputs = &.{"counts"}, .source = vk.PREFILL_ROUTE_HIST_SOURCE, .tg_bytes = 1024 * 4, .m1 = 1024, .m2 = 1024 },
    .{ .name = "prefill_route_scan", .inputs = &.{"counts"}, .outputs = &.{"offsets"}, .source = vk.PREFILL_ROUTE_SCAN_SOURCE, .tg_bytes = 1024 * 4, .m1 = 1024, .m2 = 1024 },
    .{ .name = "prefill_route_scatter", .inputs = &.{ "ids", "offsets" }, .outputs = &.{ "inverse", "sorted", "lhs" }, .source = vk.PREFILL_ROUTE_SCATTER_SOURCE, .tg_bytes = 256 * 4, .m1 = 1024, .m2 = 1024 },
};

/// The routed-id sorts: verify's all-pairs rank (`verify_route_pack`) and prefill's counting sort in three
/// dispatches (`prefill_route_*`). Integer in, integer out. Rows are tokens, k the top-k; the experts are fixed.
const Route = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kind: RouteKind,
    ne: usize,
    shape: Shape = undefined,
    ids: []f64 = &.{},
    counts: []u32 = &.{},
    args: [2]mlx.mlx_array = undefined,
    n_args: usize = 1,
    msg: [256]u8 = undefined,

    const BLOCK: usize = 256;

    fn init(a: std.mem.Allocator, kind: RouteKind, ne: usize) Route {
        return .{ .rig = metal.Rig.init(a), .launch = &route_launches[@backingInt(kind)], .kind = kind, .ne = ne };
    }

    pub fn deinit(self: *Route) void {
        const a = self.rig.a;
        self.rig.deinit();
        a.free(self.ids);
        a.free(self.counts);
    }

    fn total(shape: Shape) usize {
        return shape.rows * shape.k;
    }

    fn blocks(shape: Shape) usize {
        return (total(shape) + BLOCK - 1) / BLOCK;
    }

    pub fn geometry(self: *const Route, shape: Shape) ?metal.Geometry {
        const n = total(shape);
        if (n == 0) return null;
        const ni: c_int = @intCast(n);
        const table: c_int = @intCast(blocks(shape) * self.ne);
        var g: metal.Geometry = undefined;
        switch (self.kind) {
            .pack => {
                if (shape.k != 10 or n < 20 or n > 320) return null;
                g = .{ .grid = .{ 128, 1, 1 }, .tg = .{ 128, 1, 1 } };
                for (0..3) |_| g.out(&.{ni}, .uint32);
                return g;
            },
            .hist, .scatter => g = .{ .grid = .{ @intCast(blocks(shape) * BLOCK), 1, 1 }, .tg = .{ 256, 1, 1 } },
            .scan => g = .{ .grid = .{ 256, 1, 1 }, .tg = .{ 256, 1, 1 } },
        }
        if (self.kind == .scatter) {
            for (0..3) |_| g.out(&.{ni}, .uint32);
        } else g.out(&.{table}, .uint32);
        g.int("E", self.ne);
        g.int("TOPK", shape.k);
        return g;
    }

    pub fn prepare(self: *Route, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        a.free(self.ids);
        a.free(self.counts);
        self.ids = &.{};
        self.counts = &.{};
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = total(shape);
        self.ids = try a.alloc(f64, n);
        // Skewed, as routing is: half the ids on four hot experts.
        for (self.ids) |*v| v.* = @floatFromInt(if (rng.boolean()) rng.uintLessThan(usize, 4) else rng.uintLessThan(usize, self.ne));
        if (fill == .nonfinite) {
            // No float to make non-finite: the extremes instead, the last expert and the first, and (for the
            // histogram) ids past the bank, which it must not count.
            for (self.ids[0 .. n / 3]) |*v| v.* = @floatFromInt(self.ne - 1);
            for (self.ids[n / 3 .. 2 * n / 3]) |*v| v.* = 0;
            if (self.kind == .hist) self.ids[n - 1] = @floatFromInt(self.ne + 7);
            if (self.kind == .pack) self.ids[n - 1] = 0xffffffff;
        }
        // The histogram per 256-id block (the scan's and scatter's input).
        self.counts = try a.alloc(u32, blocks(shape) * self.ne);
        @memset(self.counts, 0);
        for (self.ids, 0..) |v, i| if (v < @as(f64, @floatFromInt(self.ne))) {
            self.counts[(i / BLOCK) * self.ne + @as(usize, @intFromFloat(v))] += 1;
        };
        const rig = &self.rig;
        rig.reset();
        const ni: c_int = @intCast(n);
        switch (self.kind) {
            .pack, .hist => {
                self.args[0] = try rig.input(.uint32, &.{ni}, self.ids);
                self.n_args = 1;
            },
            .scan => {
                const c = try a.alloc(f64, self.counts.len);
                defer a.free(c);
                for (c, self.counts) |*o, v| o.* = @floatFromInt(v);
                self.args[0] = try rig.input(.uint32, &.{@intCast(c.len)}, c);
                self.n_args = 1;
            },
            .scatter => {
                const off = try self.offsets();
                defer a.free(off);
                const c = try a.alloc(f64, off.len);
                defer a.free(c);
                for (c, off) |*o, v| o.* = @floatFromInt(v);
                self.args[0] = try rig.input(.uint32, &.{ni}, self.ids);
                self.args[1] = try rig.input(.uint32, &.{@intCast(c.len)}, c);
                self.n_args = 2;
            },
        }
        return true;
    }

    /// Each (block, expert)'s first sorted position: expert-major, blocks in order.
    fn offsets(self: *const Route) ![]u32 {
        const nb = blocks(self.shape);
        const out = try self.rig.a.alloc(u32, self.counts.len);
        var at: u32 = 0;
        for (0..self.ne) |e| for (0..nb) |b| {
            out[b * self.ne + e] = at;
            at += self.counts[b * self.ne + e];
        };
        return out;
    }

    pub fn run(self: *Route) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const Route) usize {
        return if (self.kind == .pack or self.kind == .scatter) 3 * total(self.shape) else self.counts.len;
    }

    pub fn reference(_: *Route, _: []f64, _: []f64) !void {
        return error.IntegerKernelHasNoFloat64Outputs;
    }

    /// Exact outputs: the stable argsort (`pack`, `scatter`), the per-block histogram, its offsets.
    fn want(self: *Route) ![]u32 {
        const a = self.rig.a;
        switch (self.kind) {
            .pack, .scatter => {
                const p = try stablePack(a, self.ids, self.shape.k);
                defer for (p) |v| a.free(v);
                const out = try a.alloc(u32, 3 * self.ids.len);
                for (p, 0..) |v, i| @memcpy(out[i * self.ids.len ..][0..self.ids.len], v);
                return out;
            },
            .hist => return a.dupe(u32, self.counts),
            .scan => return self.offsets(),
        }
    }

    /// The stated rule at gate 1, the exact outputs at gate 3 (the histogram and its scan, which no op chain makes).
    pub fn check(self: *Route, _: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        const w = try self.want();
        defer a.free(w);
        const out = self.rig.outputBytes();
        for (w, 0..) |v, i| {
            const got = std.mem.readInt(u32, out[4 * i ..][0..4], .little);
            if (got != v) return std.fmt.bufPrint(&self.msg, "output {d} is {d} where the exact {t} gives {d}", .{ i, got, self.kind, v }) catch &self.msg;
        }
        return null;
    }

    /// MLX's stable argsort, its inverse, the sorted ids and their rows.
    pub fn replaced(self: *Route) ![]const u8 {
        const s = self.rig.stream();
        var g = ops.Graph.init(self.rig.a, s);
        defer g.deinit();
        const ids = self.args[0];
        const order = try g.argsort(ids, 0);
        const k = try g.adopt(mlx.mlx_array_new_int(@intCast(self.shape.k)));
        const rows = try g.astype(try g.call(mlx.mlx_floor_divide, .{ order, k }), .uint32);
        return self.rig.bytesOf(&.{ try g.argsort(order, 0), try g.take(ids, order, 0), rows });
    }
};

fn routeDepth(_: Shape) usize {
    return 0;
}

const route_rule = "integer ids in, integer positions out: no non-finite value exists; the extremes (the first and last expert, an id past the bank or 0xffffffff where the kernel reads one) keep every output exact";

fn routeSpec(comptime name: []const u8, served: []const Shape, replaces: ?[]const u8) spec.Spec {
    return .{ .name = name, .accum = .none, .depth = routeDepth, .out = .f32, .replaces = replaces, .nan = .{ .stated = route_rule }, .served = served };
}

fn gqDepth(shape: Shape) usize {
    return shape.k + 4;
}

fn registerVerify(set: *kc.Set) !void {
    const a = set.a;
    const gather = "MLX gather_qmm (sorted rhs indices, row by row below four rows an expert)";
    // Flash Next's verify widths: 2..8 tokens of top-10 routed rows; N cut to 256 (a template argument; the drift
    // guard runs the wrapper's 2560).
    try set.add(Reuse, .init(a), .{ .name = "verify_expert_reuse bf16 (4-bit g64, K 640)", .accum = .f32, .depth = gqDepth, .out = .bf16, .replaces = gather, .tiles = .{ .{}, .{ .size = 16, .step = 8, .min = 8 }, .{} }, .served = &.{ .{ .rows = 20, .n = 256, .k = 640 }, .{ .rows = 80, .n = 256, .k = 640 } } }, null);
    const reduce_chain = "take -> multiply -> sum (MLX col_reduce_small, bf16)";
    // Verify (2..32 tokens) and prefill chunks, at the two (top-k, hidden) pairs whose partial-sum tree it copies.
    try set.add(Reduce, .init(a), .{ .name = "verify_expert_reduce bf16 (top-10, hidden 2560)", .accum = .bf16, .depth = routeDepth, .out = .bf16, .replaces = reduce_chain, .served = &.{ .{ .rows = 4, .n = 2560, .k = 10 }, .{ .rows = 32, .n = 2560, .k = 10 }, .{ .rows = 1024, .n = 2560, .k = 10 } } }, null);
    try set.add(Reduce, .init(a), .{ .name = "verify_expert_reduce bf16 (top-8, hidden 4096)", .accum = .bf16, .depth = routeDepth, .out = .bf16, .replaces = reduce_chain, .served = &.{ .{ .rows = 4, .n = 4096, .k = 8 }, .{ .rows = 512, .n = 4096, .k = 8 } } }, null);
    const argsort = "argsort -> argsort (inverse), take, floor_divide (MLX, stable)";
    try set.add(Route, .init(a, .pack, 512), routeSpec("verify_route_pack (top-10, 2..32 tokens)", &.{ .{ .rows = 2, .n = 1, .k = 10 }, .{ .rows = 8, .n = 1, .k = 10 }, .{ .rows = 32, .n = 1, .k = 10 } }, argsort), null);
    const prefill: []const Shape = &.{ .{ .rows = 33, .n = 1, .k = 8 }, .{ .rows = 1000, .n = 1, .k = 8 } };
    try set.add(Route, .init(a, .hist, 256), routeSpec("prefill_route_hist (256 experts, top-8)", prefill, null), null);
    try set.add(Route, .init(a, .scan, 256), routeSpec("prefill_route_scan (256 experts, top-8)", prefill, null), null);
    try set.add(Route, .init(a, .scatter, 256), routeSpec("prefill_route_scatter (256 experts, top-8)", prefill, argsort), null);
    try set.add(Route, .init(a, .scatter, 512), routeSpec("prefill_route_scatter (512 experts, top-10)", &.{.{ .rows = 100, .n = 1, .k = 10 }}, argsort), null);
}

// ── the NAX sorted gather (prefill) ──

const ksrc = @import("../../../mlx/kernels.zig").source;
const gqn = @import("gather_qmm_nax.zig");
const hc_prefill = @import("../hc/hc_prefill.zig");
const nax_header = ksrc("kernels/moe/gather_qmm_nax_header.metal");
const nax_mapped_header = nax_header ++ ksrc("kernels/moe/gather_qmm_nax_mapped_header.metal");
const nax_mm_in = [_][*:0]const u8{ "x", "w", "scales", "biases", "tiles", "tile_count", "params" };
const nax_map_in = [_][*:0]const u8{ "x", "w", "scales", "biases", "up_w", "up_scales", "up_biases", "tiles", "tile_count", "params", "sigtab", "rmap" };

/// `Ws` at the largest plan: seg 128 x 128, one 64 x (128 + 8) bf16 tile and its 8 KiB pad.
const nax_tg: u32 = 64 * 136 * 2 + 8192;
const scan_launch = metal.Launch{ .name = "msv_gqmm_tile_scan", .inputs = &.{ "idx", "params" }, .outputs = &.{ "tiles", "tile_count" }, .source = ksrc("kernels/moe/gather_qmm_nax_scan.metal"), .header = nax_header, .tg_bytes = 2 * 2048 * 4 + 32 * 4, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 17 };
const NaxKind = enum { mm, mapped, paired };
const nax_launches = [3]metal.Launch{
    .{ .name = "msv_gqmm_affine", .inputs = &nax_mm_in, .outputs = &.{"y"}, .source = ksrc("kernels/moe/gather_qmm_nax.metal"), .header = nax_header, .tg_bytes = nax_tg, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 17 },
    .{ .name = "msv_gqmm_affine_map", .inputs = &nax_map_in, .outputs = &.{"y"}, .source = ksrc("kernels/moe/gather_qmm_nax_mapped.metal"), .header = nax_mapped_header, .tg_bytes = nax_tg, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 17 },
    .{ .name = "msv_gqmm_affine_pair", .inputs = &nax_map_in, .outputs = &.{"y"}, .source = ksrc("kernels/moe/gather_qmm_nax_mapped.metal"), .header = nax_mapped_header, .tg_bytes = nax_tg, .m1 = 1024, .m2 = 1024, .offline = false, .min_gen = 17 },
};

/// Sorted ragged expert runs over `ne` experts: one in sixteen empty, one in eight four times the mean, the
/// remainder on the last.
fn raggedIds(ids: []f64, ne: usize) void {
    const weight = struct {
        fn of(e: usize) usize {
            return if (e % 16 == 5) 0 else if (e % 8 == 0) 4 else 1;
        }
    }.of;
    var total: usize = 0;
    for (0..ne) |e| total += weight(e);
    var pos: usize = 0;
    for (0..ne) |e| for (0..@min(ids.len * weight(e) / total, ids.len - pos)) |_| {
        ids[pos] = @floatFromInt(e);
        pos += 1;
    };
    while (pos < ids.len) : (pos += 1) ids[pos] = @floatFromInt(ne - 1);
}

/// The tile scan on the host: per expert in order its run cut into tiles of at most `bm` rows
/// (start, expert, rows, 0); the count capped at `max_tiles`.
fn hostTiles(ids: []const f64, ne: usize, bm: usize, max_tiles: usize, tiles: []u32) u32 {
    @memset(tiles, 0);
    var t: usize = 0;
    for (0..ne) |e| {
        var start: usize = 0;
        var end: usize = 0;
        for (ids, 0..) |v, i| if (@as(usize, @intFromFloat(v)) == e) {
            if (end == 0) start = i;
            end = i + 1;
        };
        const cnt = end - start;
        for (0..(cnt + bm - 1) / bm) |j| {
            if (t < max_tiles) {
                const r = start + j * bm;
                tiles[4 * t ..][0..4].* = .{ @intCast(r), @intCast(e), @intCast(@min(bm, start + cnt - r)), 0 };
            }
            t += 1;
        }
    }
    return @intCast(@min(t, max_tiles));
}

fn naxMaxTiles(rows: usize, ne: usize, bm: usize) usize {
    return (rows + bm - 1) / bm + @min(ne, rows);
}

/// `msv_gqmm_tile_scan`: the expert runs of the sorted ids cut into BM-row tiles. Rows are routed rows, n the BM
/// (64, 96 or 128), k unused; the experts are the registration's.
const NaxScan = struct {
    rig: metal.Rig,
    launch: *const metal.Launch = &scan_launch,
    ne: usize,
    shape: Shape = undefined,
    ids: []f64 = &.{},
    tiles: []u32 = &.{},
    count_: u32 = 0,
    args: [2]mlx.mlx_array = undefined,
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator, ne: usize) NaxScan {
        return .{ .rig = metal.Rig.init(a), .ne = ne };
    }

    pub fn deinit(self: *NaxScan) void {
        const a = self.rig.a;
        self.rig.deinit();
        a.free(self.ids);
        a.free(self.tiles);
    }

    fn maxTiles(self: *const NaxScan, shape: Shape) usize {
        return naxMaxTiles(shape.rows, self.ne, shape.n);
    }

    pub fn geometry(self: *const NaxScan, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or self.ne > 2048 or (shape.n != 64 and shape.n != 96 and shape.n != 128)) return null;
        var g = metal.Geometry{ .grid = .{ 1024, 1, 1 }, .tg = .{ 1024, 1, 1 } };
        g.out(&.{@intCast(4 * self.maxTiles(shape))}, .uint32);
        g.out(&.{1}, .uint32);
        g.int("BM", shape.n);
        g.int("MAXE", 2048);
        return g;
    }

    pub fn prepare(self: *NaxScan, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        a.free(self.ids);
        a.free(self.tiles);
        self.ids = try a.alloc(f64, shape.rows);
        raggedIds(self.ids, self.ne);
        if (fill != .random) {
            // The extremes: every row on the last expert, or (shuffled seed) on the first.
            var prng = std.Random.DefaultPrng.init(seed);
            @memset(self.ids, if (prng.random().boolean()) @as(f64, @floatFromInt(self.ne - 1)) else 0);
        }
        const mt = self.maxTiles(shape);
        self.tiles = try a.alloc(u32, 4 * mt);
        self.count_ = hostTiles(self.ids, self.ne, shape.n, mt, self.tiles);
        const rig = &self.rig;
        rig.reset();
        self.args = .{
            try rig.input(.uint32, &.{@intCast(shape.rows)}, self.ids),
            try rig.input(.int32, &.{3}, &.{ @floatFromInt(shape.rows), @floatFromInt(self.ne), @floatFromInt(mt) }),
        };
        return true;
    }

    pub fn run(self: *NaxScan) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &self.args);
    }

    pub fn count(self: *const NaxScan) usize {
        return self.tiles.len + 1;
    }

    pub fn reference(_: *NaxScan, _: []f64, _: []f64) !void {
        return error.IntegerKernelHasNoFloat64Outputs;
    }

    /// The tiles and the count exactly (the words past the count are never written, nor read).
    pub fn check(self: *NaxScan, _: spec.Gate) !?[]const u8 {
        const out = self.rig.outputBytes();
        const tc = std.mem.readInt(u32, out[4 * self.tiles.len ..][0..4], .little);
        if (tc != self.count_) return std.fmt.bufPrint(&self.msg, "tile count {d} where the exact scan gives {d}", .{ tc, self.count_ }) catch &self.msg;
        for (self.tiles[0 .. 4 * tc], 0..) |want, i| {
            const got = std.mem.readInt(u32, out[4 * i ..][0..4], .little);
            if (got != want) return std.fmt.bufPrint(&self.msg, "tile word {d} is {d} where the exact scan gives {d}", .{ i, got, want }) catch &self.msg;
        }
        return null;
    }
};

/// The NAX sorted gather-matmul over the scan's tiles: plain (`msv_gqmm_affine`), x read through a row map
/// (`_map`), and gate and up with the SwiGLU in the epilogue (`_pair`). Plain: rows are routed rows sorted by
/// expert; mapped: rows are tokens, four routed rows each. n = N, k = K; the bank's experts are the registration's.
const GqmmNax = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    kind: NaxKind,
    ne: usize,
    limit: u32 = 0,
    banks: [2]Bank,
    shape: Shape = undefined,
    m: usize = 0,
    x: []f64 = &.{},
    rmap: []f64 = &.{},
    ids: []f64 = &.{},
    table: []f64 = &.{},
    dv: []f64 = &.{},
    dm: []f64 = &.{},
    args: [12]mlx.mlx_array = undefined,
    n_args: usize = 0,

    fn init(a: std.mem.Allocator, kind: NaxKind, f: Fmt, b: u32, g: usize, ne: usize) GqmmNax {
        const bank = Bank.init(f, b, g);
        return .{ .rig = metal.Rig.init(a), .launch = &nax_launches[@backingInt(kind)], .kind = kind, .ne = ne, .banks = .{ bank, bank } };
    }

    pub fn deinit(self: *GqmmNax) void {
        const a = self.rig.a;
        self.rig.deinit();
        for (&self.banks) |*b| b.deinit(a);
        for ([_][]f64{ self.x, self.rmap, self.ids, self.table, self.dv, self.dm }) |s| a.free(s);
    }

    fn mx(self: *const GqmmNax) bool {
        return self.banks[0].fmt != .affine;
    }

    fn nb(self: *const GqmmNax) usize {
        return if (self.kind == .paired) 2 else 1;
    }

    /// Routed rows: the shape's rows, or four a token through the row map (a top-4 routing).
    fn routed(self: *const GqmmNax, shape: Shape) usize {
        return if (self.kind == .mm) shape.rows else 4 * shape.rows;
    }

    fn planAt(self: *const GqmmNax, shape: Shape) gqn.Plan {
        return gqn.plan(@intCast(self.routed(shape)), @intCast(self.ne), @intCast(shape.k), @intCast(shape.n), self.kind == .paired);
    }

    pub fn geometry(self: *const GqmmNax, shape: Shape) ?metal.Geometry {
        if (shape.rows == 0 or shape.n % 32 != 0 or shape.k % 32 != 0 or shape.k % self.banks[0].aff.gs != 0) return null;
        const p = self.planAt(shape);
        const mt: c_int = @intCast(naxMaxTiles(self.routed(shape), self.ne, @intCast(p.bm)));
        const n: c_int = @intCast(shape.n);
        const width = if (self.kind == .paired) 2 * n else n;
        const cols = @divTrunc(width + 63, 64);
        const gx = if (p.gx > 0) p.gx else cols;
        const gy = if (p.gx > 0) @divTrunc(mt + p.gx - 1, p.gx) * cols else mt;
        var g = metal.Geometry{ .grid = .{ @intCast(gx * 32), @intCast(gy * 2), @intCast(@divTrunc(p.bm, 32)) }, .tg = .{ 32, 2, @intCast(@divTrunc(p.bm, 32)) } };
        g.out(&.{ @intCast(self.routed(shape)), 1, n }, .bfloat16);
        g.dtype("T", .bfloat16);
        g.int("GS", self.banks[0].aff.gs);
        g.int("BITS", self.banks[0].aff.bits);
        g.int("MX", @intFromBool(self.mx()));
        g.int("SCHED", @intFromBool(p.sched == .db));
        g.flag("ALIGN_N", @rem(width, 64) == 0);
        g.flag("ALIGN_K", @rem(@as(c_int, @intCast(shape.k)), p.bk) == 0);
        g.int("BM", p.bm);
        g.int("BK", p.bk);
        g.int("GX", p.gx);
        g.int("PAD", p.pad);
        if (self.kind != .mm) g.int("EPI", if (self.kind == .paired) 1 + self.limit else 0);
        return g;
    }

    pub fn prepare(self: *GqmmNax, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (!device_mod.naxAvailable()) return error.NaxOnly;
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        for ([_]*[]f64{ &self.x, &self.rmap, &self.ids, &self.dv, &self.dm }) |p| {
            a.free(p.*);
            p.* = &.{};
        }
        if (self.table.len == 0 and self.kind == .paired) self.table = try host.read(a, try hc_prefill.sigmoidTable(self.rig.stream()), self.rig.stream());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const stress = fill == .stress;
        for (self.banks[0..self.nb()]) |*b| try b.fill(a, rng, self.ne * shape.n, shape.k, .bfloat16, stress);
        self.m = self.routed(shape);
        self.ids = try a.alloc(f64, self.m);
        raggedIds(self.ids, self.ne);
        const xr = shape.rows;
        self.x = try a.alloc(f64, xr * shape.k);
        if (stress) stressX(rng, self.x, shape.k) else host.uniformInto(rng, self.x, -1, 1, .bfloat16);
        if (fill == .nonfinite) plantRows(self.x, shape.k);
        self.rmap = try a.alloc(f64, self.m);
        for (self.rmap, 0..) |*v, i| v.* = @floatFromInt((i * 73 + 11) % xr);
        try self.bind();
        return true;
    }

    fn bind(self: *GqmmNax) !void {
        const rig = &self.rig;
        const a = rig.a;
        rig.reset();
        const shape = self.shape;
        const rows: c_int = @intCast(self.m);
        // The ids are input 0 of the rig (the wrapper's and the baseline's); the kernel reads them through the tiles.
        _ = try rig.input(.uint32, &.{rows}, self.ids);
        const p = self.planAt(shape);
        const mt = naxMaxTiles(self.m, self.ne, @intCast(p.bm));
        const tiles = try a.alloc(u32, 4 * mt);
        defer a.free(tiles);
        const tc = hostTiles(self.ids, self.ne, @intCast(p.bm), mt, tiles);
        const tf = try a.alloc(f64, tiles.len);
        defer a.free(tf);
        for (tf, tiles) |*o, v| o.* = @floatFromInt(v);
        const t = try rig.input(.uint32, &.{@intCast(tf.len)}, tf);
        const count_ = try rig.input(.uint32, &.{1}, &.{@floatFromInt(tc)});
        const x = try rig.input(.bfloat16, &.{ @intCast(shape.rows), 1, @intCast(shape.k) }, self.x);
        var b: [2][3]mlx.mlx_array = undefined;
        for (&b, 0..) |*v, i| v.* = try self.banks[if (i < self.nb()) i else 0].bind(rig, .bfloat16, self.ne, shape.n, shape.k);
        // An FP4 bank has no biases; its slot binds 16 zeros, as the wrapper's does.
        const zeros: [16]f64 = @splat(0);
        for (&b) |*v| if (v[2].ctx == null) {
            v[2] = try rig.input(.bfloat16, &.{16}, &zeros);
        };
        const width: f64 = @floatFromInt(if (self.kind == .paired) 2 * shape.n else shape.n);
        const params = try rig.input(.int32, &.{3}, &.{ width, @floatFromInt(shape.k), 0 });
        if (self.kind == .mm) {
            self.args[0..7].* = .{ x, b[0][0], b[0][1], b[0][2], t, count_, params };
            self.n_args = 7;
            return;
        }
        const sig = if (self.kind == .paired) try rig.input(.bfloat16, &.{65536}, self.table) else b[0][2];
        self.args = .{ x, b[0][0], b[0][1], b[0][2], b[1][0], b[1][1], b[1][2], t, count_, params, sig, try rig.input(.uint32, &.{rows}, self.rmap) };
        self.n_args = 12;
    }

    pub fn run(self: *GqmmNax) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, self.args[0..self.n_args]);
    }

    pub fn count(self: *const GqmmNax) usize {
        return self.m * self.shape.n;
    }

    fn xrow(self: *const GqmmNax, r: usize) []const f64 {
        const k = self.shape.k;
        const at: usize = if (self.kind == .mm) r else @intFromFloat(self.rmap[r]);
        return self.x[at * k ..][0..k];
    }

    /// Float64 dots over the weights the tile loader writes (dequantized, rounded to bf16), per bank.
    fn dots(self: *GqmmNax) !void {
        if (self.dv.len > 0) return;
        const a = self.rig.a;
        const n = self.shape.n;
        const k = self.shape.k;
        const cnt = self.count();
        self.dv = try a.alloc(f64, self.nb() * cnt);
        self.dm = try a.alloc(f64, self.nb() * cnt);
        const w = try a.alloc(f64, k);
        defer a.free(w);
        for (0..self.nb()) |bi| for (0..self.m) |r| for (0..n) |j| {
            const e: usize = @intFromFloat(self.ids[r]);
            for (w, 0..) |*o, i| o.* = host.rt(.bfloat16, self.banks[bi].w(e * n + j, i));
            const d = host.dot(self.xrow(r), w);
            self.dv[bi * cnt + r * n + j] = d[0];
            self.dm[bi * cnt + r * n + j] = d[1];
        };
    }

    /// The bf16 product, or for the pair MLX's SwiGLU on the two rounded products (clamped first under a limit).
    fn epilogue(self: *const GqmmNax, dv: []const f64, want: []f64) void {
        if (self.kind != .paired) return @memcpy(want, dv[0..want.len]);
        const cnt = want.len;
        const lim: f64 = @floatFromInt(self.limit);
        for (want, 0..) |*y, i| {
            var g = host.rt(.bfloat16, dv[i]);
            var u = host.rt(.bfloat16, dv[cnt + i]);
            if (self.limit > 0) {
                g = clampNaN(g, -std.math.inf(f64), lim);
                u = clampNaN(u, -lim, lim);
            }
            y.* = host.rt(.bfloat16, g * self.table[spec.toBf16(@floatCast(g))]) * u;
        }
    }

    pub fn reference(self: *GqmmNax, want: []f64, mag: []f64) !void {
        try self.dots();
        self.epilogue(self.dv, want);
        if (self.kind == .paired) @memset(mag, 0) else @memcpy(mag, self.dm);
    }

    pub fn simulate(self: *GqmmNax, acc: spec.Accum, order: spec.Order, out: []f64) !void {
        try self.dots();
        if (acc == .f32) {
            for (self.dm) |m| if (m >= 0x1p24) return error.StressPastExact;
            return self.epilogue(self.dv, out);
        }
        const a = self.rig.a;
        const n = self.shape.n;
        const cnt = self.count();
        const dv = try a.dupe(f64, self.dv);
        defer a.free(dv);
        const t = try a.alloc(f64, self.shape.k);
        defer a.free(t);
        for (0..self.nb()) |bi| for (0..self.m) |r| for (0..@min(32, n)) |j| {
            const e: usize = @intFromFloat(self.ids[r]);
            for (t, self.xrow(r), 0..) |*o, x, i| o.* = x * self.banks[bi].w(e * n + j, i);
            dv[bi * cnt + r * n + j] = spec.sumIn(acc, order, t);
        };
        self.epilogue(dv, out);
    }

    fn gatherQmm(self: *GqmmNax, g: *ops.Graph, x: mlx.mlx_array, bi: usize) !mlx.mlx_array {
        const bank = self.banks[bi];
        const mode: [*:0]const u8 = switch (bank.fmt) {
            .affine => "affine",
            .nvfp4 => "nvfp4",
            .mxfp4 => "mxfp4",
        };
        const bias: mlx.mlx_array = if (self.mx()) .{ .ctx = null } else self.args[3 + 3 * bi];
        var y = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_gather_qmm(&y, x, self.args[1 + 3 * bi], self.args[2 + 3 * bi], bias, .{ .ctx = null }, self.rig.ins.items[0].view, true, mlx.mlx_optional_int.some(@intCast(bank.aff.gs)), mlx.mlx_optional_int.some(@intCast(bank.aff.bits)), mode, true, self.rig.stream()));
        return g.adopt(y);
    }

    /// MLX's sorted gather_qmm (over the mapped rows), and for the pair MLX's SwiGLU chain on two of them.
    pub fn replaced(self: *GqmmNax) ![]const u8 {
        const s = self.rig.stream();
        var g = ops.Graph.init(self.rig.a, s);
        defer g.deinit();
        const x = if (self.kind == .mm) self.args[0] else try g.take(self.args[0], self.args[11], 0);
        const gate = try self.gatherQmm(&g, x, 0);
        if (self.kind != .paired) return self.rig.bytesOf(&.{gate});
        const up = try self.gatherQmm(&g, x, 1);
        if (self.limit > 0) return self.rig.bytesOf(&.{try g.adopt(try activations.clampedSwiGLU(s, gate, up, @floatFromInt(self.limit)))});
        return self.rig.bytesOf(&.{try g.mul(try g.mul(gate, try g.sigmoid(gate)), up)});
    }

    /// The shipped wrapper on the same inputs (its own scan, plan and canaries).
    fn wrapper(self: *GqmmNax) ![]const u8 {
        const s = self.rig.stream();
        const ids = self.rig.ins.items[0].view;
        const x = self.args;
        const none = mlx.mlx_array{ .ctx = null };
        const bias = if (self.mx()) none else x[3];
        const bits = self.banks[0].aff.bits;
        const group: u32 = @intCast(self.banks[0].aff.gs);
        const y = switch (self.kind) {
            .mm => try gqn.sortedGather(x[0], x[1], x[2], bias, ids, bits, group, self.mx(), true, s),
            .mapped => try gqn.sortedGatherMapped(x[0], x[11], x[1], x[2], bias, ids, bits, group, self.mx(), true, s),
            .paired => try gqn.sortedGateUp(x[0], x[11], x[1], x[2], bias, x[4], x[5], if (self.mx()) none else x[6], ids, x[10], self.limit, bits, group, self.mx(), true, s),
        } orelse return error.Declined;
        defer _ = mlx.mlx_array_free(y);
        return self.rig.bytesOf(&.{y});
    }
};

const nax_rule = "integer ids in, integer tiles out: no non-finite value exists; at the extremes (every row on the first or the last expert) the tiles stay exact";

fn registerNax(set: *kc.Set) !void {
    const a = set.a;
    // One shape per plan at 32 experts: db 64 (under 36 rows an expert), db 96 with 32 column groups (K under
    // 1024, 120+ rows an expert), db 64 with 32 groups (K 1024, under 48), seg 64 (N off the 64-column grid).
    const plans: []const Shape = &.{ .{ .rows = 1024, .n = 128, .k = 256 }, .{ .rows = 4096, .n = 128, .k = 512 }, .{ .rows = 1280, .n = 128, .k = 1024 }, .{ .rows = 2048, .n = 96, .k = 256 } };
    // The scan's n is BM: each tile height its plans use.
    try set.add(NaxScan, .init(a, 32), .{ .name = "gather_qmm_nax tile scan (32 experts)", .accum = .none, .depth = routeDepth, .out = .f32, .nan = .{ .stated = nax_rule }, .served = &.{ .{ .rows = 1024, .n = 64, .k = 1 }, .{ .rows = 4096, .n = 96, .k = 1 }, .{ .rows = 4096, .n = 128, .k = 1 }, .{ .rows = 33, .n = 64, .k = 1 } } }, null);
    try set.add(NaxScan, .init(a, 512), .{ .name = "gather_qmm_nax tile scan (512 experts)", .accum = .none, .depth = routeDepth, .out = .f32, .nan = .{ .stated = nax_rule }, .served = &.{ .{ .rows = 81920, .n = 96, .k = 1 }, .{ .rows = 2000, .n = 64, .k = 1 } } }, null);
    const gather = "MLX gather_qmm (sorted rhs indices)";
    const Case = struct { name: []const u8, f: Fmt, bits: u32, gs: usize, served: []const Shape };
    const cases = [_]Case{
        .{ .name = "gather_qmm_nax bf16 (4-bit g64)", .f = .affine, .bits = 4, .gs = 64, .served = plans },
        .{ .name = "gather_qmm_nax bf16 (8-bit g32)", .f = .affine, .bits = 8, .gs = 32, .served = plans[0..2] },
        .{ .name = "gather_qmm_nax bf16 (3-bit g64)", .f = .affine, .bits = 3, .gs = 64, .served = plans[0..2] },
        .{ .name = "gather_qmm_nax bf16 (2-bit g64)", .f = .affine, .bits = 2, .gs = 64, .served = plans[0..2] },
        .{ .name = "gather_qmm_nax bf16 (MXFP4)", .f = .mxfp4, .bits = 4, .gs = 32, .served = plans[0..2] },
        .{ .name = "gather_qmm_nax bf16 (NVFP4)", .f = .nvfp4, .bits = 4, .gs = 16, .served = plans[0..2] },
    };
    for (cases) |c| try set.add(GqmmNax, .init(a, .mm, c.f, c.bits, c.gs, 32), .{ .name = c.name, .accum = .f32, .depth = gqDepth, .out = .bf16, .replaces = gather, .served = c.served }, null);
    // Mapped rows are tokens (four routed rows each): the plain plans' first two, then the paired plans: db 64,
    // seg 96 x 128 with 32 groups (K under 1024), seg 128 x 128 with the pad (K 1024).
    try set.add(GqmmNax, .init(a, .mapped, .affine, 4, 64, 32), .{ .name = "gather_qmm_nax mapped bf16 (4-bit g64)", .accum = .f32, .depth = gqDepth, .out = .bf16, .replaces = "take (the row map) -> " ++ gather, .served = &.{ .{ .rows = 256, .n = 128, .k = 256 }, .{ .rows = 1024, .n = 128, .k = 512 } } }, null);
    const paired: []const Shape = &.{ .{ .rows = 512, .n = 128, .k = 512 }, .{ .rows = 960, .n = 128, .k = 512 }, .{ .rows = 1024, .n = 128, .k = 1024 } };
    const chain = "take (the row map) -> " ++ gather ++ " for gate and up -> sigmoid -> multiply -> multiply";
    try set.add(GqmmNax, .init(a, .paired, .affine, 4, 64, 32), .{ .name = "gather_qmm_nax paired bf16 (4-bit g64)", .accum = .f32, .depth = gqDepth, .out = .bf16, .replaces = chain, .served = paired }, null);
    try set.add(GqmmNax, .init(a, .paired, .mxfp4, 4, 32, 32), .{ .name = "gather_qmm_nax paired bf16 (MXFP4)", .accum = .f32, .depth = gqDepth, .out = .bf16, .replaces = chain, .served = paired[0..1] }, null);
    var lim = GqmmNax.init(a, .paired, .affine, 4, 64, 32);
    lim.limit = 7;
    try set.add(GqmmNax, lim, .{ .name = "gather_qmm_nax paired bf16 (4-bit g64, limit 7)", .accum = .f32, .depth = gqDepth, .out = .bf16, .replaces = "take (the row map) -> " ++ gather ++ " for gate and up -> clampedSwiGLU", .served = paired[0..1], .known = limit_drops_nan }, null);
}

// ── the stream-read microbenchmark ──

const ubench = @import("moe_gather_ubench.zig");

const stream_launches = [2]metal.Launch{
    .{ .name = "stream_wide", .inputs = &.{"a"}, .outputs = &.{"out"}, .source = ubench.STREAM_WIDE_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 },
    .{ .name = "stream_wide_strided", .inputs = &.{"a"}, .outputs = &.{"out"}, .source = ubench.STREAM_WIDE_SOURCE, .tg_bytes = 0, .m1 = 1024, .m2 = 1024 },
};

/// `ubench_stream_wide` (the MOE_UBENCH read-ceiling probe, never served): each simdgroup XOR-folds the 16-byte words
/// it reads and one simd_sum (a wrapping uint sum) writes its word. Rows are simdgroups, n the 64 KiB span in
/// uint4 (4096), k the loads in flight; `strided` reads grid-stride. The bf16 input is read as raw bits.
const Stream = struct {
    rig: metal.Rig,
    launch: *const metal.Launch,
    strided: bool,
    shape: Shape = undefined,
    words: []u32 = &.{},
    view: mlx.mlx_array = .{ .ctx = null },
    msg: [256]u8 = undefined,

    fn init(a: std.mem.Allocator, strided: bool) Stream {
        return .{ .rig = metal.Rig.init(a), .launch = &stream_launches[@intFromBool(strided)], .strided = strided };
    }

    pub fn deinit(self: *Stream) void {
        self.freeView();
        self.rig.deinit();
        self.rig.a.free(self.words);
    }

    fn freeView(self: *Stream) void {
        if (self.view.ctx != null) _ = mlx.mlx_array_free(self.view);
        self.view = .{ .ctx = null };
    }

    pub fn geometry(self: *const Stream, shape: Shape) ?metal.Geometry {
        if (shape.rows < 8 or shape.n != 4096 or shape.k == 0 or shape.k > 8 or shape.n % (32 * shape.k) != 0) return null;
        var g = metal.Geometry{ .grid = .{ @intCast(32 * shape.rows), 1, 1 }, .tg = .{ 256, 1, 1 } };
        g.out(&.{@intCast(shape.rows)}, .uint32);
        g.int("SPAN4", shape.n);
        g.int("UNROLL", shape.k);
        g.int("STRIDED", @intFromBool(self.strided));
        g.int("NSG", shape.rows);
        return g;
    }

    pub fn prepare(self: *Stream, shape: Shape, fill: spec.Fill, seed: u64) !bool {
        if (self.geometry(shape) == null) return false;
        const a = self.rig.a;
        self.shape = shape;
        a.free(self.words);
        self.words = try a.alloc(u32, 4 * shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        prng.random().bytes(std.mem.sliceAsBytes(self.words));
        if (fill == .nonfinite) {
            // No float to make non-finite: the extremes instead. Every lane of simdgroup 0 folds to 0xffffffff, so
            // its sum wraps (to 0xffffffe0).
            @memset(self.words[0 .. 4 * shape.n], 0);
            for (0..32) |j| self.words[4 * j] = 0xffffffff;
        }
        const halves = try a.alloc(f64, 2 * self.words.len);
        defer a.free(halves);
        for (self.words, 0..) |w, i| {
            halves[2 * i] = @floatFromInt(w & 0xffff);
            halves[2 * i + 1] = @floatFromInt(w >> 16);
        }
        self.freeView();
        self.rig.reset();
        const raw = try self.rig.input(.uint16, &.{@intCast(halves.len)}, halves);
        var v = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_view(&v, raw, .bfloat16, self.rig.stream()));
        self.view = v;
        return true;
    }

    pub fn run(self: *Stream) !void {
        try self.rig.launch(self.launch, self.geometry(self.shape).?, &.{self.view});
    }

    pub fn count(self: *const Stream) usize {
        return self.shape.rows;
    }

    pub fn reference(_: *Stream, _: []f64, _: []f64) !void {
        return error.IntegerKernelHasNoFloat64Outputs;
    }

    /// Each simdgroup's word exactly: per thread the XOR of the uint4s it reads, summed over the simdgroup mod 2^32.
    pub fn check(self: *Stream, _: spec.Gate) !?[]const u8 {
        const a = self.rig.a;
        const nsg = self.shape.rows;
        const acc = try a.alloc(u32, 32 * nsg);
        defer a.free(acc);
        @memset(acc, 0);
        for (0..nsg * self.shape.n) |j| {
            const q = self.words[4 * j ..][0..4];
            const t = if (self.strided) j % (32 * nsg) else (j / self.shape.n) * 32 + j % 32;
            acc[t] ^= q[0] ^ q[1] ^ q[2] ^ q[3];
        }
        const out = self.rig.outputBytes();
        for (0..nsg) |sg| {
            var want: u32 = 0;
            for (acc[32 * sg ..][0..32]) |x| want +%= x;
            const got = std.mem.readInt(u32, out[4 * sg ..][0..4], .little);
            if (got != want) return std.fmt.bufPrint(&self.msg, "simdgroup {d} wrote {x} where the exact fold gives {x}", .{ sg, got, want }) catch &self.msg;
        }
        return null;
    }
};

fn registerStream(set: *kc.Set) !void {
    const rule = "bits in, bits out: XOR folds and a wrapping uint sum, no float; at the extremes (all-ones words, a sum past 2^32) each word stays exact";
    const one = Shape{ .rows = 8, .n = 4096, .k = 1 };
    try set.add(Stream, .init(set.a, false), .{ .name = "ubench_stream_wide (per-simdgroup spans)", .accum = .none, .depth = routeDepth, .out = .f32, .nan = .{ .stated = rule }, .served = &.{ one, .{ .rows = 64, .n = 4096, .k = 2 }, .{ .rows = 64, .n = 4096, .k = 4 }, .{ .rows = 64, .n = 4096, .k = 8 } } }, null);
    try set.add(Stream, .init(set.a, true), .{ .name = "ubench_stream_wide (grid-stride)", .accum = .none, .depth = routeDepth, .out = .f32, .nan = .{ .stated = rule }, .served = &.{ one, .{ .rows = 64, .n = 4096, .k = 2 }, .{ .rows = 64, .n = 4096, .k = 4 } } }, null);
}

pub fn register(set: *kc.Set) !void {
    try registerGathers(set);
    try registerFp4(set);
    try registerRouter(set);
    try registerVerify(set);
    try registerNax(set);
    try registerStream(set);
}

fn registerGathers(set: *kc.Set) !void {
    const a = set.a;
    // Gate/up projections (one x, top-k 8), the down projection (x per expert), Nemotron's fc1 with ReLU^2.
    const up_served: []const Shape = &.{ .{ .rows = 8, .n = 256, .k = 2048 }, .{ .rows = 6, .n = 512, .k = 1024 } };
    const down_served: []const Shape = &.{ .{ .rows = 8, .n = 1024, .k = 512 }, .{ .rows = 8, .n = 512, .k = 768 } };
    inline for (.{ .{ 4, 64 }, .{ 3, 64 }, .{ 2, 64 }, .{ 8, 32 } }) |bg| {
        try add(set, .init(a, .qmv, .affine, bg[0], bg[1]), std.fmt.comptimePrint("gather_qmv bf16 ({d}-bit g{d})", .{ bg[0], bg[1] }), up_served, null, null);
    }
    {
        var g = Gather.init(a, .qmv, .affine, 4, 64);
        g.dt = .float16;
        try add(set, g, "gather_qmv f16 (4-bit g64)", up_served, null, null);
        g = Gather.init(a, .qmv, .affine, 4, 64);
        g.launch = &qmv_launches[0][1];
        g.px = true;
        try add(set, g, "gather_qmv bf16 (4-bit g64, x per expert)", down_served, null, null);
        // Nemotron 3's fc1: Nano (top-6 over its 2688 hidden) and Super (top-22 over the 1024 latent); a multi-token
        // step reads x per (token, expert) pair.
        g = Gather.init(a, .qmv, .affine, 4, 64);
        g.act2 = true;
        g.ne = 24;
        try add(set, g, "gather_qmv bf16 (4-bit g64, ReLU^2)", &.{ .{ .rows = 6, .n = 512, .k = 2688 }, .{ .rows = 22, .n = 256, .k = 1024 } }, null, act2_drops_nan);
        g = Gather.init(a, .qmv, .affine, 4, 64);
        g.launch = &qmv_launches[0][1];
        g.px = true;
        g.act2 = true;
        g.ne = 16;
        try add(set, g, "gather_qmv bf16 (4-bit g64, ReLU^2, x per pair)", &.{.{ .rows = 12, .n = 256, .k = 2688 }}, null, act2_drops_nan);
        g = Gather.init(a, .qmv, .nvfp4, 4, 16);
        try add(set, g, "gather_qmv bf16 (NVFP4)", up_served, null, e4m3_nan_code);
        g = Gather.init(a, .qmv, .nvfp4, 4, 16);
        g.launch = &qmv_launches[1][1];
        g.px = true;
        try add(set, g, "gather_qmv bf16 (NVFP4, x per expert)", down_served, null, null);
    }
    const chain = "gather_qmv gate and up -> sigmoid -> multiply -> multiply";
    const clamp_chain = "gather_qmv gate and up -> minimum, clip -> sigmoid -> multiply -> multiply";
    inline for (.{ .{ 4, 64 }, .{ 3, 64 }, .{ 8, 64 } }) |bg| {
        try add(set, .init(a, .gateup, .affine, bg[0], bg[1]), std.fmt.comptimePrint("gather_qmv_gateup bf16 ({d}-bit g{d})", .{ bg[0], bg[1] }), up_served, chain, null);
    }
    {
        var g = Gather.init(a, .gateup, .affine, 4, 64);
        g.dt = .float16;
        try add(set, g, "gather_qmv_gateup f16 (4-bit g64)", up_served, chain, null);
        g = Gather.init(a, .gateup, .affine, 4, 64);
        g.limit = LIMIT;
        try add(set, g, "gather_qmv_gateup bf16 (4-bit g64, limit 10)", up_served, clamp_chain, limit_drops_nan);
        try add(set, .init(a, .gateup, .nvfp4, 4, 16), "gather_qmv_gateup bf16 (NVFP4)", up_served, chain, e4m3_nan_code);
    }
    // The batched decode tick: 2..8 rows of top-k 8.
    const rows_served: []const Shape = &.{ .{ .rows = 4, .n = 256, .k = 2048 }, .{ .rows = 8, .n = 128, .k = 1024 } };
    const per_row = "gather_qmv gate and up per row -> sigmoid -> multiply -> multiply";
    try add(set, .init(a, .gateup_rows, .affine, 4, 64), "gather_qmv_gateup_rows bf16 (4-bit g64)", rows_served, per_row, null);
    try add(set, .init(a, .gateup_rows, .affine, 3, 64), "gather_qmv_gateup_rows bf16 (3-bit g64)", rows_served, per_row, null);
    try add(set, .init(a, .gateup_rows, .nvfp4, 4, 16), "gather_qmv_gateup_rows bf16 (NVFP4)", rows_served, per_row, null);
    {
        var g = Gather.init(a, .gateup_rows, .nvfp4, 4, 16);
        g.gscale = true;
        try add(set, g, "gather_qmv_gateup_rows bf16 (NVFP4, ModelOpt gate scale)", rows_served, "gather_qmv gate and up per row -> multiply by the gate scale -> sigmoid -> multiply -> multiply", null);
    }
    // Down + score weighting + top-k sum, at 8 lanes a row (every GPU but M5) and 16 (M5).
    inline for (.{ 8, 16 }) |lpr| {
        var g = Gather.init(a, .down, .affine, 4, 64);
        g.lpr = lpr;
        try add(set, g, std.fmt.comptimePrint("down_reduce bf16 (4-bit g64, {d} lanes)", .{lpr}), down_served, null, null);
        g = Gather.init(a, .down, .nvfp4, 4, 16);
        g.lpr = lpr;
        try add(set, g, std.fmt.comptimePrint("down_reduce bf16 (NVFP4, {d} lanes)", .{lpr}), down_served, null, e4m3_nan_code);
        g = Gather.init(a, .down_rows, .affine, 4, 64);
        g.lpr = lpr;
        try add(set, g, std.fmt.comptimePrint("down_reduce_rows bf16 (4-bit g64, {d} lanes)", .{lpr}), &.{ .{ .rows = 4, .n = 1024, .k = 512 }, .{ .rows = 8, .n = 256, .k = 768 } }, "down_reduce per row", null);
        g = Gather.init(a, .down_rows, .nvfp4, 4, 16);
        g.lpr = lpr;
        try add(set, g, std.fmt.comptimePrint("down_reduce_rows bf16 (NVFP4, {d} lanes)", .{lpr}), &.{.{ .rows = 4, .n = 1024, .k = 512 }}, "down_reduce per row", null);
    }
    {
        var g = Gather.init(a, .down, .affine, 2, 64);
        g.lpr = 8;
        try add(set, g, "down_reduce bf16 (2-bit g64, 8 lanes)", down_served, null, null);
        g = Gather.init(a, .down, .affine, 8, 64);
        g.dt = .float16;
        g.lpr = 8;
        try add(set, g, "down_reduce f16 (8-bit g64, 8 lanes)", down_served, null, null);
    }
}

fn registerFp4(set: *kc.Set) !void {
    const a = set.a;
    // fp4 banks at one and two tokens: NVFP4 (plain, ModelOpt's per-expert scales, GLM's clamp) and MXFP4 (MiMo).
    const gu_served: []const Shape = &.{ .{ .rows = 1, .n = 256, .k = 2048 }, .{ .rows = 2, .n = 512, .k = 1024 } };
    const dn_served: []const Shape = &.{ .{ .rows = 1, .n = 1024, .k = 512 }, .{ .rows = 2, .n = 512, .k = 768 } };
    try addFp4(set, .init(a, .gateup, .nvfp4), "moe_fp4 gateup bf16 (NVFP4)", gu_served, null);
    {
        var f = Fp4.init(a, .gateup, .nvfp4);
        f.gscale = true;
        f.uscale = true;
        try addFp4(set, f, "moe_fp4 gateup bf16 (NVFP4, gate and up scales)", gu_served, null);
        f = Fp4.init(a, .gateup, .nvfp4);
        f.gscale = true;
        f.clamp = LIMIT;
        try addFp4(set, f, "moe_fp4 gateup bf16 (NVFP4, gate scale, clamp 10)", gu_served, limit_drops_nan);
        f = Fp4.init(a, .gateup, .mxfp4);
        try addFp4(set, f, "moe_fp4 gateup bf16 (MXFP4)", gu_served, null);
        f = Fp4.init(a, .gateup, .mxfp4);
        f.dt = .float16;
        try addFp4(set, f, "moe_fp4 gateup f16 (MXFP4)", gu_served, null);
    }
    try addFp4(set, .init(a, .downred, .nvfp4), "moe_fp4 downred bf16 (NVFP4)", dn_served, null);
    try addFp4(set, .init(a, .downred, .mxfp4), "moe_fp4 downred bf16 (MXFP4)", dn_served, null);
    {
        var f = Fp4.init(a, .downred, .nvfp4);
        f.shared = true;
        try addFp4(set, f, "moe_fp4 downred bf16 (NVFP4, gated shared expert)", &.{.{ .rows = 1, .n = 1024, .k = 512 }}, null);
        f = Fp4.init(a, .downred, .mxfp4);
        f.dt = .float16;
        try addFp4(set, f, "moe_fp4 downred f16 (MXFP4)", dn_served, null);
    }
}

// ── tests ──

test "kernel contract moe: each declared launch is the shipped wrapper's, byte for byte" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    gq.gqmv_gateup_override = true;
    defer gq.gqmv_gateup_override = null;
    gqr.moe_rows_fused_override = true;
    defer gqr.moe_rows_fused_override = null;
    dr.downred_override = true;
    defer dr.downred_override = null;
    var set = kc.Set.init(a);
    defer set.deinit();
    try registerGathers(&set);
    for (set.kernels.items) |k| {
        const t: *Gather = @ptrCast(@alignCast(k.ctx));
        // The wrappers pick the lanes per row by device.
        if ((t.kind == .down or t.kind == .down_rows) and t.lpr != @as(usize, @intCast(dr.downredLpr()))) continue;
        try std.testing.expect(try t.prepare(k.spec.served[0], .random, 7));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        const theirs = try t.wrapper();
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}

test "kernel contract moe: the fp4 launches, chained, are moe_fp4.decode's bytes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try registerFp4(&set);
    for (set.kernels.items) |k| {
        const t: *Fp4 = @ptrCast(@alignCast(k.ctx));
        t.both = true;
        try std.testing.expect(try t.prepare(k.spec.served[0], .random, 7));
        const got = try t.chained();
        defer a.free(got[0]);
        if (!std.mem.eql(u8, got[0], got[1])) std.debug.print("{s}: the declared launches differ from moe_fp4.decode\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, got[1], got[0]);
    }
}

test "kernel contract moe: each router launch is moeRouterTopK's bytes" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    var set = kc.Set.init(a);
    defer set.deinit();
    try registerRouter(&set);
    for (set.kernels.items) |k| {
        const t: *Router = @ptrCast(@alignCast(k.ctx));
        try std.testing.expect(try t.prepare(k.spec.served[0], .random, 7));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        const theirs = try t.wrapper();
        if (!std.mem.eql(u8, ours, theirs)) std.debug.print("{s}: the declared launch differs from the wrapper's\n", .{k.spec.name});
        try std.testing.expectEqualSlices(u8, theirs, ours);
    }
}

test "kernel contract moe: each verify launch is its wrapper's bytes, the prefill route's three chained" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const outer = vp.verify_group_rows;
    vp.verify_group_rows = 2; // a two-stream group: the dual-die M5 takes the shared kernels only then
    defer vp.verify_group_rows = outer;
    const hw = vp.verifySharedHardware();
    if (hw) {
        var t = Reuse.init(a);
        defer t.deinit();
        try std.testing.expect(try t.prepare(.{ .rows = 20, .n = 2560, .k = 640 }, .random, 7));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try t.wrapper(), ours);
    }
    for ([_]Shape{ .{ .rows = 4, .n = 2560, .k = 10 }, .{ .rows = 64, .n = 2560, .k = 10 }, .{ .rows = 64, .n = 4096, .k = 8 } }) |shape| {
        if (shape.rows <= 32 and !hw) continue;
        var t = Reduce.init(a);
        defer t.deinit();
        try std.testing.expect(try t.prepare(shape, .random, 7));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try t.wrapper(), ours);
    }
    if (hw) {
        var t = Route.init(a, .pack, 512);
        defer t.deinit();
        try std.testing.expect(try t.prepare(.{ .rows = 8, .n = 1, .k = 10 }, .random, 7));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        const got = (try vk.verifyRoutePack(t.rig.stream(), t.args[0], 10)) orelse return error.Declined;
        defer for ([_]mlx.mlx_array{ got.inverse, got.sorted, got.lhs }) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try std.testing.expectEqualSlices(u8, try t.rig.bytesOf(&.{ got.inverse, got.sorted, got.lhs }), ours);
    }
    {
        // hist -> scan -> scatter through the declared launches, against moePrefillRoutePack.
        const shape = Shape{ .rows = 1000, .n = 1, .k = 8 };
        var t = Route.init(a, .hist, 256);
        defer t.deinit();
        try std.testing.expect(try t.prepare(shape, .random, 7));
        try t.run();
        var counts = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(counts);
        try mlx.check(mlx.mlx_array_set(&counts, t.rig.output(0)));
        var scan = Route.init(a, .scan, 256);
        try t.rig.launch(&route_launches[@backingInt(RouteKind.scan)], scan.geometry(shape).?, &.{counts});
        var offsets = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(offsets);
        try mlx.check(mlx.mlx_array_set(&offsets, t.rig.output(0)));
        const scatter = Route.init(a, .scatter, 256);
        scan.deinit();
        try t.rig.launch(&route_launches[@backingInt(RouteKind.scatter)], scatter.geometry(shape).?, &.{ t.args[0], offsets });
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        const got = (try vk.moePrefillRoutePack(t.rig.stream(), t.args[0], 8, 256)) orelse return error.Declined;
        defer for ([_]mlx.mlx_array{ got.inverse, got.sorted, got.lhs }) |x| {
            _ = mlx.mlx_array_free(x);
        };
        try std.testing.expectEqualSlices(u8, try t.rig.bytesOf(&.{ got.inverse, got.sorted, got.lhs }), ours);
    }
}

test "kernel contract moe: the NAX scan chained into each NAX gather launch is the wrapper's bytes" {
    if (mlx.noGpuBackend() or !device_mod.naxAvailable()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const Case = struct { kind: NaxKind, f: Fmt = .affine, bits: u32 = 4, gs: usize = 64, shape: Shape };
    const cases = [_]Case{
        .{ .kind = .mm, .shape = .{ .rows = 1024, .n = 128, .k = 256 } },
        .{ .kind = .mm, .shape = .{ .rows = 2048, .n = 96, .k = 256 } },
        .{ .kind = .mm, .f = .nvfp4, .gs = 16, .shape = .{ .rows = 4096, .n = 128, .k = 512 } },
        .{ .kind = .mapped, .shape = .{ .rows = 1024, .n = 128, .k = 512 } },
        .{ .kind = .paired, .shape = .{ .rows = 960, .n = 128, .k = 512 } },
        .{ .kind = .paired, .f = .mxfp4, .gs = 32, .shape = .{ .rows = 1024, .n = 128, .k = 1024 } },
    };
    for (cases) |c| {
        var t = GqmmNax.init(a, c.kind, c.f, c.bits, c.gs, 32);
        defer t.deinit();
        try std.testing.expect(try t.prepare(c.shape, .random, 7));
        const scan = NaxScan.init(a, 32);
        const bm: usize = @intCast(t.planAt(c.shape).bm);
        const shape = Shape{ .rows = t.m, .n = bm, .k = 1 };
        const mt = naxMaxTiles(shape.rows, 32, bm);
        const params = try t.rig.input(.int32, &.{3}, &.{ @floatFromInt(shape.rows), 32, @floatFromInt(mt) });
        try t.rig.launch(&scan_launch, scan.geometry(shape).?, &.{ t.rig.ins.items[0].view, params });
        var tiles = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (tiles) |x| {
            _ = mlx.mlx_array_free(x);
        };
        for (&tiles, 0..) |*x, i| try mlx.check(mlx.mlx_array_set(x, t.rig.output(i)));
        const at: usize = if (c.kind == .mm) 4 else 7;
        t.args[at] = tiles[0];
        t.args[at + 1] = tiles[1];
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        try std.testing.expectEqualSlices(u8, try t.wrapper(), ours);
    }
}

test "kernel contract moe: the stream-read launches are the microbenchmark's config" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = std.testing.allocator;
    const ins = [_][*:0]const u8{"a"};
    const outs = [_][*:0]const u8{"out"};
    const vin = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(vin);
    const vout = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(vout);
    const kernel = mlx.mlx_fast_metal_kernel_new("stream_wide", vin, vout, ubench.STREAM_WIDE_SOURCE, "", true, false);
    defer _ = mlx.mlx_fast_metal_kernel_free(kernel);
    for ([_]bool{ false, true }) |strided| {
        var t = Stream.init(a, strided);
        defer t.deinit();
        const shape = Shape{ .rows = 64, .n = 4096, .k = 4 };
        try std.testing.expect(try t.prepare(shape, .random, 7));
        try t.run();
        const ours = try a.dupe(u8, t.rig.outputBytes());
        defer a.free(ours);
        const cfg = try ubench.streamWideConfig(64, 4, @intFromBool(strided));
        defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
        const in_vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{t.view}, 1);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(out_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&out_vec, kernel, in_vec, cfg, t.rig.stream()));
        var o = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(o);
        try mlx.check(mlx.mlx_vector_array_get(&o, out_vec, 0));
        try std.testing.expectEqualSlices(u8, try t.rig.bytesOf(&.{o}), ours);
    }
}
