//! GLM-5.3-Flash's EXL3 K=3 decode forms: the decode chain of the seam (`exl3_kernels.zig` gateUp / down: in_rin, the
//! gate and up GEMVs, gu_epi, din_rin, the down GEMV, dpost; DeepSeek-V4.1's banked mul1h texts at GLM's shape) in other
//! launch geometries, each computing the seam's words (the bench's device parity checks every output word). A form is a
//! construction-time route: `install` builds the chosen form's launches for every wave width once; `gateUp` / `down`
//! keep the seam's signatures and run them with no per-call choice.
//!
//! Forms (texts in kernels/exl3_forms/):
//!   gate / up (`GuForm`): `seam` (two GEMV launches + gu_epi), `guone` (DeepSeek's gu_one: both GEMVs in one launch).
//!   down (`DnForm`): `seam`, `pair` (DeepSeek's down_pair).
//! Every form keeps the seam's arithmetic and order per output word (fma chains per lane over the k-tiles, the column
//! sums, the 128-point butterflies, the scales, the clamped SwiGLU, one rounding on dpost's store); only where values
//! live and which thread computes them change. A form that fails the device parity is not installed.
//! Wave widths: P = 1 .. `max_pairs` (64 = 8 tokens x top-8, the NVFP4 texts' widest wave; the decode lane forms at most
//! 4 rows x 8 = 32 pairs, and the scheduler's route at most `policy.max_route_ids` = 48).

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const mtl = @import("../sdk_ext/metal_launch.zig");
const kset = @import("../sdk_ext/kernel_set.zig");
const xk = @import("exl3_kernels.zig");

pub const G = xk.G;
pub const T = xk.T;
const BankArrays = xk.BankArrays;
const n_banks = xk.n_banks;
const hidden = xk.hidden;
const inter = xk.inter;

pub const max_pairs: u32 = 64;

comptime {
    if (max_pairs < @import("../sdk_ext.zig").expert.policy.max_route_ids) @compileError("glm5_next exl3 forms: the widest wave is narrower than the scheduler's route");
}

pub const GuForm = enum { seam, guone };
pub const DnForm = enum { seam, pair };
pub const Forms = struct {
    gu: GuForm = .seam,
    dn: DnForm = .seam,
};

// ── The texts ──

pub const Kernel = enum { g5k3_guone, g5k3_pair };
const n_kernels = std.meta.fieldNames(Kernel).len;
const dir = "kernels/exl3_forms/";
pub const texts: [n_kernels][:0]const u8 = blk: {
    var out: [n_kernels][:0]const u8 = undefined;
    for (std.meta.fieldNames(Kernel), 0..) |n, i| out[i] = @embedFile(dir ++ n ++ ".metal");
    break :blk out;
};
/// sha256 of each text.
pub const text_sha256 = [n_kernels]*const [64]u8{
    "85dcbd286beb00d7ec1f80b298967859090c0f8aedf29824380a5c5027df3374",
    "0732d4840889255ad9499b30e1431fc2ff78ec19d72ebc89f5a1d9df31440110",
};

pub fn checkPins() error{Glm5NextExl3FormsTextNotPinned}!void {
    for (texts, text_sha256) |t, want| if (!std.mem.eql(u8, &kset.shaHex(t), want)) return error.Glm5NextExl3FormsTextNotPinned;
}

const Sig = struct { ins: []const [*:0]const u8, outs: []const [*:0]const u8 };

fn sigOf(k: Kernel) Sig {
    return switch (k) {
        .g5k3_guone => .{ .ins = &.{ "xh_g", "xh_u", "ids", "code_g0", "code_g1", "code_g2", "code_u0", "code_u1", "code_u2" }, .outs = &.{ "out_g", "out_u" } },
        .g5k3_pair => .{ .ins = &.{ "xh", "ids", "code0", "code1", "code2" }, .outs = &.{"out"} },
    };
}

// ── Launches (pure: the tests read them) ──

pub const Launch = struct {
    grid: [3]u32,
    tg: [3]u32,
    ints: [4]struct { name: [*:0]const u8, v: u32 } = undefined,
    n_int: u8 = 0,
    outs: [2]struct { shape: [2]c_int, dtype: mlx.mlx_dtype } = undefined,
    n_out: u8 = 0,

    fn int(l: Launch, name: [*:0]const u8, v: u32) Launch {
        var o = l;
        o.ints[o.n_int] = .{ .name = name, .v = v };
        o.n_int += 1;
        return o;
    }
    fn out(l: Launch, rows: u32, cols: u32, dtype: mlx.mlx_dtype) Launch {
        var o = l;
        o.outs[o.n_out] = .{ .shape = .{ @intCast(rows), @intCast(cols) }, .dtype = dtype };
        o.n_out += 1;
        return o;
    }
};

/// DeepSeek's gu_one at GLM's shape: grid (inter / 16 x 32, 2 P): the gate's P rows, then the up's.
pub fn guoneLaunch(p: u32) Launch {
    const l: Launch = .{ .grid = .{ inter / 16 * 32, 2 * p, 1 }, .tg = .{ 32, 1, 1 } };
    return l.int("IT", hidden / 16).int("OT", inter / 16).out(p, inter, .float32).out(p, inter, .float32);
}

/// The one-simdgroup GEMV text (pair): grid (out / 16 x 32, P).
pub fn gemvLaunch(p: u32, in: u32, o: u32) Launch {
    const l: Launch = .{ .grid = .{ o / 16 * 32, p, 1 }, .tg = .{ 32, 1, 1 } };
    return l.int("IT", in / 16).int("OT", o / 16).out(p, o, .float32);
}

fn cfgOf(l: Launch) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (l.outs[0..l.n_out]) |*o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &o.shape, 2, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @intCast(l.grid[0]), @intCast(l.grid[1]), @intCast(l.grid[2])));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, @intCast(l.tg[0]), @intCast(l.tg[1]), @intCast(l.tg[2])));
    for (l.ints[0..l.n_int]) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t.name, @intCast(t.v)));
    return c;
}

/// A seam launch (exl3_kernels.zig's builders) as a config: its outputs, grid, threadgroup, ints and T.
fn seamCfg(l: xk.Launch) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (l.outs[0..l.n_out]) |*o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &o.shape, o.ndim, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, l.grid[0], l.grid[1], l.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, l.tg[0], l.tg[1], l.tg[2]));
    for (l.ints[0..l.n_int]) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t.name, t.v));
    if (l.t_dtype) |d| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", d));
    return c;
}

fn apply(g: *G, k: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, ins: []const T, outs: []T) !void {
    const in = mlx.mlx_vector_array_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_array_free(in);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mtl.mlx_fast_metal_kernel_apply(&ov, k, in, cfg, g.s));
    for (outs, 0..) |*o, i| {
        var x = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(&x, ov, i)) catch |e| {
            _ = mlx.mlx_array_free(x);
            return e;
        };
        o.* = try g.adopt(x);
    }
}

// ── Install ──

const Cfg = mlx.mlx_fast_metal_kernel_config;
const none: Cfg = .{ .ctx = null };
/// One wave width's prepared launches.
const WaveCfg = struct {
    in_rin: Cfg = none,
    g1: Cfg = none,
    epi: Cfg = none,
    din: Cfg = none,
    d1: Cfg = none,
    post: Cfg = none,

    fn free(w: WaveCfg) void {
        for ([_]Cfg{ w.in_rin, w.g1, w.epi, w.din, w.d1, w.post }) |c| if (c.ctx != null) {
            _ = mlx.mlx_fast_metal_kernel_config_free(c);
        };
    }
};

pub const Installed = struct {
    forms: Forms,
    out_dtype: mlx.mlx_dtype,
    seam: *const xk.Kernels,
    k: [n_kernels]mlx.mlx_fast_metal_kernel = @splat(.{ .ctx = null }),
    w: [max_pairs]WaveCfg = @splat(.{}),
    tok: [max_pairs]T = @splat(.{ .ctx = null }),

    pub fn deinit(self: *Installed) void {
        for (self.k) |k| if (k.ctx != null) mtl.mlx_fast_metal_kernel_free(k);
        for (self.w) |w| w.free();
        for (self.tok) |t| if (t.ctx != null) {
            _ = mlx.mlx_array_free(t);
        };
    }

    fn kern(self: *const Installed, k: Kernel) mlx.mlx_fast_metal_kernel {
        return self.k[@intFromEnum(k)];
    }
    fn seamK(self: *const Installed, k: xk.Kernel) mlx.mlx_fast_metal_kernel {
        return self.seam.k[@intFromEnum(k)];
    }
};

/// Construction, once: the pins, the texts built, every wave width's launches of the chosen forms
/// prepared (`out_dtype`: down's output, the seam's bf16; the bench also installs f32 for its reference check). The
/// seam's kernels (`seam`, from `exl3_kernels.install`) run in_rin, gu_epi, din_rin, dpost and the seam GEMV.
pub fn install(a: std.mem.Allocator, seam: *const xk.Kernels, forms: Forms, out_dtype: mlx.mlx_dtype, diag: *sdk.Diag) !*Installed {
    checkPins() catch |e| {
        diag.set("glm5_next exl3 forms: a text is not the pinned one", .{});
        return e;
    };
    const self = try a.create(Installed);
    self.* = .{ .forms = forms, .out_dtype = out_dtype, .seam = seam };
    errdefer {
        self.deinit();
        a.destroy(self);
    }
    inline for (comptime std.enums.values(Kernel), 0..) |k, i| {
        const sig = sigOf(k);
        const iv = mlx.mlx_vector_string_new_data(@ptrCast(sig.ins.ptr), sig.ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(@ptrCast(sig.outs.ptr), sig.outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        self.k[i] = mtl.mlx_fast_metal_kernel_new(@tagName(k), iv, ov, texts[i], "", true, false);
        if (self.k[i].ctx == null) {
            diag.set("glm5_next exl3 forms: {s} was not created", .{@tagName(k)});
            return error.Glm5NextExl3FormsKernel;
        }
    }
    for (0..max_pairs) |i| {
        const p: u32 = @intCast(i + 1);
        var w: WaveCfg = .{};
        errdefer w.free();
        w.in_rin = try seamCfg(xk.inRinLaunch(p));
        w.g1 = switch (forms.gu) {
            .seam => try seamCfg(xk.gemvLaunch(p, hidden, inter)),
            .guone => try cfgOf(guoneLaunch(p)),
        };
        w.epi = try seamCfg(xk.rotLaunch(p, inter, "INTER"));
        w.din = try seamCfg(xk.rotLaunch(p, inter, "INTER"));
        w.d1 = switch (forms.dn) {
            .seam => try seamCfg(xk.gemvLaunch(p, inter, hidden)),
            .pair => try cfgOf(gemvLaunch(p, inter, hidden)),
        };
        // dpost at hidden with its store in out_dtype (the seam's bf16; T is the text's store type)
        var dl = xk.rotLaunch(p, hidden, "HID");
        dl.t_dtype = out_dtype;
        dl.outs[0].dtype = out_dtype;
        w.post = try seamCfg(dl);
        self.w[i] = w;
        var idx: [max_pairs]i32 = undefined;
        for (idx[0..p], 0..) |*v, j| v.* = @intCast(j);
        const shape = [_]c_int{@intCast(p)};
        self.tok[i] = mlx.mlx_array_new_data(&idx, &shape, 1, .int32);
        if (self.tok[i].ctx == null) return error.MlxError;
    }
    return self;
}

fn widthOf(g: *G, x: T) !u32 {
    const p: u32 = @intCast(g.shapeOf(x).d[0]);
    if (p < 1 or p > max_pairs) return error.Glm5NextExl3WaveWidth;
    return p;
}

/// The seam's `gateUp` in the installed form: x bf16 [P, hidden], ids u32 [P] packed `bank << 24 | slot row` -> the
/// clamped SwiGLU f32 [P, inter].
pub fn gateUp(f: *const Installed, g: *G, x: T, ids: T, banks: *const [n_banks]BankArrays) !T {
    const p = try widthOf(g, x);
    const w = &f.w[p - 1];
    var xs: [2]T = undefined;
    try apply(g, f.seamK(.in_rin), w.in_rin, &.{ x, f.tok[p - 1], banks[0].gate.rin, banks[1].gate.rin, banks[2].gate.rin, banks[0].up.rin, banks[1].up.rin, banks[2].up.rin, ids }, &xs);
    var z: [2]T = undefined;
    switch (f.forms.gu) {
        .seam => {
            const k = f.seamK(.gemv);
            try apply(g, k, w.g1, &.{ xs[0], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code }, z[0..1]);
            try apply(g, k, w.g1, &.{ xs[1], ids, banks[0].up.code, banks[1].up.code, banks[2].up.code }, z[1..2]);
        },
        .guone => try apply(g, f.kern(.g5k3_guone), w.g1, &.{ xs[0], xs[1], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code, banks[0].up.code, banks[1].up.code, banks[2].up.code }, &z),
    }
    var out: [1]T = undefined;
    try apply(g, f.seamK(.gu_epi), w.epi, &.{ z[0], z[1], banks[0].gate.rout, banks[1].gate.rout, banks[2].gate.rout, banks[0].up.rout, banks[1].up.rout, banks[2].up.rout, ids }, &out);
    return out[0];
}

/// The seam's `down` in the installed form: h f32 [P, inter] (gateUp's), the same ids -> [P, hidden] in the installed
/// output dtype (the seam's bf16).
pub fn down(f: *const Installed, g: *G, h: T, ids: T, banks: *const [n_banks]BankArrays) !T {
    const p = try widthOf(g, h);
    const w = &f.w[p - 1];
    var out: [1]T = undefined;
    var hd: [1]T = undefined;
    try apply(g, f.seamK(.din_rin), w.din, &.{ h, banks[0].down.rin, banks[1].down.rin, banks[2].down.rin, ids }, &hd);
    const k = if (f.forms.dn == .seam) f.seamK(.gemv) else f.kern(.g5k3_pair);
    var zd: [1]T = undefined;
    try apply(g, k, w.d1, &.{ hd[0], ids, banks[0].down.code, banks[1].down.code, banks[2].down.code }, &zd);
    try apply(g, f.seamK(.dpost), w.post, &.{ zd[0], banks[0].down.rout, banks[1].down.rout, banks[2].down.rout, ids }, &out);
    return out[0];
}

// ── Tests (host only) ──

const testing = std.testing;

test "glm5_next exl3 forms: the texts are the pinned ones; the generated ones carry no DeepSeek shape literal" {
    try checkPins();
    for (texts) |t| {
        var it = std.mem.splitScalar(u8, t, '\n');
        while (it.next()) |line| {
            if (std.mem.startsWith(u8, std.mem.trimStart(u8, line, " "), "//")) continue;
            for ([_][]const u8{ "5120", "2304", "2211840", "320u", "144u" }) |lit| try testing.expect(std.mem.indexOf(u8, line, lit) == null);
        }
    }
}

test "glm5_next exl3 forms: the launches cover every output word at every wave width (one simdgroup per 16 outputs)" {
    for (1..max_pairs + 1) |pi| {
        const p: u32 = @intCast(pi);
        const go = guoneLaunch(p);
        try testing.expectEqual(2 * p, go.grid[1]);
        try testing.expectEqual(inter / 16 * 32, go.grid[0]);
    }
}
