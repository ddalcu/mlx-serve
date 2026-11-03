//! GLM-5.3-Flash's EXL3 K=3 expert kernels (the seam of the `exl3` expert family): DeepSeek-V4.1's mul1h texts at
//! GLM's expert shape (hidden 4096, inter 2048) as compile-time instantiations. The texts in kernels/exl3/ are
//! DeepSeek's with each shape literal a template constant (HID, INTER, IT, OT, KD, ND, SGS); the arithmetic, its order
//! and the bank reads stay DeepSeek's. The
//! DIG-X GEMMs read DeepSeek's `header_dig_mul1h_k3.metal` as it is (no shape literal in it).
//!
//! The seam (what the glm5_next decode stack and prompt path call; a kernel variant keeps these signatures):
//!   - Slot arrays, per projection and slot bank (the stream's base, ext and transient banks): `code` i16
//!     [rows, in / 16, out / 16, 48] (the trellis, 16 x 16 tiles of K = 3, 48 halfwords a tile), `rout` f16 [rows, out]
//!     (exl3 svh), `rin` f16 [rows, in] (exl3 suh). gate / up: in = hidden, out = inter; down: in = inter, out = hidden.
//!     Each array is row-contiguous, one slot per row. The record has no global-scale word (the codebook multiplier is
//!     the bank manifest's, fixed in the texts): no scale-line isolation applies.
//!   - Decode (`gateUp`, `down`): P routed pairs, 1 <= P <= `max_pairs` (one route: the MTP verify's rows 1..4 x top-8
//!     at most); `ids` u32 [P] packed `bank << 24 | slot row`; x bf16 [P, hidden] (each pair's token row, taken);
//!     `gateUp` -> the clamped SwiGLU f32 [P, inter]; `down` (h f32 [P, inter]) -> bf16 [P, hidden] (no route weight; the
//!     f32 word rounded once on store, the expert output type the combines read).
//!     Six launches: in_rin, the gate GEMV, the up GEMV, gu_epi, din_rin, the down GEMV, dpost.
//!   - Prompt (DIG-X, one wave of <= 16 experts of one bank, rows grouped by expert): `take2` -> `gemmGateUp` ->
//!     `onePass` -> `gemmDown` -> `widen1`; the wave table is `exl3/quant.zig digTableBm` at `m_tile` rows with
//!     `gu_tiles` / `dn_tiles` threadgroups per M tile.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;
const mtl = @import("../sdk_ext/metal_launch.zig");
const kset = @import("../sdk_ext/kernel_set.zig");
const quant = @import("../sdk_ext.zig").quant;
const xq = @import("../../../quant/exl3/stream/quant.zig");
const ops = @import("../ops.zig");

pub const G = ops.MlxOps;
pub const T = G.T;

// ── The geometry the texts are instantiated at ──

pub const hidden: u32 = 4096;
pub const inter: u32 = 2048;
/// Bits per weight of the bank's trellis (every layer): 16 x K halfwords a 16 x 16 tile.
pub const k_bits: u32 = 3;
pub const tile_words: u32 = 16 * k_bits;
/// The codebook the texts decode (exl3 mul1, the multiplier in the texts).
pub const codebook = xq.format.codebook;
pub const codebook_multiplier: u64 = xq.format.multiplier;
/// The swiglu clamp the texts fuse (gu_epi, onepass): GLM-5.3-Flash's `swiglu_limit` is DeepSeek's.
pub const swiglu_limit: f64 = xq.fused_act.swiglu_clamped;
/// The widest decode wave: one route (the scheduler's `max_route_ids`).
pub const max_pairs: u32 = @import("../sdk_ext.zig").expert.policy.max_route_ids;
/// The expert stream's slot banks: base, ext, transient.
pub const n_banks: u32 = std.meta.fieldNames(@import("../sdk_ext.zig").expert.BankKind).len;
/// DIG-X: the GEMMs' M tile and threadgroups per M tile (gate|up: both operands' 64-column tiles; down: its own).
pub const m_tile: u32 = 128;
pub const gu_tiles: u32 = 2 * inter / 64;
pub const dn_tiles: u32 = hidden / 64;
/// The onepass epilogue's simdgroups per threadgroup (inter / 128 blocks in whole threadgroups).
pub const onepass_sgs: u32 = 8;

comptime {
    if (hidden % 128 != 0 or inter % 128 != 0) @compileError("glm5_next exl3: the rotations take 128-wide blocks");
    if ((inter / 128) % onepass_sgs != 0 or (hidden / 128) % 8 != 0) @compileError("glm5_next exl3: the rotation launches take whole threadgroups");
    if (tile_words != 48) @compileError("glm5_next exl3: the texts read K = 3 tiles (48 halfwords)");
}

/// One projection's slot arrays (DeepSeek's `ProjArrays`: the stream binds them, the scheduler's default projection).
pub const Proj = xq.ProjArrays(T);
pub const BankArrays = quant.BankArrays(Proj);

/// The dims of a projection's slot arrays (without the leading rows): code, rout, rin.
pub const ProjDims = struct { code: [3]u32, rout: u32, rin: u32 };
pub const Which = enum { gate_up, down };
pub fn projDims(which: Which) ProjDims {
    const in: u32, const out: u32 = if (which == .down) .{ inter, hidden } else .{ hidden, inter };
    return .{ .code = .{ in / 16, out / 16, tile_words }, .rout = out, .rin = in };
}

// ── The texts ──

pub const Kernel = enum { in_rin, gemv, gu_epi, din_rin, dpost, take2v, gemm_gu, gemm_dn, onepass, widen1 };
const n_kernels = std.meta.fieldNames(Kernel).len;

const dir = "kernels/exl3/";
pub const texts: [n_kernels][:0]const u8 = .{
    @embedFile(dir ++ "glm5n_exl3_in_rin.metal"),
    @embedFile(dir ++ "glm5n_exl3_gemv.metal"),
    @embedFile(dir ++ "glm5n_exl3_gu_epi.metal"),
    @embedFile(dir ++ "glm5n_exl3_din_rin.metal"),
    @embedFile(dir ++ "glm5n_exl3_dpost.metal"),
    @embedFile(dir ++ "glm5n_exl3_take2v.metal"),
    @embedFile(dir ++ "glm5n_exl3_gemm_gu.metal"),
    @embedFile(dir ++ "glm5n_exl3_gemm_dn.metal"),
    @embedFile(dir ++ "glm5n_exl3_onepass.metal"),
    @embedFile(dir ++ "glm5n_exl3_widen1.metal"),
};
/// The DIG-X GEMMs' header (DeepSeek's, reused as it is) and the onepass header (DeepSeek's dig2 helpers + GLM's width
/// parameter).
pub const header_gemm: [:0]const u8 = @embedFile("../../../quant/exl3/stream/kernels/exl3/header_dig_mul1h_k3.metal");
pub const header_onepass: [:0]const u8 = @embedFile("../../../quant/exl3/stream/kernels/exl3/header_dig2_x.metal") ++ @embedFile(dir ++ "header_glm5n_exl3_dig2.metal");

/// sha256 of each text (checked once at install and in the tests).
pub const text_sha256 = [n_kernels]*const [64]u8{
    "6b5d6987c06750e44ff16cd4d92142ede9b1693643cae6c1ec6391cfad8162d2",
    "632af1fde45f91dd64b1c7c4ba05562d50bdcf4587fbe4e6ca5a91d64a5c6bbe",
    "f9526c59b09561032dad3b048a78e50b4d2c77cef487fd924a65ab6e1dab3388",
    "4cfa833c6427f68000eaac5b77d3964f5975875d95c1634720ae915fcbb4b735",
    "ba6f173e5e190177314cbadd85c6b0c8398253ce76392f1aba93454e873a35d5",
    "27755f90ed5379a9254172c794dd0f8eaa4df85fc2cee4bd0587554a333edfc8",
    "c72444223bf1b65d86ec05f681fb24672180f538474a191a0359ea8a2e02d291",
    "537d8c33523b8d703cca2a5aa8729476fb805ee12fc50170ff572e9bff4a122e",
    "155be055f25f0d138713a187c7b7915178e7feecfe6f645613abe69e9bb56e4f",
    "66885ec42a841c9f578d3e81db9b191d79d83c39126f6b92450fc14fbbbf4b55",
};

pub fn checkPins() error{Glm5NextExl3TextNotPinned}!void {
    for (texts, text_sha256) |t, want| if (!std.mem.eql(u8, &kset.shaHex(t), want)) return error.Glm5NextExl3TextNotPinned;
}

/// A text's MLX signature: inputs, outputs, whether MLX makes the inputs row-contiguous (the texts that read strides
/// do not), its header.
const Sig = struct { ins: []const [*:0]const u8, outs: []const [*:0]const u8, contiguous: bool, header: [:0]const u8 = "" };

fn sigOf(k: Kernel) Sig {
    return switch (k) {
        .in_rin => .{ .ins = &.{ "x", "tok", "rg0", "rg1", "rg2", "ru0", "ru1", "ru2", "ids" }, .outs = &.{ "xg", "xu" }, .contiguous = false },
        .gemv => .{ .ins = &.{ "xh", "ids", "code0", "code1", "code2" }, .outs = &.{"out"}, .contiguous = true },
        .gu_epi => .{ .ins = &.{ "zg", "zu", "rg0", "rg1", "rg2", "ru0", "ru1", "ru2", "ids" }, .outs = &.{"out"}, .contiguous = false },
        .din_rin => .{ .ins = &.{ "hid", "rn0", "rn1", "rn2", "ids" }, .outs = &.{"out"}, .contiguous = false },
        .dpost => .{ .ins = &.{ "zd", "rd0", "rd1", "rd2", "ids" }, .outs = &.{"out"}, .contiguous = false },
        .take2v => .{ .ins = &.{ "act", "ridx", "rhs", "slots", "rin_g", "rin_u" }, .outs = &.{ "out_g", "out_u" }, .contiguous = false },
        .gemm_gu => .{ .ins = &.{ "x0", "x1", "code0", "code1", "tbl" }, .outs = &.{ "z0", "z1" }, .contiguous = true, .header = header_gemm },
        .gemm_dn => .{ .ins = &.{ "x", "code0", "tbl" }, .outs = &.{"z0"}, .contiguous = true, .header = header_gemm },
        .onepass => .{ .ins = &.{ "z0", "z1", "rhs", "tbl", "rout0", "rout1", "rin2" }, .outs = &.{"hd"}, .contiguous = true, .header = header_onepass },
        .widen1 => .{ .ins = &.{ "act", "rhs", "slots", "rout" }, .outs = &.{"out"}, .contiguous = false },
    };
}

const MKernel = mlx.mlx_fast_metal_kernel;
const Config = mlx.mlx_fast_metal_kernel_config;

/// One launch's geometry: grid, threadgroup, template ints, outputs (shape, dtype), an optional template dtype T. Fixed
/// arrays: a launch is a value (no slice into a caller's frame).
pub const IntArg = struct { name: [*:0]const u8, v: c_int };
pub const OutArg = struct { shape: [3]c_int, ndim: u8, dtype: mlx.mlx_dtype };
pub const Launch = struct {
    grid: [3]c_int,
    tg: [3]c_int,
    ints: [2]IntArg = undefined,
    n_int: u8 = 0,
    t_dtype: ?mlx.mlx_dtype = null,
    outs: [2]OutArg = undefined,
    n_out: u8 = 0,

    fn int(l: Launch, name: [*:0]const u8, v: u32) Launch {
        var o = l;
        o.ints[o.n_int] = .{ .name = name, .v = @intCast(v) };
        o.n_int += 1;
        return o;
    }

    fn out(l: Launch, shape: []const u32, dtype: mlx.mlx_dtype) Launch {
        var o = l;
        var sh: [3]c_int = @splat(0);
        for (shape, 0..) |d, i| sh[i] = @intCast(d);
        o.outs[o.n_out] = .{ .shape = sh, .ndim = @intCast(shape.len), .dtype = dtype };
        o.n_out += 1;
        return o;
    }
};

fn configOf(l: Launch) !Config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    for (l.outs[0..l.n_out]) |*o| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &o.shape, o.ndim, o.dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, l.grid[0], l.grid[1], l.grid[2]));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, l.tg[0], l.tg[1], l.tg[2]));
    for (l.ints[0..l.n_int]) |t| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, t.name, t.v));
    if (l.t_dtype) |d| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", d));
    return c;
}

fn apply(g: *G, k: MKernel, cfg: Config, ins: []const T, outs: []T) !void {
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

fn ci(x: u32) c_int {
    return @intCast(x);
}

// ── The launches (pure: the tests read them) ──

pub fn inRinLaunch(p: u32) Launch {
    const l: Launch = .{ .grid = .{ ci(hidden / 128 * 64), ci(p), 1 }, .tg = .{ 64, 1, 1 } };
    return l.int("HID", hidden).out(&.{ p, hidden }, .float32).out(&.{ p, hidden }, .float32);
}

/// The GEMV of one projection at P pairs: in -> out (gate / up: hidden -> inter; down: inter -> hidden).
pub fn gemvLaunch(p: u32, in: u32, out: u32) Launch {
    const l: Launch = .{ .grid = .{ ci(out / 16 * 32), ci(p), 1 }, .tg = .{ 32, 1, 1 } };
    return l.int("IT", in / 16).int("OT", out / 16).out(&.{ p, out }, .float32);
}

/// in_rin's / dpost's (width hidden) and gu_epi's / din_rin's (width inter) launch: one 64-thread group per 128 columns.
pub fn rotLaunch(p: u32, width: u32, name: [*:0]const u8) Launch {
    const l: Launch = .{ .grid = .{ ci(width / 128 * 64), ci(p), 1 }, .tg = .{ 64, 1, 1 } };
    return l.int(name, width).out(&.{ p, width }, .float32);
}

pub fn take2Launch(rows: u32) Launch {
    const l: Launch = .{ .grid = .{ ci(hidden / 128 * 32), ci(rows), 1 }, .tg = .{ 256, 1, 1 } };
    return l.int("HID", hidden).out(&.{ rows, 1, hidden }, .float16).out(&.{ rows, 1, hidden }, .float16);
}

pub fn gemmGuLaunch(rows: u32, tgs: u32) Launch {
    const l: Launch = .{ .grid = .{ ci(256 * tgs), 1, 1 }, .tg = .{ 256, 1, 1 } };
    return l.int("KD", hidden).int("ND", inter).out(&.{ rows, inter }, .float32).out(&.{ rows, inter }, .float32);
}

pub fn onePassLaunch(rows: u32) Launch {
    const l: Launch = .{ .grid = .{ ci(inter / 128 * 32), ci(rows), 1 }, .tg = .{ ci(onepass_sgs * 32), 1, 1 } };
    return l.int("INTER", inter).int("SGS", onepass_sgs).out(&.{ rows, 1, inter }, .float16);
}

pub fn gemmDnLaunch(rows: u32, tgs: u32) Launch {
    const l: Launch = .{ .grid = .{ ci(256 * tgs), 1, 1 }, .tg = .{ 256, 1, 1 } };
    return l.int("KD", inter).int("ND", hidden).out(&.{ rows, hidden }, .float32);
}

pub fn widen1Launch(rows: u32, dtype: mlx.mlx_dtype) Launch {
    var l: Launch = .{ .grid = .{ ci(hidden / 128 * 32), ci(rows), 1 }, .tg = .{ 256, 1, 1 }, .t_dtype = dtype };
    l = l.int("HID", hidden).out(&.{ rows, 1, hidden }, dtype);
    return l;
}

/// dpost: rotLaunch at hidden, its output in bf16 (the expert output type: the glm5_next combines read bf16 rows).
pub fn dpostLaunch(p: u32) Launch {
    const l: Launch = .{ .grid = .{ ci(hidden / 128 * 64), ci(p), 1 }, .tg = .{ 64, 1, 1 }, .t_dtype = .bfloat16 };
    return l.int("HID", hidden).out(&.{ p, hidden }, .bfloat16);
}

/// The decode configs of one wave width (prepared at install).
const DecodeCfg = struct { in_rin: Config, gemv_gu: Config, gu_epi: Config, din_rin: Config, gemv_dn: Config, dpost: Config };

pub const Kernels = struct {
    k: [n_kernels]MKernel = @splat(.{ .ctx = null }),
    /// [P - 1]: the decode launches of a P-pair wave.
    dec: [max_pairs]?DecodeCfg = @splat(null),
    /// [P - 1]: int32 [P] 0..P-1 (in_rin's row map: the pairs' rows are taken already), kept.
    tok: [max_pairs]T = @splat(.{ .ctx = null }),

    pub fn deinit(self: *Kernels) void {
        for (self.k) |k| if (k.ctx != null) mtl.mlx_fast_metal_kernel_free(k);
        for (self.dec) |d| if (d) |c| for ([_]Config{ c.in_rin, c.gemv_gu, c.gu_epi, c.din_rin, c.gemv_dn, c.dpost }) |x| {
            _ = mlx.mlx_fast_metal_kernel_config_free(x);
        };
        for (self.tok) |t| if (t.ctx != null) {
            _ = mlx.mlx_array_free(t);
        };
        self.* = .{};
    }

    fn get(self: *const Kernels, k: Kernel) MKernel {
        return self.k[@intFromEnum(k)];
    }
};

/// Construction, once (the module's GPU stream): the pins, every text built, the decode launches prepared for every wave
/// width, one launch of each decode text over zeros as the compile check. Refused by name.
pub fn install(a: std.mem.Allocator, s: mlx.mlx_stream, diag: *sdk.Diag) !*Kernels {
    checkPins() catch |e| {
        diag.set("glm5_next exl3 kernels: a text is not the pinned one", .{});
        return e;
    };
    if (!mlx.streamIsGpu(s)) {
        diag.set("glm5_next exl3 kernels: the module's stream is not a GPU stream", .{});
        return error.Glm5NextExl3NotGpu;
    }
    const self = try a.create(Kernels);
    self.* = .{};
    errdefer {
        self.deinit();
        a.destroy(self);
    }
    inline for (comptime std.enums.values(Kernel), 0..) |k, i| {
        const sig = sigOf(k);
        const in_vec = mlx.mlx_vector_string_new_data(@ptrCast(sig.ins.ptr), sig.ins.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(@ptrCast(sig.outs.ptr), sig.outs.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        self.k[i] = mtl.mlx_fast_metal_kernel_new("glm5n_exl3_" ++ @tagName(k), in_vec, out_vec, texts[i], sig.header, sig.contiguous, false);
        if (self.k[i].ctx == null) {
            diag.set("glm5_next exl3 kernels: glm5n_exl3_{s} was not created", .{@tagName(k)});
            return error.Glm5NextExl3Kernel;
        }
    }
    for (0..max_pairs) |i| {
        const p: u32 = @intCast(i + 1);
        self.dec[i] = .{
            .in_rin = try configOf(inRinLaunch(p)),
            .gemv_gu = try configOf(gemvLaunch(p, hidden, inter)),
            .gu_epi = try configOf(rotLaunch(p, inter, "INTER")),
            .din_rin = try configOf(rotLaunch(p, inter, "INTER")),
            .gemv_dn = try configOf(gemvLaunch(p, inter, hidden)),
            .dpost = try configOf(dpostLaunch(p)),
        };
        var idx: [max_pairs]i32 = undefined;
        for (idx[0..p], 0..) |*v, j| v.* = @intCast(j);
        self.tok[i] = mlx.mlx_array_new_data(&idx, &[_]c_int{ci(p)}, 1, .int32);
        if (self.tok[i].ctx == null) return error.MlxError;
    }
    compileCheck(a, s, self) catch |e| {
        var eb: [400]u8 = undefined;
        diag.set("glm5_next exl3 kernels: the decode compile check failed ({s}): {s}", .{ @errorName(e), mlx.takeError(&eb) orelse "" });
        return e;
    };
    return self;
}

/// One decode wave of one pair over zero banks of one slot row, evaluated.
fn compileCheck(a: std.mem.Allocator, s: mlx.mlx_stream, k: *const Kernels) !void {
    var g = try G.init(a, s);
    defer g.deinit();
    const gu = projDims(.gate_up);
    const dn = projDims(.down);
    const pg: Proj = .{ .code = try g.zeros(&.{ 1, ci(gu.code[0]), ci(gu.code[1]), ci(gu.code[2]) }, .int16), .rout = try g.zeros(&.{ 1, ci(gu.rout) }, .float16), .rin = try g.zeros(&.{ 1, ci(gu.rin) }, .float16) };
    const pd: Proj = .{ .code = try g.zeros(&.{ 1, ci(dn.code[0]), ci(dn.code[1]), ci(dn.code[2]) }, .int16), .rout = try g.zeros(&.{ 1, ci(dn.rout) }, .float16), .rin = try g.zeros(&.{ 1, ci(dn.rin) }, .float16) };
    const banks: [n_banks]BankArrays = @splat(.{ .gate = pg, .up = pg, .down = pd });
    const ids = try g.zeros(&.{1}, .uint32);
    const x = try g.zeros(&.{ 1, ci(hidden) }, .bfloat16);
    const h = try gateUp(k, &g, x, ids, &banks);
    try g.evalAll(&.{try down(k, &g, h, ids, &banks)});
}

// ── Decode: the banked PREP=rin chain ──

/// x bf16 [P, hidden] (each pair's token row), ids u32 [P] packed `bank << 24 | slot row` -> the clamped SwiGLU f32
/// [P, inter]: in_rin, the gate GEMV, the up GEMV, gu_epi.
pub fn gateUp(k: *const Kernels, g: *G, x: T, ids: T, banks: *const [n_banks]BankArrays) !T {
    const p: u32 = @intCast(g.shapeOf(x).d[0]);
    const c = k.dec[p - 1].?;
    var xs: [2]T = undefined;
    try apply(g, k.get(.in_rin), c.in_rin, &.{ x, k.tok[p - 1], banks[0].gate.rin, banks[1].gate.rin, banks[2].gate.rin, banks[0].up.rin, banks[1].up.rin, banks[2].up.rin, ids }, &xs);
    var zg: [1]T = undefined;
    var zu: [1]T = undefined;
    try apply(g, k.get(.gemv), c.gemv_gu, &.{ xs[0], ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code }, &zg);
    try apply(g, k.get(.gemv), c.gemv_gu, &.{ xs[1], ids, banks[0].up.code, banks[1].up.code, banks[2].up.code }, &zu);
    var out: [1]T = undefined;
    try apply(g, k.get(.gu_epi), c.gu_epi, &.{ zg[0], zu[0], banks[0].gate.rout, banks[1].gate.rout, banks[2].gate.rout, banks[0].up.rout, banks[1].up.rout, banks[2].up.rout, ids }, &out);
    return out[0];
}

/// h f32 [P, inter] (gateUp's), the same packed ids -> bf16 [P, hidden]: din_rin, the down GEMV, dpost (its f32 word
/// rounded once on store).
pub fn down(k: *const Kernels, g: *G, h: T, ids: T, banks: *const [n_banks]BankArrays) !T {
    const p: u32 = @intCast(g.shapeOf(h).d[0]);
    const c = k.dec[p - 1].?;
    var hd: [1]T = undefined;
    try apply(g, k.get(.din_rin), c.din_rin, &.{ h, banks[0].down.rin, banks[1].down.rin, banks[2].down.rin, ids }, &hd);
    var zd: [1]T = undefined;
    try apply(g, k.get(.gemv), c.gemv_dn, &.{ hd[0], ids, banks[0].down.code, banks[1].down.code, banks[2].down.code }, &zd);
    var out: [1]T = undefined;
    try apply(g, k.get(.dpost), c.dpost, &.{ zd[0], banks[0].down.rout, banks[1].down.rout, banks[2].down.rout, ids }, &out);
    return out[0];
}

// ── Prompt: DIG-X over one wave (one bank, <= 16 experts, rows grouped by expert in table order) ──

fn launchOnce(g: *G, k: MKernel, l: Launch, ins: []const T, outs: []T) !void {
    const c = try configOf(l);
    defer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try apply(g, k, c, ins, outs);
}

/// act bf16 [A, hidden], ridx i32 [rows] (each row's act row), rhs u32 [rows] (each row's table entry), tbl i32 [80]
/// (the wave table), the gate / up rin -> (f16(t128(act * rin_g[slot])), f16(t128(act * rin_u[slot]))) [rows, 1, hidden].
pub fn take2(k: *const Kernels, g: *G, act: T, ridx: T, rhs: T, tbl: T, rin_g: T, rin_u: T) ![2]T {
    const rows: u32 = @intCast(g.shapeOf(ridx).d[0]);
    var out: [2]T = undefined;
    try launchOnce(g, k.get(.take2v), take2Launch(rows), &.{ act, ridx, rhs, tbl, rin_g, rin_u }, &out);
    return out;
}

/// x0 / x1 f16 [rows, 1, hidden] (take2's), the bank's gate / up code, tbl (`gu_tiles` at `m_tile`), its threadgroups ->
/// (zg, zu) f32 [rows, inter].
pub fn gemmGateUp(k: *const Kernels, g: *G, x0: T, x1: T, code_g: T, code_u: T, tbl: T, tgs: u32) ![2]T {
    const rows: u32 = @intCast(g.shapeOf(x0).d[0]);
    var out: [2]T = undefined;
    try launchOnce(g, k.get(.gemm_gu), gemmGuLaunch(rows, tgs), &.{ x0, x1, code_g, code_u, tbl }, &out);
    return out;
}

/// zg / zu f32 [rows, inter] -> hd = f16(t128(clamped SwiGLU(t128(zg) rout_g, t128(zu) rout_u) rin_d)) [rows, 1, inter].
pub fn onePass(k: *const Kernels, g: *G, zg: T, zu: T, rhs: T, tbl: T, rout_g: T, rout_u: T, rin_d: T) !T {
    const rows: u32 = @intCast(g.shapeOf(zg).d[0]);
    var out: [1]T = undefined;
    try launchOnce(g, k.get(.onepass), onePassLaunch(rows), &.{ zg, zu, rhs, tbl, rout_g, rout_u, rin_d }, &out);
    return out[0];
}

/// hd f16 [rows, 1, inter] (onePass's), the bank's down code, tbl (`dn_tiles` at `m_tile`), its threadgroups -> zd f32
/// [rows, hidden].
pub fn gemmDown(k: *const Kernels, g: *G, hd: T, code_d: T, tbl: T, tgs: u32) !T {
    const rows: u32 = @intCast(g.shapeOf(hd).d[0]);
    var out: [1]T = undefined;
    try launchOnce(g, k.get(.gemm_dn), gemmDnLaunch(rows, tgs), &.{ hd, code_d, tbl }, &out);
    return out[0];
}

/// zd f32 [rows, hidden] -> t128(zd) * rout_d[slot] [rows, 1, hidden] in `dtype` (the f32 word, rounded once on store).
pub fn widen1(k: *const Kernels, g: *G, zd: T, rhs: T, tbl: T, rout_d: T, dtype: mlx.mlx_dtype) !T {
    const rows: u32 = @intCast(g.shapeOf(zd).d[0]);
    var out: [1]T = undefined;
    try launchOnce(g, k.get(.widen1), widen1Launch(rows, dtype), &.{ zd, rhs, tbl, rout_d }, &out);
    return out[0];
}

const testing = std.testing;

test "glm5_next exl3 kernels: the texts are the pinned ones, every DeepSeek shape literal replaced, GLM's geometry" {
    try checkPins();
    // Outside comments (the generated head names the DeepSeek base text), no DeepSeek shape literal is left.
    for (texts) |t| {
        var lines = std.mem.splitScalar(u8, t, '\n');
        while (lines.next()) |line| {
            const code = if (std.mem.indexOf(u8, line, "//")) |c| line[0..c] else line;
            for ([_][]const u8{ "5120", "2304", "2211840", "320u", "144u" }) |lit| try testing.expect(std.mem.indexOf(u8, code, lit) == null);
        }
    }
    try testing.expect(std.mem.indexOf(u8, header_onepass, "glm5n_dig2_swiglu_one") != null);
    try testing.expectEqual(@as(u32, 64), gu_tiles);
    try testing.expectEqual(@as(u32, 64), dn_tiles);
    try testing.expectEqual([3]u32{ 256, 128, 48 }, projDims(.gate_up).code);
    try testing.expectEqual([3]u32{ 128, 256, 48 }, projDims(.down).code);
    // One record: three projections' code + rout + rin, no global-scale word (9,474,048 B: a whole number of pages).
    const gu = projDims(.gate_up);
    const rec = 2 * (gu.code[0] * gu.code[1] * gu.code[2] * 2 + gu.rout * 2 + gu.rin * 2) + (projDims(.down).code[0] * projDims(.down).code[1] * 48 * 2 + hidden * 2 + inter * 2);
    try testing.expectEqual(@as(u32, 9_474_048), rec);
    try testing.expectEqual(@as(u32, 0), rec % 4096);
}

test "glm5_next exl3 kernels: the decode launches at every wave width cover the outputs (one group per 16 outputs / 128 columns)" {
    for (1..max_pairs + 1) |pi| {
        const p: u32 = @intCast(pi);
        const gv = gemvLaunch(p, hidden, inter);
        try testing.expectEqual(@as(c_int, ci(inter / 16 * 32)), gv.grid[0]);
        try testing.expectEqual(@as(c_int, ci(p)), gv.grid[1]);
        const dv = gemvLaunch(p, inter, hidden);
        try testing.expectEqual(@as(c_int, ci(hidden / 16 * 32)), dv.grid[0]);
        try testing.expectEqual(@as(c_int, 128), dv.ints[0].v);
        try testing.expectEqual(@as(c_int, ci(hidden / 128 * 64)), inRinLaunch(p).grid[0]);
        try testing.expectEqual(@as(c_int, ci(inter / 128 * 64)), rotLaunch(p, inter, "INTER").grid[0]);
    }
}
