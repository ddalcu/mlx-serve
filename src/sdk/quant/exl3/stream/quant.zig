//! The EXL3 routed-expert quant, C2's first client (`quant`): the streamed EXL3 bank's pinned
//! Metal texts. The texts decode the mul1 codebook at K = 3 and are compiled for hidden 5120 /
//! inter 2304.
//!   - Decode (at most 48 routed rows) runs the tier of record's PREP=rin chain: in_rin, the two
//!     GEMVs, gu_epi (the clamped SwiGLU, fused), din_rin, the down GEMV, dpost. Every launch
//!     comes from a per-M table of prepared launches.
//!   - Prefill runs the DIG-X waves (DIG2 onepass, SHAPE's schedule), one wave state per layer.
//! Its kernels (`kernels`, 18 of the set) are self-checked by `accept` on the kernel set the load
//! context owns. The routes below the C2 section are the lanes' own launches, moved here verbatim
//! from exl3_kernel_ops.zig.

const std = @import("std");
const env = @import("mlx_host").env;
const kset = @import("../../../weights/streamed/sdk_ext/kernel_set.zig");
const mlx = @import("sdk").mlx;
const xk = @import("kernels.zig");
const selfcheck = @import("selfcheck.zig");
const kr = sdk_ext.kernels.Routes(xk);
const ks = sdk_ext.kernels.KernelSet(xk);
const quant = @import("../../../weights/streamed/sdk_ext.zig").quant;
const sdk = @import("sdk");
const sdk_ext = @import("../../../weights/streamed/sdk_ext.zig");

const Allocator = std.mem.Allocator;
const Kernel = xk.Kernel;
const Entry = xk.Entry;
const Vars = xk.Vars;
const LaunchConfig = xk.LaunchConfig;
const Dtype = mlx.mlx_dtype;
const Diag = xk.Diag;
const Refusal = kr.Refusal;
const refuse = kr.refuse;
const argOf = kr.argOf;
const expectInput = kr.expectInput;
const launchRule = kr.launchRule;
const rowsOf = kr.rowsOf;

/// Which text set a code array takes: 0 for K = 3 (48 int16 per tile), 1 for K = 2 (32). A layer's K reaches every
/// dispatch this way: the bank's per-layer arrays are shaped by the v2 manifest's K (the claim admits only bank_ks).
fn kIndexOf(comptime G: type, g: *G, code: G.T) usize {
    return @intFromBool(rowsOf(G, g, code, 3) == 32);
}
const Statics = kr.Statics;
const RowPlans = kr.RowPlans;
const rowsVars = kr.rowsVars;
const no_vars = kr.no_vars;

// ── C2: the quant ──

pub const name = "exl3-mul1-k3";

/// G5: the kernel set this quant self-checks at accept is pinned by the registry's manifest.
pub const kernel_pin: sdk_ext.kernels.Pin = .{ .manifest_sha256 = xk.manifest_sha256 };

/// This consumer's subset of the kernel set: the EXL3 families (the decode GEMV, the rin stage,
/// the rebuild, DIG-X and its golden-tile check texts).
pub const kernels = [_]Kernel{
    .dsv41_exl3_mul1h_k3_2304,
    .dsv41_exl3_mul1h_k3_5120,
    .q3_exl3_prep_in_rin,
    .q3_exl3_prep_gu_epi,
    .q3_exl3_prep_din_rin,
    .q3_moeprep_dpost,
    .q3_prefill_fused_exl3x3_mul1lut_k3_bf16,
    .q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3,
    .q3_prefill_dig_gemm_2304x5120_xmul1hk3,
    .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128,
    .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128,
    .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1,
    .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut,
    .q3_prefill_dig_rot_take2_5120,
    .dsv41_prefill_dig_take2v_5120,
    .q3_prefill_dig_rot_roundx_2304,
    .q3_prefill_dig2_swiglu_2304_x,
    .q3_prefill_dig_rot_widen2_2304,
    .q3_prefill_dig_rot_widen1_5120,
    .dsv41_prefill_dig_rot_widen1_5120_obf16,
    .q3_exl3_dig_decmat_5120x2304_mul1hk3,
    .q3_exl3_dig_decmat_2304x5120_mul1hk3,
    .q3_exl3_dig_decmat_5120x2304_mul1k3,
    .q3_exl3_dig_decmat_2304x5120_mul1k3,
    .dsv41_exl3_pair_k3_5120,
    .dsv41_exl3_guone_k3_2304,
    .dsv41_exl3_b3_mul1h_k3_2304,
    .dsv41_exl3_b3_mul1h_k3_5120,
    .dsv41_exl3_b3_prep_in_rin,
    .dsv41_exl3_b3_prep_gu_epi,
    .dsv41_exl3_b3_prep_din_rin,
    .dsv41_exl3_b3_moeprep_dpost,
    .dsv41_exl3_b3_pair_k3_5120,
    .dsv41_exl3_b3_guone_k3_2304,
    .dsv41_exl3_mul1h_k2_2304,
    .dsv41_exl3_mul1h_k2_5120,
    .dsv41_exl3_pair_k2_5120,
    .dsv41_exl3_guone_k2_2304,
    .dsv41_exl3_b3_mul1h_k2_2304,
    .dsv41_exl3_b3_mul1h_k2_5120,
    .dsv41_exl3_b3_pair_k2_5120,
    .dsv41_exl3_b3_guone_k2_2304,
    .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128,
    .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128,
    .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128w1,
    .q3_exl3_dig_decmat_5120x2304_mul1hk2,
    .q3_exl3_dig_decmat_2304x5120_mul1hk2,
};

/// The routed decode forms (`DSV41_CELL_ROUTED_FORMS`): each independently selectable, chosen at construction.
pub const Forms = struct { down_pair: bool = false, gu_one: bool = false };

/// One projection's per-slot arrays: the streamer's `ProjArrays` {code, rout, rin}.
pub fn Arrays(comptime T: type) type {
    return ProjArrays(T);
}

/// The activation the texts fuse: the clamped SwiGLU at 10.0 (the literals of
/// q3_exl3_prep_gu_epi.metal:47-52 / :64-69 and of header_dig2_x.metal:34-37).
pub const fused_act: quant.Activation = .{ .swiglu_clamped = 10.0 };
/// The dims the texts are compiled for.
pub const compiled_hidden: u32 = 5120;
pub const compiled_inter: u32 = 2304;
/// The decode tables' widest call: 8 verify rows x top-k 6.
pub const decode_table_rows: u32 = 48;

/// The bank format the texts decode, field by field, as the bank manifest's `quantization`
/// object writes it (exllamav3 v1.4.2: codebook.cuh, exl3_dq.cuh, pack.cu, hadamard.cu).
pub const format = struct {
    pub const mode = "exl3";
    pub const codebook = xk.bank_codebook;
    pub const multiplier = xk.bank_multiplier;
    pub const tile_size = 16;
    pub const tile_layout = "exl3-tensor-core";
    pub const tile_bitstream = "msb-first-swap16";
    pub const hadamard_size = 128;
    pub const hadamard_order = "sylvester-natural";
    pub const hadamard_scale = "1/sqrt(128)";
    pub const scale_rin = "exl3.suh";
    pub const scale_rout = "exl3.svh";
};

fn strIs(v: std.json.Value, field: []const u8, want: []const u8) bool {
    const s = quant.str(v, field) orelse return false;
    return std.mem.eql(u8, s, want);
}

fn intIs(v: std.json.Value, field: []const u8, want: i64) bool {
    return (quant.int(v, field) orelse return false) == want;
}

/// The projections' tensors at K (bankv2 layer_segments): per projection code I16 [in/16,
/// out/16, 16 K], rout F16 [out], rin F16 [in]; gate / up map hidden -> inter, down inter -> hidden.
fn segmentOk(s: quant.Segment, k: u32) ?[]const u8 {
    const P3 = struct { prefix: []const u8, in: u64, out: u64 };
    const projs = [_]P3{ .{ .prefix = "gate_proj.", .in = compiled_hidden, .out = compiled_inter }, .{ .prefix = "up_proj.", .in = compiled_hidden, .out = compiled_inter }, .{ .prefix = "down_proj.", .in = compiled_inter, .out = compiled_hidden } };
    for (projs) |p| {
        if (!std.mem.startsWith(u8, s.name, p.prefix)) continue;
        const part = s.name[p.prefix.len..];
        const want_dtype: []const u8, const want: []const u64 = if (std.mem.eql(u8, part, "code"))
            .{ "I16", &.{ p.in / 16, p.out / 16, 16 * @as(u64, k) } }
        else if (std.mem.eql(u8, part, "rout"))
            .{ "F16", &.{p.out} }
        else if (std.mem.eql(u8, part, "rin"))
            .{ "F16", &.{p.in} }
        else
            return "an unknown tensor";
        if (!std.mem.eql(u8, s.dtype, want_dtype)) return "its dtype";
        if (!std.mem.eql(u64, s.shape, want)) return "its shape";
        return null;
    }
    return "an unknown projection";
}

/// At load, once per weight group: `.native` for a bank these texts decode in full (the
/// format's every field, every layer's K, the compiled dims, every layer's nine tensors), else
/// null with the first mismatch in `why`.
pub fn claims(peek: *const quant.BankPeek, why: ?*Diag) ?quant.Priority {
    const q = peek.quantization;
    if (!strIs(q, "mode", format.mode)) return quant.decline(why, "exl3 quant: quantization.mode \"{s}\" (the texts decode {s})", .{ quant.str(q, "mode") orelse "", format.mode });
    if (!strIs(q, "codebook", format.codebook)) return quant.decline(why, "exl3 quant: quantization.codebook \"{s}\" (the texts decode {s})", .{ quant.str(q, "codebook") orelse "", format.codebook });
    if (!intIs(q, "codebook_multiplier", format.multiplier)) return quant.decline(why, "exl3 quant: quantization.codebook_multiplier {?d} (mul1 is {d})", .{ quant.int(q, "codebook_multiplier"), format.multiplier });
    const tile = quant.obj(q, "tile");
    if (!intIs(tile, "size", format.tile_size) or !strIs(tile, "layout", format.tile_layout) or !strIs(tile, "bitstream", format.tile_bitstream))
        return quant.decline(why, "exl3 quant: quantization.tile {?d} / {s} / {s} (the texts read {d} / {s} / {s})", .{ quant.int(tile, "size"), quant.str(tile, "layout") orelse "", quant.str(tile, "bitstream") orelse "", format.tile_size, format.tile_layout, format.tile_bitstream });
    const had = quant.obj(q, "hadamard");
    if (!intIs(had, "size", format.hadamard_size) or !strIs(had, "order", format.hadamard_order) or !strIs(had, "scale", format.hadamard_scale))
        return quant.decline(why, "exl3 quant: quantization.hadamard {?d} / {s} / {s} (the texts rotate {d} / {s} / {s})", .{ quant.int(had, "size"), quant.str(had, "order") orelse "", quant.str(had, "scale") orelse "", format.hadamard_size, format.hadamard_order, format.hadamard_scale });
    const sc = quant.obj(q, "scales");
    if (!strIs(sc, "rin", format.scale_rin) or !strIs(sc, "rout", format.scale_rout))
        return quant.decline(why, "exl3 quant: quantization.scales rin {s} / rout {s} (the texts read {s} / {s})", .{ quant.str(sc, "rin") orelse "", quant.str(sc, "rout") orelse "", format.scale_rin, format.scale_rout });
    if (peek.hidden != compiled_hidden or peek.inter != compiled_inter)
        return quant.decline(why, "exl3 quant: dims hidden {d} / inter {d} (the texts are compiled for {d} / {d})", .{ peek.hidden, peek.inter, compiled_hidden, compiled_inter });
    if (peek.n_layers == 0 or peek.layers.len != peek.n_layers) return quant.decline(why, "exl3 quant: {d} layer entries for dims.n_layers {d}", .{ peek.layers.len, peek.n_layers });
    for (peek.layers, 0..) |l, li| {
        if (std.mem.indexOfScalar(u32, &xk.bank_ks, l.bits) == null) return quant.decline(why, "exl3 quant: layer {d} K {d} (the texts decode K {any})", .{ li, l.bits, xk.bank_ks });
        if (l.segments.len != 9) return quant.decline(why, "exl3 quant: layer {d} has {d} tensors (9)", .{ li, l.segments.len });
        for (l.segments) |s| if (segmentOk(s, l.bits)) |bad| return quant.decline(why, "exl3 quant: layer {d} segment {s}: {s} ({s} {any})", .{ li, s.name, bad, s.dtype, s.shape });
    }
    return .native;
}

/// The Spec the texts implement, checked at construction (never a per-call branch): the fused
/// activation, the compiled dims, a bf16 MoE input (in_rin and take2 read bf16 rows) and decode
/// rows (8 tokens x top_k) within the 48-row tables.
pub fn checkSpec(spec: quant.Spec, diag: *Diag) quant.Refusal!void {
    const lim = fused_act.swiglu_clamped;
    switch (spec.act) {
        .swiglu_clamped => |l| if (l != lim) return quant.refuse(diag, error.ActivationNotFused, "exl3 quant: SwiGLU clamped at {d} (the texts fuse {d})", .{ l, lim }),
        else => return quant.refuse(diag, error.ActivationNotFused, "exl3 quant: activation {t} (the texts fuse the SwiGLU clamped at {d})", .{ spec.act, lim }),
    }
    if (spec.hidden != compiled_hidden or spec.inter != compiled_inter) return quant.refuse(diag, error.DimsNotImplemented, "exl3 quant: dims {d} / {d} (the texts are compiled for {d} / {d})", .{ spec.hidden, spec.inter, compiled_hidden, compiled_inter });
    if (spec.input != .bfloat16) return quant.refuse(diag, error.InputDtype, "exl3 quant: MoE input {t} (in_rin and take2 read bf16)", .{spec.input});
    if (spec.top_k == 0 or spec.top_k * 8 > decode_table_rows) return quant.refuse(diag, error.TopKTooWide, "exl3 quant: top_k {d}: 8 decode rows x top_k exceed the {d}-row tables", .{ spec.top_k, decode_table_rows });
    if (spec.n_layers == 0) return quant.refuse(diag, error.DimsNotImplemented, "exl3 quant: no layers", .{});
}

/// The quant accepted on backend G: the decode chain's routes (built once: statics, the per-M
/// launch tables, prepared configs), the row maps and one prefill wave state per layer.
pub fn Accepted(comptime G: type) type {
    return struct {
        const Self = @This();
        const A = ProjArrays(G.T);
        pub const max_decode_rows: u32 = decode_table_rows;
        a: Allocator,
        reg: *const xk.Registry,
        /// this subset's self-check results (its receipt: `report.writeJsonLines(a)`)
        report: selfcheck.Report = .{},
        gemv: Gemv(G),
        prep: RinPrep(G),
        /// tok[m - 1] = int32 [m] 0..m-1, kept: in_rin's row map for m routed rows already taken
        tok: [48]G.T,
        n_tok: usize = 0,
        /// one DIG-X wave state per layer (the lane has one dispatcher per layer)
        waves: []DigXPrefill(G) = &.{},
        /// the waves' down stage is the fused down GEMM (`routeFusedDown`)
        fused_down: bool = false,
        /// kv16-opt: the waves' expert outputs in bf16 (`routeExpertBf16`)
        expert_bf16: bool = false,
        /// the decode GEMVs' routed forms (`routeForms`); all false: the stock mul1h texts
        forms: Forms = .{},
        /// the banked route (`routeBanked`): every bank's rows of a wave in one launch per stage; null: not installed
        banked: ?Banked(G) = null,

        /// Once per bank bind and per grow: the three projections' arrays are the kernels' (cap
        /// within the kernels' bound, shapes, dtypes).
        pub fn checkBank(self: *const Self, g: *G, bank: quant.BankArrays(A), diag: *Diag) !void {
            try checkProjArrays(G, g, self.reg, .gate, bank.gate, diag);
            try checkProjArrays(G, g, self.reg, .up, bank.up, diag);
            try checkProjArrays(G, g, self.reg, .down, bank.down, diag);
        }

        /// x [rows, 5120] bf16 (the routed rows, taken), slot_ids u32 [rows], rows 1..48 ->
        /// the clamped SwiGLU [rows, 2304] f32: in_rin -> the gate and up GEMVs -> gu_epi.
        pub fn gateUp(self: *const Self, g: *G, x: G.T, slot_ids: G.T, gate: A, up: A) !G.T {
            const rows = rowsOf(G, g, x, 0);
            std.debug.assert(!(rows < 1 or rows > max_decode_rows));
            const xs = try self.prep.inRin(g, x, self.tok[rows - 1], gate.rin, up.rin, slot_ids);
            const z = try self.gemv.projectGu(g, xs[0], xs[1], slot_ids, gate.code, up.code);
            return self.prep.guEpi(g, z[0], z[1], gate.rout, up.rout, slot_ids);
        }

        /// h [rows, 2304] f32 (gateUp's), slot_ids u32 [rows] -> [rows, 5120] f32: din_rin ->
        /// the down GEMV -> dpost.
        pub fn down(self: *const Self, g: *G, h: G.T, slot_ids: G.T, d: A) !G.T {
            const hd = try self.prep.dinRin(g, h, d.rin, slot_ids);
            const zd = try self.gemv.project(g, .down, hd, slot_ids, d.code);
            return self.prep.dpost(g, zd, d.rout, slot_ids);
        }

        /// The banked route (`routeBanked` installed it): x [rows, 5120] bf16, ids u32 [rows] packed (bank << 24 | slot
        /// row), `banks` the three banks' arrays (base, ext, transient; a bank no row names may repeat another's) -> the
        /// clamped SwiGLU [rows, 2304] f32, every word gateUp's per bank.
        pub fn gateUpBanked(self: *const Self, g: *G, x: G.T, ids: G.T, banks: *const [3]quant.BankArrays(A)) !G.T {
            const rows = rowsOf(G, g, x, 0);
            std.debug.assert(!(rows < 1 or rows > max_decode_rows));
            return self.banked.?.gateUp(g, x, self.tok[rows - 1], ids, banks);
        }

        /// The banked route's down: h [rows, 2304] f32 (gateUpBanked's), the same packed ids -> [rows, 5120] f32.
        pub fn downBanked(self: *const Self, g: *G, h: G.T, ids: G.T, banks: *const [3]quant.BankArrays(A)) !G.T {
            return self.banked.?.down(g, h, ids, banks);
        }

        /// Layer `layer`'s DIG-X waves over the call's routed rows (`DigXPrefill.call`): a KEPT
        /// f32 [rows, 5120] in routed-row order (release it).
        pub fn prefill(self: *Self, g: *G, layer: u32, x: G.T, rows: quant.PrefillRows, bank: quant.BankArrays(A)) !G.T {
            return self.waves[layer].call(g, x, rows, bank);
        }

        /// `prefill` unjoined: the waves' KEPT outputs (wave order) and each wave-ordered row's
        /// assignment row (`DigXPrefill.callParts`).
        pub fn prefillParts(self: *Self, g: *G, layer: u32, x: G.T, rows: quant.PrefillRows, bank: quant.BankArrays(A), alloc: Allocator, outs: *std.ArrayList(G.T), pos: *std.ArrayList(u32)) !void {
            return self.waves[layer].callParts(g, x, rows, bank, alloc, outs, pos);
        }

        /// The prefill boundary: every layer's waves still in flight evaluated, oldest first.
        pub fn finishPrefill(self: *Self, g: *G) !void {
            for (self.waves) |*w| try w.finish(g);
        }

        /// The fused down GEMM arm, at construction (before any prefill): its self-checks (compile, composition,
        /// fused) on `set`, the accepting set, join the report, then every layer's waves launch it in place of the
        /// 128-row down text and rot_widen1. A failed check refuses it by name (SelfCheckFailed, `diag`) and the
        /// waves stay stock.
        pub fn routeFusedDown(self: *Self, set: *const ks.Set, diag: *Diag) !void {
            try set.selfCheck(self.a, &w1_texts, .compile, &self.report, diag);
            for (self.waves) |*w| w.installDown(.fused);
            self.fused_down = true;
        }

        /// The decode route texts (the forms', the banked route's) compiled at construction with the rest of the set
        /// (`.compile`: compile, bind, one launch), so no route text compiles at its first decode launch.
        pub fn compileTexts(self: *Self, set: *const ks.Set, texts: []const Kernel, diag: *Diag) !void {
            try set.selfCheck(self.a, texts, .compile, &self.report, diag);
        }

        /// kv16-opt, at construction (before any prefill): every layer's DIG-X waves store their expert outputs as bf16
        /// (rot_widen1's bf16-out instantiation: the same f32 word, rounded to bf16), the reference's `Expert` output
        /// type (`type_as(x)`); the combines read them in f32 as before. The fused down GEMM's epilogue writes f32, so it
        /// refuses with the fused arm installed.
        pub fn routeExpertBf16(self: *Self) !void {
            if (self.fused_down) return error.ExpertBf16NeedsChainDown;
            for (self.waves) |*w| w.dig.widen1_e = self.reg.get(.dsv41_prefill_dig_rot_widen1_5120_obf16);
            self.expert_bf16 = true;
        }

        /// The routed decode forms, at construction (before any decode): the GEMVs rebuilt on the forms' texts. Exact by
        /// the registry's twin checks (every word == mul1h's) and kbench v9; no device check here.
        pub fn routeForms(self: *Self, g: *G, forms: Forms) !void {
            // The banked route aliases the installed GEMVs' statics: the forms come first.
            if (self.banked != null) return error.FormsAfterBanked;
            var next = try Gemv(G).initForms(g, self.reg, forms);
            errdefer next.deinit(g);
            self.gemv.deinit(g);
            self.gemv = next;
            self.forms = forms;
        }

        /// The banked route, at construction (before any decode; after `routeForms`, whose forms it takes): the decode
        /// stages on the banked texts, one launch per stage over a wave's rows of all three banks. Exact by the registry's
        /// twin checks (every word == the stock text's, per bank) and kbench v6d / v9b; no device check here. It allocates
        /// no device array: its statics are the installed GEMVs' (aliased by input name), its plans prepared configs.
        pub fn routeBanked(self: *Self, g: *G) !void {
            if (self.banked != null) return error.BankedRoutedTwice;
            self.banked = try Banked(G).init(g, self.reg, self.forms, &self.gemv);
        }

        /// Releases the routes (statics, prepared configs, the row maps, the wave states) and
        /// the plan's results. The kernel set stays the load context's.
        pub fn deinit(self: *Self, g: *G) void {
            if (self.banked) |*b| b.deinit(g);
            for (self.waves) |*w| w.deinit(g);
            self.a.free(self.waves);
            for (self.tok[0..self.n_tok]) |x| g.release(x);
            self.prep.deinit(g);
            self.gemv.deinit(g);
            self.report.deinit(self.a);
            self.a.destroy(self);
        }
    };
}

/// Once per backend, before the weights bind: the Spec contract, this subset's self-check plan
/// on the load context's kernel set (judged), then the routes. Refused by name: no kernel set
/// (NoKernelSet), the Spec (ActivationNotFused / DimsNotImplemented / InputDtype / TopKTooWide),
/// a self-check failure (SelfCheckFailed, `diag` naming kernel / check / site). The set's
/// launcher must be installed on `g` first (`Set.install`): a backend that prepares launches
/// prepares them through it.
/// The 128-row DIG-X GEMM texts (their 0b smoke checks them by name).
const m128_texts = [_]Kernel{ .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128, .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128, .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128, .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128 };
/// The fused down GEMM: a construction-time arm (`Accepted.routeFusedDown` self-checks it, then the waves launch it);
/// the stock accept neither routes nor checks it. Its 0b smoke checks it by name.
pub const w1_texts = [_]Kernel{ .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1, .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128w1 };
/// The table-codebook gate|up GEMM: registered, not routed (its 0b smoke checks it by name).
const lut_texts = [_]Kernel{.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut};
/// The routed decode forms: installed at construction by `Accepted.routeForms` (their twins are registry checks, run in
/// a check window and by kbench v9); the stock accept neither routes nor checks them.
pub const form_texts = [_]Kernel{ .dsv41_exl3_pair_k3_5120, .dsv41_exl3_guone_k3_2304, .dsv41_exl3_pair_k2_5120, .dsv41_exl3_guone_k2_2304 };
/// The banked route's texts: installed at construction by `Accepted.routeBanked` (their twins are registry checks, run
/// in a check window and by kbench v6d / v9b); the stock accept neither routes nor checks them.
pub const banked_texts = [_]Kernel{
    .dsv41_exl3_b3_mul1h_k3_2304,     .dsv41_exl3_b3_mul1h_k3_5120,  .dsv41_exl3_b3_prep_in_rin,  .dsv41_exl3_b3_prep_gu_epi,
    .dsv41_exl3_b3_prep_din_rin,      .dsv41_exl3_b3_moeprep_dpost,  .dsv41_exl3_b3_pair_k3_5120, .dsv41_exl3_b3_guone_k3_2304,
    .dsv41_exl3_b3_mul1h_k2_2304,     .dsv41_exl3_b3_mul1h_k2_5120,  .dsv41_exl3_b3_pair_k2_5120, .dsv41_exl3_b3_guone_k2_2304,
};
pub const checked_at_accept = blk: {
    var out: [kernels.len - w1_texts.len - lut_texts.len - form_texts.len - banked_texts.len]Kernel = undefined;
    var n: usize = 0;
    for (kernels) |k| if (std.mem.indexOfScalar(Kernel, &(w1_texts ++ lut_texts ++ form_texts ++ banked_texts), k) == null) {
        out[n] = k;
        n += 1;
    };
    break :blk out;
};

pub fn accept(comptime G: type, a: Allocator, g: *G, ctx: quant.Context, spec: quant.Spec, diag: *Diag) !*Accepted(G) {
    const ref = ctx.kernels orelse return quant.refuse(diag, error.NoKernelSet, "exl3 quant: accepted without the load context's kernel set", .{});
    const set = ks.Set.of(ref) orelse return quant.refuse(diag, error.NoKernelSet, "exl3 quant: accepted with a kernel set over another registry (manifest {s})", .{ref.manifest_sha256});
    try checkSpec(spec, diag);
    const acc = try a.create(Accepted(G));
    acc.* = .{ .a = a, .reg = &set.reg, .gemv = undefined, .prep = undefined, .tok = undefined };
    errdefer {
        acc.report.deinit(a);
        a.destroy(acc);
    }
    try set.selfCheck(a, &checked_at_accept, .compile, &acc.report, diag);
    acc.gemv = try Gemv(G).init(g, &set.reg);
    errdefer acc.gemv.deinit(g);
    acc.prep = try RinPrep(G).init(g, &set.reg);
    errdefer acc.prep.deinit(g);
    errdefer for (acc.tok[0..acc.n_tok]) |x| g.release(x);
    var idx: [48]i32 = undefined;
    for (&idx, 0..) |*v, i| v.* = @intCast(i);
    for (1..decode_table_rows + 1) |m| {
        acc.tok[m - 1] = g.keep(try g.hostArray(std.mem.sliceAsBytes(idx[0..m]), &.{@intCast(m)}, .int32));
        acc.n_tok = m;
    }
    const waves = try a.alloc(DigXPrefill(G), spec.n_layers);
    var built: usize = 0;
    errdefer {
        for (waves[0..built]) |*w| w.deinit(g);
        a.free(waves);
    }
    for (waves) |*w| {
        w.* = try DigXPrefill(G).init(a, &set.reg, .tier, null);
        built += 1;
    }
    acc.waves = waves;
    return acc;
}

// ── The routes (moved from exl3_kernel_ops.zig) ──

// ── The expert path over the streamer's slot banks (DSV41_EXL3_BANK = on, PREP = rin) ──

pub const Proj = enum { gate, up, down };

/// One projection's slot-bank arrays, as the streamer's `ProjArrays`: code i16 [rows, in/16,
/// out/16, 48], rout f16 [rows, out], rin f16 [rows, in] (row = slot).
pub fn ProjArrays(comptime T: type) type {
    return struct { code: T, rout: T, rin: T };
}

/// A layer bank's projection arrays are what the kernels read (bind time, once per bank).
pub fn checkBank(comptime G: type, g: *G, reg: *const xk.Registry, proj: Proj, a: ProjArrays(G.T), diag: ?*xk.Diag) Refusal!void {
    var vars: Vars = .initFill(0);
    const cap = rowsOf(G, g, a.code, 0);
    const bound = reg.get(.dsv41_exl3_mul1h_k3_2304).bounds.get(.cap).?;
    if (cap < bound[0] or cap > bound[1]) return refuse(diag, error.RouteInput, "exl3 kernel ops: a bank of {d} slots (the kernels take {d}..{d})", .{ cap, bound[0], bound[1] });
    vars.set(.cap, cap);
    switch (proj) {
        .gate, .up => {
            try expectInput(G, g, reg.get(if (kIndexOf(G, g, a.code) == 1) .dsv41_exl3_mul1h_k2_2304 else .dsv41_exl3_mul1h_k3_2304), "code", a.code, &vars, diag);
            try expectInput(G, g, reg.get(.q3_exl3_prep_gu_epi), "rg", a.rout, &vars, diag);
            try expectInput(G, g, reg.get(.q3_exl3_prep_in_rin), "rg", a.rin, &vars, diag);
        },
        .down => {
            try expectInput(G, g, reg.get(if (kIndexOf(G, g, a.code) == 1) .dsv41_exl3_mul1h_k2_5120 else .dsv41_exl3_mul1h_k3_5120), "code", a.code, &vars, diag);
            try expectInput(G, g, reg.get(.q3_moeprep_dpost), "rd", a.rout, &vars, diag);
            try expectInput(G, g, reg.get(.q3_exl3_prep_din_rin), "rn", a.rin, &vars, diag);
        },
    }
}

/// The decode GEMVs (form mul1h, K 3): `Provider.project`.
pub fn Gemv(comptime G: type) type {
    return struct {
        const Self = @This();
        gu: *const Entry,
        dn: *const Entry,
        gu_statics: Statics(G),
        dn_statics: Statics(G),
        gu_p: RowPlans(G, 48),
        dn_p: RowPlans(G, 48),
        /// gu_one: gate and up in one launch (`dsv41_exl3_guone_k3_2304`), bound by `initForms`.
        gu1_p: ?RowPlans(G, 48) = null,
        /// The K = 2 texts (the same forms), for the layers whose code tiles are 32 int16 wide.
        gu2: *const Entry,
        dn2: *const Entry,
        gu2_statics: Statics(G),
        dn2_statics: Statics(G),
        gu2_p: RowPlans(G, 48),
        dn2_p: RowPlans(G, 48),
        gu12_p: ?RowPlans(G, 48) = null,
        /// The gate / up call bound at construction: `guOne` (gu_one's one launch) or `guPair` (the two stock launches).
        gu_call: *const GuCall = guPair,

        const GuCall = fn (self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, code_g: G.T, code_u: G.T) anyerror![2]G.T;

        pub fn init(g: *G, reg: *const xk.Registry) !Self {
            return initForms(g, reg, .{});
        }

        /// The GEMVs under the routed forms (stock when none): down_pair's text for the down projection, gu_one's one
        /// launch for gate and up. Every form computes the stock words (registry twin checks; kbench v9).
        pub fn initForms(g: *G, reg: *const xk.Registry, forms: Forms) !Self {
            const gu = reg.get(.dsv41_exl3_mul1h_k3_2304);
            const dn = reg.get(if (forms.down_pair) .dsv41_exl3_pair_k3_5120 else .dsv41_exl3_mul1h_k3_5120);
            const gu2 = reg.get(.dsv41_exl3_mul1h_k2_2304);
            const dn2 = reg.get(if (forms.down_pair) .dsv41_exl3_pair_k2_5120 else .dsv41_exl3_mul1h_k2_5120);
            var gu_p: RowPlans(G, 48) = try .init(g, gu, null, null);
            errdefer gu_p.deinit(g);
            var dn_p: RowPlans(G, 48) = try .init(g, dn, null, null);
            errdefer dn_p.deinit(g);
            var gu2_p: RowPlans(G, 48) = try .init(g, gu2, null, null);
            errdefer gu2_p.deinit(g);
            var dn2_p: RowPlans(G, 48) = try .init(g, dn2, null, null);
            errdefer dn2_p.deinit(g);
            var gu1_p: ?RowPlans(G, 48) = if (forms.gu_one) try RowPlans(G, 48).init(g, reg.get(.dsv41_exl3_guone_k3_2304), null, null) else null;
            errdefer if (gu1_p) |*x| x.deinit(g);
            var gu12_p: ?RowPlans(G, 48) = if (forms.gu_one) try RowPlans(G, 48).init(g, reg.get(.dsv41_exl3_guone_k2_2304), null, null) else null;
            errdefer if (gu12_p) |*x| x.deinit(g);
            var gs = try Statics(G).init(g, gu);
            errdefer gs.deinit(g);
            var ds = try Statics(G).init(g, dn);
            errdefer ds.deinit(g);
            var gs2 = try Statics(G).init(g, gu2);
            errdefer gs2.deinit(g);
            return .{ .gu = gu, .dn = dn, .gu_statics = gs, .dn_statics = ds, .gu_p = gu_p, .dn_p = dn_p, .gu1_p = gu1_p, .gu2 = gu2, .dn2 = dn2, .gu2_statics = gs2, .dn2_statics = try Statics(G).init(g, dn2), .gu2_p = gu2_p, .dn2_p = dn2_p, .gu12_p = gu12_p, .gu_call = if (forms.gu_one) guOne else guPair };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.gu_p.deinit(g);
            self.dn_p.deinit(g);
            if (self.gu1_p) |*x| x.deinit(g);
            self.gu_statics.deinit(g);
            self.dn_statics.deinit(g);
            self.gu2_p.deinit(g);
            self.dn2_p.deinit(g);
            if (self.gu12_p) |*x| x.deinit(g);
            self.gu2_statics.deinit(g);
            self.dn2_statics.deinit(g);
        }

        /// Gate and up through the call `initForms` bound (no per-call choice).
        pub fn projectGu(self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, code_g: G.T, code_u: G.T) ![2]G.T {
            return self.gu_call(self, g, xg, xu, ids, code_g, code_u);
        }

        fn guPair(self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, code_g: G.T, code_u: G.T) anyerror![2]G.T {
            return .{ try self.project(g, .gate, xg, ids, code_g), try self.project(g, .up, xu, ids, code_u) };
        }

        fn guOne(self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, code_g: G.T, code_u: G.T) anyerror![2]G.T {
            var out: [2]G.T = undefined;
            const p = if (kIndexOf(G, g, code_g) == 1) &self.gu12_p.? else &self.gu1_p.?;
            try p.launch(g, rowsOf(G, g, xg, 0), &.{ xg, xu, ids, code_g, code_u }, &out);
            return out;
        }

        /// xh [rows, in] f32 (rotated), ids [rows] u32 (each row's slot), code = the projection's
        /// bank code -> z [rows, out] f32 (gate / up: out 2304, down: out 5120).
        pub fn project(self: *const Self, g: *G, proj: Proj, xh: G.T, ids: G.T, code: G.T) !G.T {
            const k2 = kIndexOf(G, g, code) == 1;
            const e, const st, const p = if (proj == .down)
                (if (k2) .{ self.dn2, &self.dn2_statics, &self.dn2_p } else .{ self.dn, &self.dn_statics, &self.dn_p })
            else
                (if (k2) .{ self.gu2, &self.gu2_statics, &self.gu2_p } else .{ self.gu, &self.gu_statics, &self.gu_p });
            var ins: [9]G.T = undefined;
            ins[0] = xh;
            ins[1] = ids;
            ins[2] = code;
            for (3..e.inputs.len) |i| ins[i] = st.arrays[i];
            var out: [1]G.T = undefined;
            try p.launch(g, rowsOf(G, g, xh, 0), ins[0..e.inputs.len], &out);
            return out[0];
        }
    };
}

/// The banked route: the decode stages on the banked texts (`banked_texts`), each launch over a wave's rows of all three
/// banks (a row's bank is its packed id's top byte: bank << 24 | slot row; banks base, ext, transient). The forms are
/// the installed GEMVs' (`Gemv.initForms`), bound here at construction: gu_one's banked one launch or the banked
/// mul1h pair, down_pair's banked text or the banked mul1h. The GEMV statics are the installed GEMVs' arrays (aliased by
/// input name, checked once here): the route keeps no device array of its own.
pub fn Banked(comptime G: type) type {
    return struct {
        const Self = @This();
        const A = ProjArrays(G.T);
        const BA = quant.BankArrays(A);
        in_rin_p: RowPlans(G, 48),
        gu_epi_p: RowPlans(G, 48),
        din_rin_p: RowPlans(G, 48),
        dpost_p: RowPlans(G, 48),
        /// gate / up: the banked mul1h (twice) or gu_one's banked one launch
        gu_p: RowPlans(G, 48),
        dn_p: RowPlans(G, 48),
        gu_e: *const Entry,
        dn_e: *const Entry,
        /// the GEMVs' statics, aliased (the installed `Gemv`'s, released by it)
        gu_st: [xk.max_inputs]G.T = undefined,
        dn_st: [xk.max_inputs]G.T = undefined,
        /// the K = 2 texts (the same forms) and their aliased statics
        gu2_p: RowPlans(G, 48),
        dn2_p: RowPlans(G, 48),
        gu2_e: *const Entry,
        dn2_e: *const Entry,
        gu2_st: [xk.max_inputs]G.T = undefined,
        dn2_st: [xk.max_inputs]G.T = undefined,
        gu_call: *const GuCall,

        const GuCall = fn (self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, banks: *const [3]BA) anyerror![2]G.T;

        pub fn init(g: *G, reg: *const xk.Registry, forms: Forms, gemv: *const Gemv(G)) !Self {
            const gu_e = reg.get(if (forms.gu_one) .dsv41_exl3_b3_guone_k3_2304 else .dsv41_exl3_b3_mul1h_k3_2304);
            const dn_e = reg.get(if (forms.down_pair) .dsv41_exl3_b3_pair_k3_5120 else .dsv41_exl3_b3_mul1h_k3_5120);
            const gu2_e = reg.get(if (forms.gu_one) .dsv41_exl3_b3_guone_k2_2304 else .dsv41_exl3_b3_mul1h_k2_2304);
            const dn2_e = reg.get(if (forms.down_pair) .dsv41_exl3_b3_pair_k2_5120 else .dsv41_exl3_b3_mul1h_k2_5120);
            var self: Self = .{ .in_rin_p = undefined, .gu_epi_p = undefined, .din_rin_p = undefined, .dpost_p = undefined, .gu_p = undefined, .dn_p = undefined, .gu_e = gu_e, .dn_e = dn_e, .gu2_p = undefined, .dn2_p = undefined, .gu2_e = gu2_e, .dn2_e = dn2_e, .gu_call = if (forms.gu_one) guOne else guPair };
            // The statics: the stock GEMV's by name (gu_one has none; its gate / up statics are compiled in).
            if (!forms.gu_one) try alias(gu_e, gemv.gu, &gemv.gu_statics, &self.gu_st);
            try alias(dn_e, gemv.dn, &gemv.dn_statics, &self.dn_st);
            if (!forms.gu_one) try alias(gu2_e, gemv.gu2, &gemv.gu2_statics, &self.gu2_st);
            try alias(dn2_e, gemv.dn2, &gemv.dn2_statics, &self.dn2_st);
            self.in_rin_p = try .init(g, reg.get(.dsv41_exl3_b3_prep_in_rin), null, null);
            errdefer self.in_rin_p.deinit(g);
            self.gu_epi_p = try .init(g, reg.get(.dsv41_exl3_b3_prep_gu_epi), null, null);
            errdefer self.gu_epi_p.deinit(g);
            self.din_rin_p = try .init(g, reg.get(.dsv41_exl3_b3_prep_din_rin), null, null);
            errdefer self.din_rin_p.deinit(g);
            self.dpost_p = try .init(g, reg.get(.dsv41_exl3_b3_moeprep_dpost), null, null);
            errdefer self.dpost_p.deinit(g);
            self.gu_p = try .init(g, gu_e, null, null);
            errdefer self.gu_p.deinit(g);
            self.dn_p = try .init(g, dn_e, null, null);
            errdefer self.dn_p.deinit(g);
            self.gu2_p = try .init(g, gu2_e, null, null);
            errdefer self.gu2_p.deinit(g);
            self.dn2_p = try .init(g, dn2_e, null, null);
            return self;
        }

        /// Every static input of the banked text `be` is the stock text `se`'s input of the same name, dtype and values
        /// (the manifest's banked entries are deep copies): alias the stock GEMV's array.
        fn alias(be: *const Entry, se: *const Entry, st: *const Statics(G), out: *[xk.max_inputs]G.T) !void {
            for (be.inputs, 0..) |*arg, i| {
                if (arg.role != .static) continue;
                const j = for (se.inputs, 0..) |*sa, jj| {
                    if (std.mem.eql(u8, sa.name, arg.name)) break jj;
                } else return error.BankedStaticUnmatched;
                const sa = &se.inputs[j];
                if (sa.role != .static or sa.dtype != arg.dtype or st.mask & (@as(u32, 1) << @intCast(j)) == 0) return error.BankedStaticUnmatched;
                var b0: [1024]u8 = undefined;
                var b1: [1024]u8 = undefined;
                const sh0, const v0 = kr.staticBytes(arg, &b0);
                const sh1, const v1 = kr.staticBytes(sa, &b1);
                if (!std.mem.eql(c_int, sh0.slice(), sh1.slice()) or !std.mem.eql(u8, v0, v1)) return error.BankedStaticUnmatched;
                out[i] = st.arrays[j];
            }
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.in_rin_p.deinit(g);
            self.gu_epi_p.deinit(g);
            self.din_rin_p.deinit(g);
            self.dpost_p.deinit(g);
            self.gu_p.deinit(g);
            self.dn_p.deinit(g);
            self.gu2_p.deinit(g);
            self.dn2_p.deinit(g);
        }

        /// x [tokens, 5120] bf16 (taken rows), tok [rows] i32, ids packed -> the clamped SwiGLU [rows, 2304] f32.
        pub fn gateUp(self: *const Self, g: *G, x: G.T, tok: G.T, ids: G.T, banks: *const [3]BA) !G.T {
            const rows = rowsOf(G, g, tok, 0);
            var xs: [2]G.T = undefined;
            try self.in_rin_p.launch(g, rows, &.{ x, tok, banks[0].gate.rin, banks[1].gate.rin, banks[2].gate.rin, banks[0].up.rin, banks[1].up.rin, banks[2].up.rin, ids }, &xs);
            const z = try self.gu_call(self, g, xs[0], xs[1], ids, banks);
            var out: [1]G.T = undefined;
            try self.gu_epi_p.launch(g, rows, &.{ z[0], z[1], banks[0].gate.rout, banks[1].gate.rout, banks[2].gate.rout, banks[0].up.rout, banks[1].up.rout, banks[2].up.rout, ids }, &out);
            return out[0];
        }

        /// h [rows, 2304] f32, ids packed -> [rows, 5120] f32: din_rin -> the down GEMV -> dpost, banked.
        pub fn down(self: *const Self, g: *G, h: G.T, ids: G.T, banks: *const [3]BA) !G.T {
            const rows = rowsOf(G, g, h, 0);
            var hd: [1]G.T = undefined;
            try self.din_rin_p.launch(g, rows, &.{ h, banks[0].down.rin, banks[1].down.rin, banks[2].down.rin, ids }, &hd);
            const k2 = kIndexOf(G, g, banks[0].down.code) == 1;
            var ins: [xk.max_inputs]G.T = if (k2) self.dn2_st else self.dn_st;
            ins[0] = hd[0];
            ins[1] = ids;
            ins[2] = banks[0].down.code;
            ins[3] = banks[1].down.code;
            ins[4] = banks[2].down.code;
            var zd: [1]G.T = undefined;
            const dn_e, const dn_p = if (k2) .{ self.dn2_e, &self.dn2_p } else .{ self.dn_e, &self.dn_p };
            try dn_p.launch(g, rows, ins[0..dn_e.inputs.len], &zd);
            var out: [1]G.T = undefined;
            try self.dpost_p.launch(g, rows, &.{ zd[0], banks[0].down.rout, banks[1].down.rout, banks[2].down.rout, ids }, &out);
            return out[0];
        }

        fn guPair(self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, banks: *const [3]BA) anyerror![2]G.T {
            const rows = rowsOf(G, g, xg, 0);
            var out: [2]G.T = undefined;
            const k2 = kIndexOf(G, g, banks[0].gate.code) == 1;
            const gu_e, const gu_p = if (k2) .{ self.gu2_e, &self.gu2_p } else .{ self.gu_e, &self.gu_p };
            var ins: [xk.max_inputs]G.T = if (k2) self.gu2_st else self.gu_st;
            ins[1] = ids;
            inline for (.{ .gate, .up }, 0..) |proj, o| {
                ins[0] = if (o == 0) xg else xu;
                ins[2] = @field(banks[0], @tagName(proj)).code;
                ins[3] = @field(banks[1], @tagName(proj)).code;
                ins[4] = @field(banks[2], @tagName(proj)).code;
                var z: [1]G.T = undefined;
                try gu_p.launch(g, rows, ins[0..gu_e.inputs.len], &z);
                out[o] = z[0];
            }
            return out;
        }

        fn guOne(self: *const Self, g: *G, xg: G.T, xu: G.T, ids: G.T, banks: *const [3]BA) anyerror![2]G.T {
            var out: [2]G.T = undefined;
            const p = if (kIndexOf(G, g, banks[0].gate.code) == 1) &self.gu2_p else &self.gu_p;
            try p.launch(g, rowsOf(G, g, xg, 0), &.{ xg, xu, ids, banks[0].gate.code, banks[1].gate.code, banks[2].gate.code, banks[0].up.code, banks[1].up.code, banks[2].up.code }, &out);
            return out;
        }
    };
}

/// PREP = rin: the rin / rout stages around the GEMVs (`build_kernels` of the rinprep lane).
pub fn RinPrep(comptime G: type) type {
    return struct {
        const Self = @This();
        in_rin_e: *const Entry,
        gu_epi_e: *const Entry,
        din_rin_e: *const Entry,
        dpost_e: *const Entry,
        in_rin_p: RowPlans(G, 48),
        gu_epi_p: RowPlans(G, 48),
        din_rin_p: RowPlans(G, 48),
        dpost_p: RowPlans(G, 48),

        pub fn init(g: *G, reg: *const xk.Registry) !Self {
            const a, const b, const c, const d = .{ reg.get(.q3_exl3_prep_in_rin), reg.get(.q3_exl3_prep_gu_epi), reg.get(.q3_exl3_prep_din_rin), reg.get(.q3_moeprep_dpost) };
            var pa: RowPlans(G, 48) = try .init(g, a, null, null);
            errdefer pa.deinit(g);
            var pb: RowPlans(G, 48) = try .init(g, b, null, null);
            errdefer pb.deinit(g);
            var pc: RowPlans(G, 48) = try .init(g, c, null, null);
            errdefer pc.deinit(g);
            return .{ .in_rin_e = a, .gu_epi_e = b, .din_rin_e = c, .dpost_e = d, .in_rin_p = pa, .gu_epi_p = pb, .din_rin_p = pc, .dpost_p = try .init(g, d, null, null) };
        }

        pub fn deinit(self: *Self, g: *G) void {
            self.in_rin_p.deinit(g);
            self.gu_epi_p.deinit(g);
            self.din_rin_p.deinit(g);
            self.dpost_p.deinit(g);
        }

        /// x [tokens, 5120] bf16, tok [rows] i32, rg / ru = gate / up rin, ids [rows] u32 ->
        /// (xg, xu) [rows, 5120] f32 = t128(x[tok] * rin[ids]).
        pub fn inRin(self: *const Self, g: *G, x: G.T, tok: G.T, rg: G.T, ru: G.T, ids: G.T) ![2]G.T {
            var out: [2]G.T = undefined;
            try self.in_rin_p.launch(g, rowsOf(G, g, tok, 0), &.{ x, tok, rg, ru, ids }, &out);
            return out;
        }

        /// zg / zu [rows, 2304] f32, rg / ru = gate / up rout -> clamped SwiGLU [rows, 2304] f32.
        pub fn guEpi(self: *const Self, g: *G, zg: G.T, zu: G.T, rg: G.T, ru: G.T, ids: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.gu_epi_p.launch(g, rowsOf(G, g, zg, 0), &.{ zg, zu, rg, ru, ids }, &out);
            return out[0];
        }

        /// hid [rows, 2304] f32, rn = down rin -> t128(hid * rin[ids]) [rows, 2304] f32.
        pub fn dinRin(self: *const Self, g: *G, hid: G.T, rn: G.T, ids: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.din_rin_p.launch(g, rowsOf(G, g, hid, 0), &.{ hid, rn, ids }, &out);
            return out[0];
        }

        /// zd [rows, 5120] f32, rd = down rout -> t128(zd) * rout[ids] [rows, 5120] f32.
        pub fn dpost(self: *const Self, g: *G, zd: G.T, rd: G.T, ids: G.T) !G.T {
            var out: [1]G.T = undefined;
            try self.dpost_p.launch(g, rowsOf(G, g, zd, 0), &.{ zd, rd, ids }, &out);
            return out[0];
        }
    };
}

// ── Prefill: REBUILD (DSV41_PREFILL_FUSED_BANK = exl3, mul1lut) and DIG-X (DIG = 1, DIG2 = onepass) ──

pub const wave_max = 16;

/// The rebuild's slot table: the wave's slots, padded with the first (the lane's `slots8`).
pub fn rebuildSlots(slots: []const u32) [wave_max]i32 {
    var t: [wave_max]i32 = undefined;
    for (&t, 0..) |*v, i| v.* = @intCast(slots[if (i < slots.len) i else 0]);
    return t;
}

/// The EXL3 rebuild of a wave's experts into bf16 weights: `Exl3Rebuild3Kernel.launch`.
pub fn Rebuild(comptime G: type) type {
    return struct {
        const Self = @This();
        e: *const Entry,

        pub fn init(reg: *const xk.Registry) Self {
            return .{ .e = reg.get(.q3_prefill_fused_exl3x3_mul1lut_k3_bf16) };
        }

        /// slots i32 [16] (`rebuildSlots`), `experts` of them used -> (og, ou [n, 5120, 2304], od [n, 2304, 5120]) bf16.
        pub fn call(self: *const Self, g: *G, gate: ProjArrays(G.T), up: ProjArrays(G.T), down: ProjArrays(G.T), slots: G.T, experts: u32) ![3]G.T {
            var vars: Vars = .initFill(0);
            vars.set(.experts, experts);
            var out: [3]G.T = undefined;
            try launchRule(G, g, self.e, &vars, &.{ gate.code, gate.rout, gate.rin, up.code, up.rout, up.rin, down.code, down.rout, down.rin, slots }, &out);
            return out;
        }
    };
}

pub const WaveExpert = struct { slot: u32, rows: u32 };
pub const DigTable = struct { table: [80]i32, tgs: u32 };

/// q3_prefill_dig_candidate.wave_table: per expert (<= 16, rows grouped by expert in order)
/// slot, first row, rows, first threadgroup (unused: INT32_MAX); [64] experts, [65] threadgroups.
/// `tiles` = the GEMM's threadgroups per `bm`-row M tile (`digTiles`; `bm` 64, or 128 for the 128-row GEMM texts):
/// the first-threadgroup column counts ceil(rows / bm) tiles per expert.
pub fn digTableBm(experts: []const WaveExpert, tiles: u32, bm: u32) DigTable {
    var t: [80]i32 = @splat(0);
    var row0: i64 = 0;
    var tg: i64 = 0;
    for (0..wave_max) |j| {
        if (j >= experts.len) {
            t[48 + j] = std.math.maxInt(i32);
            continue;
        }
        t[j] = @intCast(experts[j].slot);
        t[16 + j] = @intCast(row0);
        t[32 + j] = @intCast(experts[j].rows);
        t[48 + j] = @intCast(tg);
        row0 += experts[j].rows;
        tg += @as(i64, @intCast((experts[j].rows + bm - 1) / bm)) * tiles;
    }
    t[64] = @intCast(experts.len);
    t[65] = @intCast(tg);
    return .{ .table = t, .tgs = @intCast(tg) };
}

/// DIG-X on DIG2 onepass: the NAX GEMMs whose B loader decodes the trellis, the rotation
/// stages and the onepass SwiGLU (`DigGemmX`, `RotKernelsX`, `Dig2OnePassX`).
pub fn DigX(comptime G: type) type {
    return struct {
        const Self = @This();
        /// the 64-row GEMM texts: the 128-row texts' twin reference (their device self-check, the 0b smoke), not
        /// launched by the route
        gemm_gu: *const Entry,
        gemm_dn: *const Entry,
        /// the GEMMs the route launches: 128-row M tiles (one decoded B stage feeds 128 rows); the down text and
        /// `widen1_e` are also the fused down GEMM's reference (its device self-check `fused`, the 0b smoke)
        gemm_gu128: *const Entry,
        gemm_dn128: *const Entry,
        /// the lane's take2 text: the retune's bitwise reference (its 0b smoke), not launched by the route
        take2_e: *const Entry,
        /// the take2 retune the route launches (both outputs per simdgroup from one act load)
        take2v_e: *const Entry,
        roundx_e: *const Entry,
        onepass_e: *const Entry,
        widen2_e: *const Entry,
        widen1_e: *const Entry,
        /// the fused arm's down stage (`Accepted.routeFusedDown`): the 128-row down text at BN 128 with rot_widen1 as
        /// its epilogue
        gemm_dn_w1: *const Entry,
        /// the 128-row gate|up text with the mul1h codebook through a threadgroup table: registered, not launched by
        /// the route
        gemm_gu128lut: *const Entry,
        /// the K = 2 twins of the launched GEMMs (layers whose code tiles are 32 int16 wide)
        gemm_gu128_k2: *const Entry,
        gemm_dn128_k2: *const Entry,
        gemm_dn_w1_k2: *const Entry,

        pub fn init(reg: *const xk.Registry) Self {
            return .{
                .gemm_gu = reg.get(.q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3),
                .gemm_dn = reg.get(.q3_prefill_dig_gemm_2304x5120_xmul1hk3),
                .gemm_gu128 = reg.get(.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128),
                .gemm_dn128 = reg.get(.dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128),
                .take2_e = reg.get(.q3_prefill_dig_rot_take2_5120),
                .take2v_e = reg.get(.dsv41_prefill_dig_take2v_5120),
                .roundx_e = reg.get(.q3_prefill_dig_rot_roundx_2304),
                .onepass_e = reg.get(.q3_prefill_dig2_swiglu_2304_x),
                .widen2_e = reg.get(.q3_prefill_dig_rot_widen2_2304),
                .widen1_e = reg.get(.q3_prefill_dig_rot_widen1_5120),
                .gemm_dn_w1 = reg.get(.dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1),
                .gemm_gu128lut = reg.get(.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut),
                .gemm_gu128_k2 = reg.get(.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128),
                .gemm_dn128_k2 = reg.get(.dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128),
                .gemm_dn_w1_k2 = reg.get(.dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128w1),
            };
        }

        /// The routed GEMMs' M tile rows: their wave tables count ceil(rows / m_tile) tiles per expert (`digTableBm`).
        pub const m_tile: u32 = 128;

        /// The GEMMs' threadgroups per M tile: gate|up (both operands), down (the same at 64 and 128 rows).
        pub fn digTiles(self: *const Self, proj: enum { gate_up, down }) u32 {
            const e = if (proj == .down) self.gemm_dn else self.gemm_gu;
            return argOf(e, "tbl").domain.tiles;
        }

        /// x0 / x1 [rows, 1, 5120] f16 (take2), gate / up code, the gate|up table at `m_tile` rows -> (zg, zu)
        /// [rows, 2304] f32: the 128-row text, the 64-row text's words.
        pub fn gemmGateUp(self: *const Self, g: *G, x0: G.T, x1: G.T, code_g: G.T, code_u: G.T, tbl: G.T, tgs: u32) ![2]G.T {
            var vars = rowsVars(rowsOf(G, g, x0, 0));
            vars.set(.tgs, tgs);
            var out: [2]G.T = undefined;
            try launchRule(G, g, if (kIndexOf(G, g, code_g) == 1) self.gemm_gu128_k2 else self.gemm_gu128, &vars, &.{ x0, x1, code_g, code_u, tbl }, &out);
            return out;
        }

        /// `gemmGateUp` through the table-codebook text: the same inputs and words.
        pub fn gemmGateUpLut(self: *const Self, g: *G, x0: G.T, x1: G.T, code_g: G.T, code_u: G.T, tbl: G.T, tgs: u32) ![2]G.T {
            var vars = rowsVars(rowsOf(G, g, x0, 0));
            vars.set(.tgs, tgs);
            var out: [2]G.T = undefined;
            try launchRule(G, g, self.gemm_gu128lut, &vars, &.{ x0, x1, code_g, code_u, tbl }, &out);
            return out;
        }

        /// x [rows, 1, 2304] f16 (onepass), down code, down rout, its table (`widenTiles` N tiles at `m_tile` rows) ->
        /// widen1(z) [rows, 5120] f32 in one launch: the words of `gemmDown`, then `widen1`.
        pub fn gemmDownWiden(self: *const Self, g: *G, x: G.T, code_d: G.T, rout_d: G.T, tbl: G.T, tgs: u32) !G.T {
            var vars = rowsVars(rowsOf(G, g, x, 0));
            vars.set(.tgs, tgs);
            var out: [1]G.T = undefined;
            try launchRule(G, g, if (kIndexOf(G, g, code_d) == 1) self.gemm_dn_w1_k2 else self.gemm_dn_w1, &vars, &.{ x, code_d, rout_d, tbl }, &out);
            return out[0];
        }

        /// The fused down GEMM's threadgroups per M tile (its BN 128 column tiles).
        pub fn widenTiles(self: *const Self) u32 {
            return argOf(self.gemm_dn_w1, "tbl").domain.tiles;
        }

        /// x [rows, 1, 2304] f16 (onepass), down code, the down table at `m_tile` rows -> z [rows, 5120] f32: the
        /// 128-row text, the 64-row text's words.
        pub fn gemmDown(self: *const Self, g: *G, x: G.T, code_d: G.T, tbl: G.T, tgs: u32) !G.T {
            var vars = rowsVars(rowsOf(G, g, x, 0));
            vars.set(.tgs, tgs);
            var out: [1]G.T = undefined;
            try launchRule(G, g, if (kIndexOf(G, g, code_d) == 1) self.gemm_dn128_k2 else self.gemm_dn128, &vars, &.{ x, code_d, tbl }, &out);
            return out[0];
        }

        /// act [A, 5120] bf16 rows ridx [rows] i32, rhs [rows] u32 (expert per row), slots = a
        /// table -> (f16(t128(act * rin_g[slot])), f16(t128(act * rin_u[slot]))) [rows, 1, 5120]. The retune
        /// (`dsv41_prefill_dig_take2v_5120`): one simdgroup per (row, 128-block) writes both outputs from one act
        /// load, the lane text's butterflies in its bit order, so its words are the lane's (the device self-check
        /// `mlx_chain`, bitwise; the 0b smoke against the lane text on real records).
        pub fn take2(self: *const Self, g: *G, act: G.T, ridx: G.T, rhs: G.T, slots: G.T, rin_g: G.T, rin_u: G.T) ![2]G.T {
            const vars = rowsVars(rowsOf(G, g, ridx, 0));
            var out: [2]G.T = undefined;
            try launchRule(G, g, self.take2v_e, &vars, &.{ act, ridx, rhs, slots, rin_g, rin_u }, &out);
            return out;
        }

        /// act [rows, 1, 2304] f32 -> f16(t128(act * rin_d[slot])) [rows, 1, 2304].
        pub fn roundx(self: *const Self, g: *G, act: G.T, rhs: G.T, slots: G.T, rin_d: G.T) !G.T {
            const vars = rowsVars(rowsOf(G, g, act, 0));
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.roundx_e, &vars, &.{ act, rhs, slots, rin_d }, &out);
            return out[0];
        }

        /// zg / zu [rows, 2304] f32 -> hd = f16(t128(clamped SwiGLU(t128(zg) rout_g, t128(zu) rout_u) rin_d)) [rows, 1, 2304].
        pub fn onePass(self: *const Self, g: *G, zg: G.T, zu: G.T, rhs: G.T, tbl: G.T, rout_g: G.T, rout_u: G.T, rin_d: G.T) !G.T {
            const vars = rowsVars(rowsOf(G, g, zg, 0));
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.onepass_e, &vars, &.{ zg, zu, rhs, tbl, rout_g, rout_u, rin_d }, &out);
            return out[0];
        }

        /// act_g / act_u [rows, 2304] f32 -> t128(act) * rout[slot] [rows, 1, 2304] f32 each.
        pub fn widen2(self: *const Self, g: *G, act_g: G.T, act_u: G.T, rhs: G.T, slots: G.T, rout_g: G.T, rout_u: G.T) ![2]G.T {
            const vars = rowsVars(rowsOf(G, g, act_g, 0));
            var out: [2]G.T = undefined;
            try launchRule(G, g, self.widen2_e, &vars, &.{ act_g, act_u, rhs, slots, rout_g, rout_u }, &out);
            return out;
        }

        /// act [rows, 5120] f32 -> t128(act) * rout_d[slot] [rows, 1, 5120] f32.
        pub fn widen1(self: *const Self, g: *G, act: G.T, rhs: G.T, slots: G.T, rout_d: G.T) !G.T {
            const vars = rowsVars(rowsOf(G, g, act, 0));
            var out: [1]G.T = undefined;
            try launchRule(G, g, self.widen1_e, &vars, &.{ act, rhs, slots, rout_d }, &out);
            return out[0];
        }

        /// `Dig2GemmOnePassX`: take2 -> the gate|up GEMM -> onepass: (hd, zg, zu).
        pub fn gateUpOnePass(self: *const Self, g: *G, act: G.T, ridx: G.T, rhs: G.T, gu: DigTableArray(G), gate: ProjArrays(G.T), up: ProjArrays(G.T), down_rin: G.T) ![3]G.T {
            const x = try self.take2(g, act, ridx, rhs, gu.tbl, gate.rin, up.rin);
            const z = try self.gemmGateUp(g, x[0], x[1], gate.code, up.code, gu.tbl, gu.tgs);
            const hd = try self.onePass(g, z[0], z[1], rhs, gu.tbl, gate.rout, up.rout, down_rin);
            return .{ hd, z[0], z[1] };
        }
    };
}

/// A wave table on the device and its threadgroup count (`digTableBm`).
pub fn DigTableArray(comptime G: type) type {
    return struct { tbl: G.T, tgs: u32 };
}

// ── Prefill: the routed wave dispatch (DIG-X, DIG2 onepass, SHAPE balance / rebuildahead / carry) ──

/// The installed prefill wave shape: the fused point's wave / inflight / row budget
/// (Q3_PREFILL_FUSED_INSTALL config) and SHAPE's carry rows (Q3_PREFILL_SHAPE_INSTALL).
pub const PrefillShape = struct {
    /// experts per wave (1..16, the wave table's expert rows)
    wave: u32,
    /// waves in flight (>= 2, SHAPE's overlap route): a wave first waits for all but `inflight - 1` older ones
    inflight: u32,
    /// assignment rows per wave; an expert above it forms a wave alone (solo: drained before, evaluated after)
    row_budget: u32,
    /// a call of at most `carry_rows` rows leaves its last waves in flight (to the next call or `finish`)
    carry_rows: u32,

    /// Record 3 (run 3r-record3-fast-typical-exl3-30-guard-20260928.log): the lane's shape, the one its wave
    /// samples and the move's pinned launches were taken at.
    pub const record3: PrefillShape = .{ .wave = 4, .inflight = 2, .row_budget = 7168, .carry_rows = 8192 };
    /// The served tier: Record 3 with L1, 8 experts per wave. Under K16 a group call's experts carry about 256
    /// rows each (98,304 routed rows over 384 experts), so Record 3's four-expert cap bound, not the row budget:
    /// eight halve the waves, and with them the per-wave host encode (five launches, four host arrays, one async
    /// eval). Exact: a wave's rows are independent of its composition (each 64-row tile reads one expert's rows
    /// and weights; the join restores the assignment order).
    pub const tier: PrefillShape = .{ .wave = 8, .inflight = record3.inflight, .row_budget = record3.row_budget, .carry_rows = record3.carry_rows };
};

/// A layer bank's three projections (the streamer's `BankArrays`).
pub fn BankArrays(comptime T: type) type {
    return quant.BankArrays(ProjArrays(T));
}

/// One call's routed rows: `slot[i]` = assignment row i's bank slot (its binding's bank_index).
/// Row i reads act row i (the switch's `selected`), or act row `act_row[i]` when given (the
/// chunk's tokens and position / top_k: the same words without the switch's row take).
pub const PrefillRows = quant.PrefillRows;

/// The lane of record's `Tcq3FusedPrefillDispatch...__dig2_onepass_exl3.__call__` over one call's
/// routed rows (one route per layer, as the lane has one dispatcher per layer): experts grouped
/// by slot in first-appearance order, snake-ordered (largest, smallest, ...; stable), packed
/// greedily into waves of <= `wave` experts and <= `row_budget` rows; per wave rot_take2 -> the
/// gate|up GEMM (72-tile table) -> dig2 onepass -> the down GEMM (80-tile table) -> rot_widen1
/// (the route: the take2 retune and the 128-row GEMMs; the fused arm launches the fused down GEMM, the
/// 128-row down text with rot_widen1 as its epilogue (40-tile table), in place of the last two: the lane's
/// words either way);
/// then `take(concatenate(waves), argsort(positions))`, the permutation made on the host. The
/// eval schedule is SHAPE's: a wave waits for all but `inflight - 1` older waves (a solo wave for
/// all, and is evaluated at once); a call above `carry_rows` rows drains every wave and evaluates
/// the join and the result, a carried call leaves its last waves in flight and async-evaluates
/// the result; `finish` (the prefill boundary) drains them. The lane's rebuild-ahead is the next
/// wave's two host tables (no GPU work) and has no counterpart here.
///
/// Wave lifecycle (the backend's `mark` / `resetTo`: `resetTo` frees every array tracked since
/// the mark, kept handles survive): each wave's ops run between a mark and a `resetTo` right
/// after the wave is submitted, so its intermediates live only while the GPU still needs them
/// (a pending graph holds its inputs); the wave's output survives as two kept handles (the
/// join's, the in-flight queue's). The join runs between its own mark and `resetTo`; `call`
/// returns a KEPT result (the caller releases it). Nothing else the call builds outlives it.
pub fn DigXPrefill(comptime G: type) type {
    if (!@hasDecl(G, "mark") or !@hasDecl(G, "resetTo"))
        @compileError("exl3 kernel ops: DigXPrefill needs a backend with mark / resetTo (the wave lifecycle)");
    return struct {
        const Self = @This();
        /// G7: the backend's prompt-pass split (the arch injects it; off on every other backend and build).
        const prof = sdk_ext.profile.of(G).prefill;
        const hidden = 5120;
        /// The wave's down stage, installed at construction (`installDown`): the 128-row down text then rot_widen1
        /// (stock), or the fused down GEMM (the same words in one launch). Each builds its own table.
        pub const Down = enum { chain, fused };
        const DownStage = *const fn (self: *const Self, g: *G, hd: G.T, bank: BankArrays(G.T), ex: []const WaveExpert, rhs: G.T) anyerror!G.T;
        dig: DigX(G),
        shape: PrefillShape,
        tiles_gu: u32,
        tiles_dn: u32,
        /// the fused down GEMM's threadgroups per M tile (`DigX.widenTiles`)
        tiles_w: u32,
        rows_hi: u64,
        down: DownStage = downChain,
        /// one wave's launches: the take2 retune, the gate|up GEMM, onepass, then the down stage's (2 stock, 1 fused)
        wave_launches: u32 = 5,
        a: Allocator,
        diag: ?*xk.Diag,
        /// waves in flight, oldest first (kept handles; persist across carried calls)
        flight: std.ArrayList(G.T) = .empty,
        parts: std.ArrayList(G.T) = .empty,
        // host scratch, reused across calls
        group_of: std.ArrayList(i32) = .empty,
        gslot: std.ArrayList(u32) = .empty,
        gcount: std.ArrayList(u32) = .empty,
        gnext: std.ArrayList(u32) = .empty,
        snake: std.ArrayList(u32) = .empty,
        sorted: std.ArrayList(u32) = .empty,
        grows: std.ArrayList(u32) = .empty,
        pos: std.ArrayList(u32) = .empty,
        ridx: std.ArrayList(i32) = .empty,
        rhs: std.ArrayList(u32) = .empty,
        inv: std.ArrayList(u32) = .empty,

        /// `diag` (optional) receives the refusal messages of `init` and of every call.
        pub fn init(a: Allocator, reg: *const xk.Registry, shape: PrefillShape, diag: ?*xk.Diag) Refusal!Self {
            if (shape.wave < 1 or shape.wave > wave_max or shape.inflight < 2 or shape.row_budget < 1)
                return refuse(diag, error.RouteInput, "exl3 kernel ops: prefill shape wave {d} (1..{d}), inflight {d} (>= 2: SHAPE's overlap route), row budget {d} (>= 1)", .{ shape.wave, wave_max, shape.inflight, shape.row_budget });
            const dig = DigX(G).init(reg);
            return .{ .dig = dig, .shape = shape, .tiles_gu = dig.digTiles(.gate_up), .tiles_dn = dig.digTiles(.down), .tiles_w = dig.widenTiles(), .rows_hi = dig.gemm_gu.bounds.get(.rows).?[1], .a = a, .diag = diag };
        }

        /// Releases the waves still in flight (without evaluating them) and the scratch.
        pub fn deinit(self: *Self, g: *G) void {
            for (self.flight.items) |x| g.release(x);
            for (self.parts.items) |x| g.release(x);
            self.flight.deinit(self.a);
            self.parts.deinit(self.a);
            inline for (.{ &self.group_of, &self.gslot, &self.gcount, &self.gnext, &self.snake, &self.sorted, &self.grows, &self.pos, &self.ridx, &self.rhs, &self.inv }) |l| l.deinit(self.a);
        }

        /// The prefill boundary: every wave still in flight evaluated, oldest first.
        pub fn finish(self: *Self, g: *G) !void {
            const t = prof.now();
            while (self.flight.items.len > 0) try self.drainOne(g);
            prof.charge(.drain, t);
        }

        fn drainOne(self: *Self, g: *G) !void {
            const x = self.flight.orderedRemove(0);
            defer g.release(x);
            try g.evalAll(&.{x});
        }

        /// act bf16 [a_rows, 5120], `rows` (A assignment rows), the call's bank -> a KEPT f32
        /// [A, 5120] in assignment-row order (the lane's `result`; release it). The caller's route guarantees A within
        /// 1..the kernels' row bound, every slot inside the bank and the act rows (asserted, not re-checked).
        /// The call's waves (every wave's output KEPT in `parts`, wave order; `pos` maps each wave-ordered
        /// row to its assignment row): true when the call is carried (its last waves left in flight).
        fn runWaves(self: *Self, g: *G, act: G.T, rows: PrefillRows, bank: BankArrays(G.T)) !bool {
            const n_rows = rows.slot.len;
            std.debug.assert(!(n_rows == 0 or n_rows > self.rows_hi));
            const a_rows = rowsOf(G, g, act, 0);
            std.debug.assert(!(rows.act_row == null and a_rows != n_rows));
            if (rows.act_row) |ar| std.debug.assert(ar.len == n_rows);
            var tp = prof.now();
            prof.countCall();
            try self.group(rows, rowsOf(G, g, bank.gate.code, 0), a_rows);
            const carried = n_rows <= self.shape.carry_rows;
            const a = self.a;
            const cnt = self.gcount.items;
            const order = self.snake.items;
            for (self.parts.items) |x| g.release(x);
            self.parts.clearRetainingCapacity();
            var i: usize = 0;
            var off: usize = 0;
            while (i < order.len) {
                const first = i;
                var wave_rows: usize = cnt[order[i]];
                i += 1;
                while (i < order.len and i - first < self.shape.wave and wave_rows + cnt[order[i]] <= self.shape.row_budget) : (i += 1) wave_rows += cnt[order[i]];
                const solo = wave_rows > self.shape.row_budget;
                const keep: usize = if (solo) 0 else self.shape.inflight - 1;
                prof.charge(.encode, tp);
                tp = prof.now();
                while (self.flight.items.len > keep) try self.drainOne(g);
                prof.charge(.drain, tp);
                tp = prof.now();
                prof.count(1, self.wave_launches);
                var ex: [wave_max]WaveExpert = undefined;
                for (order[first..i], 0..) |gi, j| {
                    ex[j] = .{ .slot = self.gslot.items[gi], .rows = cnt[gi] };
                    @memset(self.rhs.items[off..][0..cnt[gi]], @intCast(j));
                    off += cnt[gi];
                }
                const r0 = off - wave_rows;
                try self.parts.ensureUnusedCapacity(a, 1);
                try self.flight.ensureUnusedCapacity(a, 1);
                const m = g.mark();
                const y = try self.submit(g, act, bank, ex[0 .. i - first], self.ridx.items[r0..off], self.rhs.items[r0..off]);
                self.parts.appendAssumeCapacity(g.keep(y));
                if (solo) {
                    prof.charge(.encode, tp);
                    tp = prof.now();
                    try g.evalAll(&.{y});
                    prof.charge(.drain, tp);
                    tp = prof.now();
                } else {
                    try g.asyncEval(&.{y});
                    self.flight.appendAssumeCapacity(g.keep(y));
                }
                g.resetTo(m);
            }
            prof.charge(.encode, tp);
            return carried;
        }

        pub fn call(self: *Self, g: *G, act: G.T, rows: PrefillRows, bank: BankArrays(G.T)) !G.T {
            const n_rows = rows.slot.len;
            const carried = try self.runWaves(g, act, rows, bank);
            var tp = prof.now();
            if (!carried) while (self.flight.items.len > 0) try self.drainOne(g);
            prof.charge(.drain, tp);
            tp = prof.now();
            const m = g.mark();
            const joined = try g.concat(self.parts.items, 0);
            for (self.parts.items) |x| g.release(x);
            self.parts.clearRetainingCapacity();
            prof.charge(.join, tp);
            tp = prof.now();
            if (!carried) try g.evalAll(&.{joined});
            prof.charge(.drain, tp);
            tp = prof.now();
            for (self.pos.items, 0..) |p, j| self.inv.items[p] = @intCast(j);
            const ord = try g.hostArray(std.mem.sliceAsBytes(self.inv.items), &.{@intCast(n_rows)}, .uint32);
            const result = try g.take(joined, ord, 0);
            const kept = g.keep(result);
            errdefer g.release(kept);
            prof.charge(.join, tp);
            tp = prof.now();
            if (carried) try g.asyncEval(&.{result}) else try g.evalAll(&.{result});
            prof.charge(.drain, tp);
            g.resetTo(m);
            return kept;
        }

        /// `call` without its join (the CALLFEED scatter join's counterpart for a caller that reads
        /// unjoined outputs, JOINLESS): the waves' KEPT outputs are appended to `outs` in wave order
        /// (the caller releases them) and, per wave-ordered row, its assignment row to `pos`. The
        /// values are `call`'s: the same wave arrays, no concatenate or take.
        pub fn callParts(self: *Self, g: *G, act: G.T, rows: PrefillRows, bank: BankArrays(G.T), alloc: Allocator, outs: *std.ArrayList(G.T), pos: *std.ArrayList(u32)) !void {
            const carried = try self.runWaves(g, act, rows, bank);
            const tp = prof.now();
            if (!carried) while (self.flight.items.len > 0) try self.drainOne(g);
            prof.charge(.drain, tp);
            try outs.ensureUnusedCapacity(alloc, self.parts.items.len);
            try pos.appendSlice(alloc, self.pos.items);
            outs.appendSliceAssumeCapacity(self.parts.items);
            self.parts.clearRetainingCapacity();
        }

        /// The down stage (`Down`), before any call: the stage's function and the wave's launch count.
        pub fn installDown(self: *Self, d: Down) void {
            switch (d) {
                .chain => {
                    self.down = downChain;
                    self.wave_launches = 5;
                },
                .fused => {
                    self.down = downFused;
                    self.wave_launches = 4;
                },
            }
        }

        /// One wave's launches (`wave_launches`) -> its rows' output f32 [R, 5120] (the lane's `y.reshape(-1, H)`).
        fn submit(self: *Self, g: *G, act: G.T, bank: BankArrays(G.T), ex: []const WaveExpert, ridx: []const i32, rhs: []const u32) !G.T {
            const n: c_int = @intCast(ridx.len);
            // at the routed GEMMs' M tile; take2 and onepass read the gate|up table's slot column only
            const tg = digTableBm(ex, self.tiles_gu, DigX(G).m_tile);
            const tgu = try g.hostArray(std.mem.sliceAsBytes(&tg.table), &.{80}, .int32);
            const ridx_a = try g.hostArray(std.mem.sliceAsBytes(ridx), &.{n}, .int32);
            const rhs_a = try g.hostArray(std.mem.sliceAsBytes(rhs), &.{n}, .uint32);
            const hz = try self.dig.gateUpOnePass(g, act, ridx_a, rhs_a, .{ .tbl = tgu, .tgs = tg.tgs }, bank.gate, bank.up, bank.down.rin);
            const y = try self.down(self, g, hz[0], bank, ex, rhs_a);
            return g.reshape(y, &.{ n, hidden });
        }

        /// Stock: the 128-row down text over its 80-tile table, then rot_widen1 (the table's slot column).
        fn downChain(self: *const Self, g: *G, hd: G.T, bank: BankArrays(G.T), ex: []const WaveExpert, rhs: G.T) anyerror!G.T {
            const td = digTableBm(ex, self.tiles_dn, DigX(G).m_tile);
            const tdn = try g.hostArray(std.mem.sliceAsBytes(&td.table), &.{80}, .int32);
            const zd = try self.dig.gemmDown(g, hd, bank.down.code, tdn, td.tgs);
            return self.dig.widen1(g, zd, rhs, tdn, bank.down.rout);
        }

        /// Fused: the fused down GEMM over its 40-tile table (rot_widen1 as its epilogue reads the table's slot column).
        fn downFused(self: *const Self, g: *G, hd: G.T, bank: BankArrays(G.T), ex: []const WaveExpert, rhs: G.T) anyerror!G.T {
            _ = rhs;
            const tw = digTableBm(ex, self.tiles_w, DigX(G).m_tile);
            const tdw = try g.hostArray(std.mem.sliceAsBytes(&tw.table), &.{80}, .int32);
            return self.dig.gemmDownWiden(g, hd, bank.down.code, bank.down.rout, tdw, tw.tgs);
        }

        /// The host plan: groups (slot -> rows, first-appearance order, rows ascending), the snake
        /// order, and per wave-ordered row its assignment row (`pos`) and act row (`ridx`).
        fn group(self: *Self, rows: PrefillRows, cap: u64, a_rows: u64) !void {
            const a = self.a;
            const n = rows.slot.len;
            try self.group_of.resize(a, @intCast(cap));
            @memset(self.group_of.items, -1);
            self.gslot.clearRetainingCapacity();
            self.gcount.clearRetainingCapacity();
            for (rows.slot, 0..) |s, i| {
                std.debug.assert(!(s >= cap));
                if (rows.act_row) |ar| std.debug.assert(ar[i] < a_rows);
                var gi = self.group_of.items[s];
                if (gi < 0) {
                    gi = @intCast(self.gslot.items.len);
                    self.group_of.items[s] = gi;
                    try self.gslot.append(a, s);
                    try self.gcount.append(a, 0);
                }
                self.gcount.items[@intCast(gi)] += 1;
            }
            const ng = self.gslot.items.len;
            // each group's rows, ascending (the lane appends rows in order)
            try self.gnext.resize(a, ng);
            var start: u32 = 0;
            for (self.gcount.items, self.gnext.items) |c, *nx| {
                nx.* = start;
                start += c;
            }
            try self.grows.resize(a, n);
            for (rows.slot, 0..) |s, i| {
                const gi: usize = @intCast(self.group_of.items[s]);
                self.grows.items[self.gnext.items[gi]] = @intCast(i);
                self.gnext.items[gi] += 1;
            }
            // snake over the rows-descending order (stable: ties keep first appearance)
            try self.sorted.resize(a, ng);
            for (self.sorted.items, 0..) |*v, i| v.* = @intCast(i);
            std.mem.sort(u32, self.sorted.items, @as([]const u32, self.gcount.items), struct {
                fn more(c: []const u32, x: u32, y: u32) bool {
                    return c[x] > c[y];
                }
            }.more);
            try self.snake.resize(a, ng);
            var lo: usize = 0;
            var hi: usize = ng;
            var k: usize = 0;
            while (lo < hi) {
                self.snake.items[k] = self.sorted.items[lo];
                k += 1;
                lo += 1;
                if (lo < hi) {
                    hi -= 1;
                    self.snake.items[k] = self.sorted.items[hi];
                    k += 1;
                }
            }
            // rows in wave order: the snake order's groups, each group's rows ascending
            try self.pos.resize(a, n);
            try self.ridx.resize(a, n);
            try self.rhs.resize(a, n);
            try self.inv.resize(a, n);
            var off: usize = 0;
            for (self.snake.items) |gi| {
                const c = self.gcount.items[gi];
                const first = self.gnext.items[gi] - c;
                for (self.grows.items[first..][0..c]) |row| {
                    self.pos.items[off] = row;
                    self.ridx.items[off] = @intCast(if (rows.act_row) |ar| ar[row] else row);
                    off += 1;
                }
            }
        }
    };
}

/// The per-projection bank check (the route above), under a name `Accepted.checkBank` can call.
const checkProjArrays = checkBank;

// ── Tests ──

const testing = std.testing;
const kt = sdk_ext.kernels.Trace(xk);
const Trace = kt.Trace;
const testRegistry = kt.testRegistry;
const expectLaunch = kt.expectLaunch;
const sampleAt = kt.sampleAt;
const traceRef = kt.traceRef;
const traceRefs = kt.traceRefs;
const traceEvent = kt.traceEvent;
const shapeStr = kt.shapeStr;
const Shape = kr.Shape;
const expert_bank = @import("../../../weights/streamed/expert/bank.zig");

const v41_spec: quant.Spec = .{ .hidden = 5120, .inter = 2304, .top_k = 6, .n_layers = 40, .act = .{ .swiglu_clamped = 10.0 }, .input = .bfloat16 };

// ── 2. claims on C1's description of the bank ──

const bank_peek_fixture = @embedFile("../../../weights/streamed/fixtures/dsv41_bank_peek.json");

test "dsv41 kernels c2: the EXL3 quant claims the bank of record's description (C1's peek) and declines each mutation, by field" {
    const a = testing.allocator;
    var why: Diag = .{};
    {
        var p = try expert_bank.peekText(a, bank_peek_fixture, null);
        defer p.deinit();
        const v = &p.view;
        try testing.expectEqual(@as(usize, 40), v.layers.len);
        try testing.expectEqualStrings("down_proj.code", v.layers[39].segments[6].name);
        try testing.expectEqualSlices(u64, &.{ 144, 320, 48 }, v.layers[39].segments[6].shape);
        try testing.expectEqual(@as(?quant.Priority, .native), claims(v, &why));
        // the stock gather quant declines it (exl3 is no MLX quantization mode)
        try testing.expectEqual(@as(?quant.Priority, null), quant.GatherQmm.claims(v, &why));
    }
    const Case = struct { needle: []const u8, replacement: []const u8, names: []const u8 };
    const cases = [_]Case{
        .{ .needle = "\"mode\":\"exl3\"", .replacement = "\"mode\":\"exl2\"", .names = "quantization.mode \"exl2\"" },
        .{ .needle = "\"codebook\":\"mul1\"", .replacement = "\"codebook\":\"3inst\"", .names = "quantization.codebook \"3inst\"" },
        .{ .needle = "\"codebook_multiplier\":2212286765", .replacement = "\"codebook_multiplier\":2212286766", .names = "codebook_multiplier 2212286766" },
        .{ .needle = "\"layout\":\"exl3-tensor-core\"", .replacement = "\"layout\":\"row-major\"", .names = "quantization.tile 16 / row-major" },
        .{ .needle = "\"layer\":7,\"K\":3", .replacement = "\"layer\":7,\"K\":4", .names = "layer 7 K 4" },
        .{ .needle = "\"hidden\":5120", .replacement = "\"hidden\":4096", .names = "dims hidden 4096" },
        .{ .needle = "\"component\":\"up_proj.rout\",\"dtype\":\"F16\"", .replacement = "\"component\":\"up_proj.rout\",\"dtype\":\"F32\"", .names = "segment up_proj.rout: its dtype" },
        .{ .needle = "\"order\":\"sylvester-natural\"", .replacement = "\"order\":\"walsh\"", .names = "quantization.hadamard 128 / walsh" },
        .{ .needle = "\"rin\":\"exl3.suh\"", .replacement = "\"rin\":\"exl3.sv\"", .names = "quantization.scales rin exl3.sv" },
        .{ .needle = "\"component\":\"gate_proj.rout\"", .replacement = "\"component\":\"gate_prj.rout\"", .names = "segment gate_prj.rout: an unknown projection" },
        .{ .needle = "\"component\":\"down_proj.rin\"", .replacement = "\"component\":\"down_proj.bias\"", .names = "segment down_proj.bias: an unknown tensor" },
        .{ .needle = "\"component\":\"gate_proj.code\",\"dtype\":\"I16\",\"shape\":[320,144,48]", .replacement = "\"component\":\"gate_proj.code\",\"dtype\":\"I16\",\"shape\":[144,320,48]", .names = "segment gate_proj.code: its shape" },
    };
    for (cases) |c| {
        const text = try std.mem.replaceOwned(u8, a, bank_peek_fixture, c.needle, c.replacement);
        defer a.free(text);
        var p = try expert_bank.peekText(a, text, null);
        defer p.deinit();
        why = .{};
        try testing.expectEqual(@as(?quant.Priority, null), claims(&p.view, &why));
        if (std.mem.indexOf(u8, why.message(), c.names) == null) {
            std.debug.print("declined for: {s}\n", .{why.message()});
            return error.TestUnexpectedResult;
        }
    }
    // C1's peek refuses a manifest it cannot describe, by name
    var bd: expert_bank.Diag = .{};
    try testing.expectError(error.ManifestFormat, expert_bank.peekText(a, "{\"format\":\"x\",\"quantization\":{},\"dims\":{\"hidden\":1,\"inter\":1,\"n_experts\":1,\"n_layers\":0},\"layers\":[]}", &bd));
    try testing.expectError(error.LayerGeometry, expert_bank.peekText(a, "{\"format\":\"mtplx-expert-manifest-v2\",\"quantization\":{},\"dims\":{\"hidden\":1,\"inter\":1,\"n_experts\":1,\"n_layers\":1},\"layers\":[{\"layer\":1,\"K\":3,\"segments\":[]}]}", &bd));
    try testing.expectError(error.ManifestSyntax, expert_bank.peekText(a, "{", &bd));
}

test "dsv41 kernels c2: the EXL3 quant is accepted only with the load's kernel set, and checks its Spec before building" {
    const a = testing.allocator;
    var diag: Diag = .{};
    var tb: Trace = .{ .a = a };
    defer tb.deinit();
    try testing.expectError(error.NoKernelSet, accept(Trace, a, &tb, .{}, v41_spec, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "without the load context's kernel set") != null);
    const set = try ks.Set.init(a, .{ .device = .{ .stub = .{} } }, &diag);
    defer set.deinit();
    set.install(Trace, &tb);
    var bad = v41_spec;
    bad.top_k = 9;
    try testing.expectError(error.TopKTooWide, accept(Trace, a, &tb, .{ .kernels = set.ref() }, bad, &diag));
    const acc = try accept(Trace, a, &tb, .{ .kernels = set.ref() }, v41_spec, &diag);
    defer acc.deinit(&tb);
    try testing.expectEqual(@as(usize, v41_spec.n_layers), acc.waves.len);
    try testing.expectEqual(decode_table_rows, acc.n_tok);
}

test "dsv41 kernels c2: the real 3.0 bank's own manifest is claimed (DSV41_BANK)" {
    const dir = std.mem.span(env.get(.DSV41_BANK) orelse return error.SkipZigTest);
    var bd: expert_bank.Diag = .{};
    var p = expert_bank.peek(testing.allocator, std.testing.io, dir, &bd) catch |e| {
        std.debug.print("refused: {s}\n", .{bd.message()});
        return e;
    };
    defer p.deinit();
    var why: Diag = .{};
    try testing.expectEqual(@as(?quant.Priority, .native), claims(&p.view, &why));
}

test "dsv41 kernels c2: the load's peek streams the bank's description past the page cache, as the text peek reads it" {
    const a = testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "expert-manifest-v2.json", .data = bank_peek_fixture });
    var rbuf: [512]u8 = undefined;
    const root = try expert_bank.tmpRoot(&tmp, &rbuf);
    var bd: expert_bank.Diag = .{};
    var s = try expert_bank.peek(a, std.testing.io, root, &bd);
    defer s.deinit();
    var t = try expert_bank.peekText(a, bank_peek_fixture, null);
    defer t.deinit();
    const qs = try std.fmt.allocPrint(a, "{f}", .{std.json.fmt(s.view.quantization, .{})});
    defer a.free(qs);
    const qt = try std.fmt.allocPrint(a, "{f}", .{std.json.fmt(t.view.quantization, .{})});
    defer a.free(qt);
    try testing.expectEqualStrings(qt, qs);
    try testing.expectEqual(t.view.hidden, s.view.hidden);
    try testing.expectEqual(t.view.inter, s.view.inter);
    try testing.expectEqual(t.view.n_experts, s.view.n_experts);
    try testing.expectEqual(t.view.n_layers, s.view.n_layers);
    try testing.expectEqual(t.view.layers.len, s.view.layers.len);
    for (t.view.layers, s.view.layers) |lt, ls| {
        try testing.expectEqual(lt.bits, ls.bits);
        try testing.expectEqual(lt.segments.len, ls.segments.len);
        for (lt.segments, ls.segments) |gt, gs| {
            try testing.expectEqualStrings(gt.name, gs.name);
            try testing.expectEqualStrings(gt.dtype, gs.dtype);
            try testing.expectEqualSlices(u64, gt.shape, gs.shape);
        }
    }
    var why: Diag = .{};
    try testing.expectEqual(@as(?quant.Priority, .native), claims(&s.view, &why));
}

// ── 6. The Spec contract ──

test "dsv41 kernels c2: the EXL3 quant refuses a Spec its texts do not implement, by name; the texts carry the 10.0 it declares" {
    var diag: Diag = .{};
    try checkSpec(v41_spec, &diag);
    const Case = struct { spec: quant.Spec, err: quant.Refusal, names: []const u8 };
    var s = v41_spec;
    const cases = [_]Case{
        .{ .spec = blk: {
            s = v41_spec;
            s.act = .swiglu;
            break :blk s;
        }, .err = error.ActivationNotFused, .names = "activation swiglu" },
        .{ .spec = blk: {
            s = v41_spec;
            s.act = .{ .swiglu_clamped = 7.0 };
            break :blk s;
        }, .err = error.ActivationNotFused, .names = "clamped at 7" },
        .{ .spec = blk: {
            s = v41_spec;
            s.hidden = 4096;
            s.inter = 1536;
            break :blk s;
        }, .err = error.DimsNotImplemented, .names = "dims 4096 / 1536" },
        .{ .spec = blk: {
            s = v41_spec;
            s.input = .float32;
            break :blk s;
        }, .err = error.InputDtype, .names = "MoE input float32" },
        .{ .spec = blk: {
            s = v41_spec;
            s.top_k = 9;
            break :blk s;
        }, .err = error.TopKTooWide, .names = "top_k 9" },
        .{ .spec = blk: {
            s = v41_spec;
            s.top_k = 0;
            break :blk s;
        }, .err = error.TopKTooWide, .names = "top_k 0" },
        .{ .spec = blk: {
            s = v41_spec;
            s.n_layers = 0;
            break :blk s;
        }, .err = error.DimsNotImplemented, .names = "no layers" },
    };
    for (cases) |c| {
        try testing.expectError(c.err, checkSpec(c.spec, &diag));
        if (std.mem.indexOf(u8, diag.message(), c.names) == null) {
            std.debug.print("refused with: {s}\n", .{diag.message()});
            return error.TestUnexpectedResult;
        }
    }
    // the literals of the fused activation: gu_epi's 10.0f and the DIG2 header's f32 bits of +-10.0
    const gu = xk.embedded.sources[@backingInt(Kernel.q3_exl3_prep_gu_epi)];
    try testing.expectEqual(@as(usize, 12), std.mem.count(u8, gu, "10.0f"));
    try testing.expectEqual(@as(usize, 4), std.mem.count(u8, gu, "-10.0f"));
    const dig2 = xk.embedded.headers[@backingInt(xk.Header.dig2_x)];
    try testing.expect(std.mem.indexOf(u8, dig2, "LIMIT = as_type<float>(1092616192u)") != null);
    try testing.expect(std.mem.indexOf(u8, dig2, "NLIMIT = as_type<float>(3240099840u)") != null);
    try testing.expectEqual(@as(u32, 1092616192), @as(u32, @bitCast(@as(f32, @floatCast(fused_act.swiglu_clamped)))));
    try testing.expectEqual(@as(u32, 3240099840), @as(u32, @bitCast(@as(f32, @floatCast(-fused_act.swiglu_clamped)))));
    var reg = try testRegistry();
    defer reg.deinit();
    try testing.expectEqual(xk.Header.dig2_x, reg.get(.q3_prefill_dig2_swiglu_2304_x).header.?);
}

test "dsv41 kernels ops: wave tables are the lanes' (DIG wave_table, rebuild slots8)" {
    var reg = try testRegistry();
    defer reg.deinit();
    const dx = DigX(Trace).init(&reg);
    try testing.expectEqual(@as(u32, 72), dx.digTiles(.gate_up));
    try testing.expectEqual(@as(u32, 80), dx.digTiles(.down));
    const w = digTableBm(&.{ .{ .slot = 3, .rows = 70 }, .{ .slot = 0, .rows = 37 }, .{ .slot = 2, .rows = 20 } }, 72, 64);
    try testing.expectEqualSlices(i32, &.{ 3, 0, 2, 0 }, w.table[0..4]);
    try testing.expectEqualSlices(i32, &.{ 0, 70, 107, 0 }, w.table[16..20]);
    try testing.expectEqualSlices(i32, &.{ 70, 37, 20, 0 }, w.table[32..36]);
    try testing.expectEqualSlices(i32, &.{ 0, 144, 216, std.math.maxInt(i32) }, w.table[48..52]);
    try testing.expectEqual(@as(i32, 3), w.table[64]);
    try testing.expectEqual(@as(u32, 288), w.tgs);
    try testing.expectEqual(@as(i32, 288), w.table[65]);
    try testing.expectEqualSlices(i32, &.{ 3, 0, 2, 3, 3 }, rebuildSlots(&.{ 3, 0, 2 })[0..5]);
}

// ── The prefill wave route vs the lane of record's own dispatch (dump_prefill_waves.py --samples) ──

const prefill_samples = @embedFile("../../../weights/streamed/fixtures/dsv41_prefill_wave_samples.json");
const JRoute = struct { seed: u64, slots: []const u32, counts: []const u32 };
const JCall = struct { name: []const u8, a_rows: u32, route: JRoute, events: []const []const u8, ret: []const u8, ret_shape: []const i64 };
const JShapeCfg = struct { wave: u32, inflight: u32, row_budget: u32, carry_rows: u32 };
const JSampleCase = struct { case: []const u8, shape: JShapeCfg, cap: u32, calls: []const JCall, finish: []const []const u8 };
const JSamples = struct { format: []const u8, cases: []const JSampleCase };

/// dump_prefill_waves.route_rows: slot j repeated counts[j] times, then Fisher-Yates from the end
/// with j = splitmix64(seed) output k % (i + 1), k = 0, 1, ... as i runs A - 1 .. 1.
fn routeRows(a: Allocator, seed: u64, slots: []const u32, counts: []const u32) ![]u32 {
    var n: usize = 0;
    for (counts) |c| n += c;
    const rows = try a.alloc(u32, n);
    var k: usize = 0;
    for (slots, counts) |s, c| for (0..c) |_| {
        rows[k] = s;
        k += 1;
    };
    var st = seed;
    var i = n;
    while (i > 1) {
        i -= 1;
        const j: usize = @intCast(xk.splitmix64(&st) % (i + 1));
        std.mem.swap(u32, &rows[i], &rows[j]);
    }
    return rows;
}


/// A take2 launch line's two forms. The lane samples and the move's pinned log name the lane's take2
/// (`q3_prefill_dig_rot_take2_5120`, grid z 2: one simdgroup per output); the route launches the retune
/// (`dsv41_prefill_dig_take2v_5120`, grid z 1: both outputs per simdgroup) over the same inputs and outputs, so the
/// two lines differ in the kernel name and the grid's z only.
const Take2Form = struct { prefix: []const u8, z: u8 };
const take2_lane: Take2Form = .{ .prefix = "launch q3_prefill_dig_rot_take2_5120 g=1280,", .z = '2' };
const take2_retune: Take2Form = .{ .prefix = "launch dsv41_prefill_dig_take2v_5120 g=1280,", .z = '1' };

/// `line` in the `to` form when it is a take2 launch in the `from` form (allocated in `a`), else `line` itself.
fn swapTake2(a: Allocator, line: []const u8, from: Take2Form, to: Take2Form) ![]const u8 {
    if (!std.mem.startsWith(u8, line, from.prefix)) return line;
    const rest = line[from.prefix.len..];
    const comma = std.mem.indexOfScalar(u8, rest, ',') orelse return error.TestUnexpectedResult;
    if (comma + 2 >= rest.len or rest[comma + 1] != from.z or rest[comma + 2] != ' ') return error.TestUnexpectedResult;
    return std.mem.concat(a, u8, &.{ to.prefix, rest[0..comma], &.{ ',', to.z }, rest[comma + 2 ..] });
}

/// The route's lines in the lane's text. The lane samples and the move's pinned log name the lane's take2 (grid z 2),
/// the 64-row GEMMs (grid 128 x tgs, t=128) over 64-row wave tables and rot_widen1 after the down GEMM: five launches
/// per wave. The route launches the take2 retune (grid z 1) and the 128-row GEMMs (grid 256 x tgs', t=256); the fused
/// arm launches, in place of the down GEMM and rot_widen1, the fused down GEMM (grid 512 x tgs'', t=512: the 128-row
/// down text at its 40 column tiles with rot_widen1 as its epilogue, the two launches' output words). Their 128-row
/// tables' slot / first-row / row columns are the lane's; their first-threadgroup column counts 128-row tiles. `init`
/// takes, per 128-row table the trace holds (recomputed and compared word for word), its 64-row twin's reference and
/// threadgroups (the fused down GEMM's twin: the lane's 80-tile down table); per route launch, its output's index in
/// the lane's numbering (one more per fused down GEMM before it, a fused down GEMM's output its rot_widen1's); and per
/// fused down GEMM, the rhs its lane rot_widen1 reads (onepass's). `lane` rewrites a rendered line: every launch
/// reference renumbered, a take2 retune launch into the lane's, a 128-row GEMM launch's name, grid and threadgroup into
/// the 64-row text's, a fused down GEMM launch into the lane's two (the 64-row down GEMM, then rot_widen1 over its
/// output), and every 128-row table reference into its twin's. Nothing else changes.
const LaneMap = struct {
    const table_ref = "host:int32:[80]:";
    const Twin = struct { ref: [16]u8, tgs: u32, tgs128: u32 };
    const GemmForm = struct { m128: []const u8, lane: []const u8 };
    const gemm_forms = [_]GemmForm{
        .{ .m128 = "launch dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128 g=", .lane = "launch q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3 g=" },
        .{ .m128 = "launch dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128 g=", .lane = "launch q3_prefill_dig_gemm_2304x5120_xmul1hk3 g=" },
    };
    const w1_kernel: Kernel = .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1;
    const w1_route = "launch " ++ @tagName(w1_kernel) ++ " g=";
    const dn_lane = "launch " ++ @tagName(Kernel.q3_prefill_dig_gemm_2304x5120_xmul1hk3) ++ " g=";
    const widen1_lane = "launch " ++ @tagName(Kernel.q3_prefill_dig_rot_widen1_5120) ++ " g=1280,";
    /// A fused down GEMM's lane lines: the down GEMM's lane launch index (rot_widen1 reads its output) and the rhs.
    const W1 = struct { down: u32, rhs: []const u8 };
    /// the fused down GEMM's column tiles per M tile; the lane's down table's (its twin's)
    w_tiles: u32,
    dn_tiles: u32,
    twins: std.AutoHashMapUnmanaged([16]u8, Twin) = .empty,
    /// per route launch, its output's launch index in the lane's text
    lane_of: std.ArrayList(u32) = .empty,
    /// per fused down GEMM, keyed by its input's (onepass's) lane launch index
    w1s: std.AutoHashMapUnmanaged(u32, W1) = .empty,
    n_take2: usize = 0,
    n_gemm: usize = 0,
    n_w1: usize = 0,

    fn refOf(tb: *const [80]i32) [16]u8 {
        const hex = kset.shaHex(std.mem.sliceAsBytes(tb));
        return hex[0..16].*;
    }

    /// The 64-row twin of an 80-word table when it is a 128-row wave table (recomputed, word for word), else null: at
    /// its own tiles per M tile, the lane's down table's for the fused down GEMM's.
    fn twinOf(m: *const LaneMap, tb: *const [80]i32) ?Twin {
        const n: usize = @intCast(@max(tb[64], 0));
        if (n == 0 or n > wave_max or tb[65] <= 0) return null;
        var ex: [wave_max]WaveExpert = undefined;
        var t128: u64 = 0;
        for (0..n) |j| {
            if (tb[j] < 0 or tb[32 + j] <= 0) return null;
            ex[j] = .{ .slot = @intCast(tb[j]), .rows = @intCast(tb[32 + j]) };
            t128 += (ex[j].rows + 127) / 128;
        }
        const total: u64 = @intCast(tb[65]);
        if (total % t128 != 0) return null;
        const tiles: u32 = @intCast(total / t128);
        const as128 = digTableBm(ex[0..n], tiles, 128);
        if (!std.mem.eql(i32, &as128.table, tb)) return null;
        const twin = digTableBm(ex[0..n], if (tiles == m.w_tiles) m.dn_tiles else tiles, 64);
        return .{ .ref = refOf(&twin.table), .tgs = twin.tgs, .tgs128 = as128.tgs };
    }

    fn init(a: Allocator, t: *const Trace, reg: *const xk.Registry) !LaneMap {
        const dx = DigX(Trace).init(reg);
        var m: LaneMap = .{ .w_tiles = dx.widenTiles(), .dn_tiles = dx.digTiles(.down) };
        errdefer m.deinit(a);
        for (t.nodes.items) |nd| {
            if (nd.origin != .host or nd.dtype != .int32 or nd.bytes.len != 80 * 4) continue;
            var tb: [80]i32 = undefined;
            @memcpy(std.mem.sliceAsBytes(&tb), nd.bytes);
            if (m.twinOf(&tb)) |tw| try m.twins.put(a, refOf(&tb), tw);
        }
        try m.lane_of.resize(a, t.launches.items.len);
        var shift: u32 = 0;
        for (t.launches.items, m.lane_of.items, 0..) |*l, *lo, li| {
            shift += @intFromBool(l.k == w1_kernel);
            lo.* = @as(u32, @intCast(li)) + shift;
        }
        var rb: std.ArrayList(u8) = .empty;
        defer rb.deinit(t.a);
        for (t.launches.items, m.lane_of.items) |*l, lo| {
            if (l.k != w1_kernel) continue;
            // its input is onepass's output; the lane's rot_widen1 reads the wave's rhs (onepass's)
            const op = switch (t.nodes.items[l.inputs[0]].origin) {
                .out => |o| o[0],
                else => return error.TestUnexpectedResult,
            };
            const opl = &t.launches.items[op];
            if (opl.k != .q3_prefill_dig2_swiglu_2304_x) return error.TestUnexpectedResult;
            rb.clearRetainingCapacity();
            try traceRef(t, opl.inputs[2], &rb);
            const rhs = try a.dupe(u8, rb.items);
            errdefer a.free(rhs);
            const gop = try m.w1s.getOrPut(a, m.lane_of.items[op]);
            if (gop.found_existing) return error.TestUnexpectedResult;
            gop.value_ptr.* = .{ .down = lo - 1, .rhs = rhs };
        }
        return m;
    }

    fn deinit(m: *LaneMap, a: Allocator) void {
        var it = m.w1s.valueIterator();
        while (it.next()) |w| a.free(w.rhs);
        m.w1s.deinit(a);
        m.lane_of.deinit(a);
        m.twins.deinit(a);
    }

    /// `line` in the lane's text (allocated in `a` when rewritten): one line, or for a fused down GEMM the lane's two
    /// joined by a newline.
    fn lane(m: *LaneMap, a: Allocator, line: []const u8) ![]const u8 {
        const r = try m.renumber(a, line);
        var l = try swapTake2(a, r, take2_retune, take2_lane);
        m.n_take2 += @intFromBool(l.ptr != r.ptr);
        for (gemm_forms) |f| {
            if (!std.mem.startsWith(u8, l, f.m128)) continue;
            l = try m.gemm128(a, l, f);
            m.n_gemm += 1;
            break;
        }
        if (std.mem.startsWith(u8, l, w1_route)) {
            l = try m.fusedDown(a, l);
            m.n_w1 += 1;
        }
        return m.twinRefs(a, l);
    }

    /// `line` with every launch reference `L<i>.<o>` at its lane index.
    fn renumber(m: *const LaneMap, a: Allocator, line: []const u8) ![]const u8 {
        var out: std.ArrayList(u8) = .empty;
        var at: usize = 0;
        var i: usize = 1;
        while (i < line.len) : (i += 1) {
            if (line[i] != 'L' or std.mem.indexOfScalar(u8, "=; ", line[i - 1]) == null) continue;
            var j = i + 1;
            while (j < line.len and std.ascii.isDigit(line[j])) j += 1;
            if (j == i + 1 or j == line.len or line[j] != '.') continue;
            const li = try std.fmt.parseInt(usize, line[i + 1 .. j], 10);
            if (li >= m.lane_of.items.len) return error.TestUnexpectedResult;
            try out.appendSlice(a, line[at .. i + 1]);
            try out.print(a, "{d}", .{m.lane_of.items[li]});
            at = j;
            i = j;
        }
        if (at == 0) return line;
        try out.appendSlice(a, line[at..]);
        return out.items;
    }

    /// The twin of a 128-row table reference.
    fn tableTwin(m: *const LaneMap, ref: []const u8) !Twin {
        if (!std.mem.startsWith(u8, ref, table_ref) or ref.len != table_ref.len + 16) return error.TestUnexpectedResult;
        return m.twins.get(ref[table_ref.len..][0..16].*) orelse error.TestUnexpectedResult;
    }

    /// A 128-row GEMM launch line with the 64-row text's name, grid and threadgroup (its table is its last input).
    fn gemm128(m: *const LaneMap, a: Allocator, l: []const u8, f: GemmForm) ![]const u8 {
        const refs_at = (std.mem.indexOf(u8, l, " in=") orelse return error.TestUnexpectedResult) + " in=".len;
        const refs_end = std.mem.indexOfPos(u8, l, refs_at, " out=") orelse return error.TestUnexpectedResult;
        const refs = l[refs_at..refs_end];
        const tw = try m.tableTwin(refs[(std.mem.lastIndexOfScalar(u8, refs, ';') orelse return error.TestUnexpectedResult) + 1 ..]);
        const geo = l[f.m128.len..];
        const sp = std.mem.indexOfScalar(u8, geo, ' ') orelse return error.TestUnexpectedResult;
        var gb: [32]u8 = undefined;
        if (!std.mem.eql(u8, geo[0..sp], try std.fmt.bufPrint(&gb, "{d},1,1", .{256 * tw.tgs128}))) return error.TestUnexpectedResult;
        const t256 = " t=256,1,1 ";
        if (!std.mem.startsWith(u8, geo[sp..], t256)) return error.TestUnexpectedResult;
        return std.fmt.allocPrint(a, "{s}{d},1,1 t=128,1,1 {s}", .{ f.lane, 128 * tw.tgs, geo[sp + t256.len ..] });
    }

    /// A fused down GEMM launch line (its references already the lane's) as the lane's two: the 64-row down GEMM over
    /// onepass's output, then rot_widen1 over the down GEMM's output, the rhs, the down table and the down rout.
    fn fusedDown(m: *const LaneMap, a: Allocator, l: []const u8) ![]const u8 {
        const geo = l[w1_route.len..];
        const sp = std.mem.indexOfScalar(u8, geo, ' ') orelse return error.TestUnexpectedResult;
        const rest = geo[sp..];
        const t512 = " t=512,1,1";
        if (!std.mem.startsWith(u8, rest, t512)) return error.TestUnexpectedResult;
        const in_at = std.mem.indexOf(u8, rest, " in=") orelse return error.TestUnexpectedResult;
        // what lies between the threadgroup and the inputs (`renderLog`'s prepared flag and template), in both lines
        const mid = rest[t512.len..in_at];
        const out_at = std.mem.indexOfPos(u8, rest, in_at, " out=") orelse return error.TestUnexpectedResult;
        var it = std.mem.splitScalar(u8, rest[in_at + " in=".len .. out_at], ';');
        var ins: [4][]const u8 = undefined;
        for (&ins) |*r| r.* = it.next() orelse return error.TestUnexpectedResult;
        if (it.next() != null) return error.TestUnexpectedResult;
        const x, const code, const rout, const tbl = ins;
        const tw = try m.tableTwin(tbl);
        var gb: [32]u8 = undefined;
        if (!std.mem.eql(u8, geo[0..sp], try std.fmt.bufPrint(&gb, "{d},1,1", .{512 * tw.tgs128}))) return error.TestUnexpectedResult;
        const out = rest[out_at + " out=".len ..];
        const o0, const o1 = .{ "float32[", ",5120]" };
        if (!std.mem.startsWith(u8, out, o0) or !std.mem.endsWith(u8, out, o1)) return error.TestUnexpectedResult;
        const rows = out[o0.len .. out.len - o1.len];
        // x: onepass's output, L<its lane index>.0
        if (x.len < 4 or x[0] != 'L' or !std.mem.endsWith(u8, x, ".0")) return error.TestUnexpectedResult;
        const w = m.w1s.get(try std.fmt.parseInt(u32, x[1 .. x.len - 2], 10)) orelse return error.TestUnexpectedResult;
        return std.fmt.allocPrint(a, "{s}{d},1,1 t=128,1,1{s} in={s};{s};{s} out={s}\n{s}{s},1 t=256,1,1{s} in=L{d}.0;{s};{s};{s} out=float32[{s},1,5120]", .{ dn_lane, 128 * tw.tgs, mid, x, code, tbl, out, widen1_lane, rows, mid, w.down, w.rhs, tbl, rout, rows });
    }

    /// `l` with every 128-row table reference its twin's.
    fn twinRefs(m: *const LaneMap, a: Allocator, l: []const u8) ![]const u8 {
        if (std.mem.indexOf(u8, l, table_ref) == null) return l;
        var out: std.ArrayList(u8) = .empty;
        var at: usize = 0;
        while (std.mem.indexOfPos(u8, l, at, table_ref)) |p| {
            const h = p + table_ref.len;
            try out.appendSlice(a, l[at..h]);
            at = h;
            if (h + 16 <= l.len) if (m.twins.get(l[h..][0..16].*)) |tw| {
                try out.appendSlice(a, &tw.ref);
                at = h + 16;
            };
        }
        try out.appendSlice(a, l[at..]);
        return out.items;
    }
};

/// The trace's log from `from` on, in the lane's text (`LaneMap.lane`: a fused down GEMM's event is two lines), is the
/// lane's `want`, line for line.
fn expectEventsVia(t: *const Trace, from: usize, want: []const []const u8, case: []const u8, what: []const u8, lm: *LaneMap) !void {
    var arena = std.heap.ArenaAllocator.init(t.a);
    defer arena.deinit();
    const aa = arena.allocator();
    var got: std.ArrayList([]const u8) = .empty;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(t.a);
    for (t.log.items[from..]) |e| {
        buf.clearRetainingCapacity();
        try traceEvent(t, e, &buf);
        var lines = std.mem.splitScalar(u8, try lm.lane(aa, buf.items), '\n');
        while (lines.next()) |l| try got.append(aa, try aa.dupe(u8, l));
    }
    for (got.items[0..@min(got.items.len, want.len)], want[0..@min(got.items.len, want.len)], 0..) |line, w, i| {
        if (!std.mem.eql(u8, line, w)) {
            std.debug.print("prefill {s} {s} line {d}:\n  route: {s}\n  lane:  {s}\n", .{ case, what, i, line, w });
            return error.TestExpectedEqual;
        }
    }
    if (got.items.len != want.len) {
        std.debug.print("prefill {s} {s}: {d} lines, the lane {d}\n", .{ case, what, got.items.len, want.len });
        return error.TestExpectedEqual;
    }
}

/// Each wave's `resetTo` runs right after the wave's `per_wave` launches and its eval / async_eval
/// (only drains of older waves may come between) and frees those launches' outputs; the last
/// `resetTo` of the call follows the join (concatenate, take, the result's eval) and frees the
/// concatenation and the take.
fn expectLifecycle(t: *const Trace, log0: usize, resets0: usize, waves: usize, per_wave: usize, case: []const u8, what: []const u8) !void {
    var at = log0;
    for (t.freed.items[resets0..][0..waves], 0..) |r, w| {
        var launches: usize = 0;
        var last_out: ?Trace.T = null;
        var own_eval = false;
        for (t.log.items[at..r.at]) |e| switch (e) {
            .launch => |li| {
                launches += 1;
                const l = &t.launches.items[li];
                last_out = l.outs[0];
                for (l.outs[0..l.cfg.n_out]) |o| if (o < r.from or o >= r.to) {
                    std.debug.print("prefill {s} {s} wave {d}: launch output {d} outside its reset [{d}, {d})\n", .{ case, what, w, o, r.from, r.to });
                    return error.TestUnexpectedResult;
                };
            },
            // the wave's own eval / async_eval: its down stage's output (through the reshape)
            .eval, .async_eval => |xs| own_eval = own_eval or (launches == per_wave and xs.len == 1 and last_out != null and t.root(xs[0]) == last_out.?),
            .concat, .take, .op => {
                std.debug.print("prefill {s} {s} wave {d}: a join inside a wave's reset\n", .{ case, what, w });
                return error.TestUnexpectedResult;
            },
        };
        if (launches != per_wave or !own_eval) {
            std.debug.print("prefill {s} {s} wave {d}: {d} launches, own eval {} before its reset\n", .{ case, what, w, launches, own_eval });
            return error.TestUnexpectedResult;
        }
        // its output kept (the join's handle, and the in-flight queue's unless solo) before its reset
        var keeps: usize = 0;
        for (t.kept.items) |k| keeps += @intFromBool(k.resets == resets0 + w and t.root(k.node) == last_out.?);
        if (keeps < 1 or keeps > 2) {
            std.debug.print("prefill {s} {s} wave {d}: its output kept {d} times before its reset\n", .{ case, what, w, keeps });
            return error.TestUnexpectedResult;
        }
        at = r.at;
    }
    const j = t.freed.items[resets0 + waves];
    var cat = false;
    var tk = false;
    for (t.log.items[at..j.at]) |e| switch (e) {
        .launch => return error.TestUnexpectedResult,
        .concat => cat = true,
        .take => tk = true,
        else => {},
    };
    const last = t.nodes.items.len - 1;
    // the result (the take, the call's last node) kept before the join's reset
    var result_kept = false;
    for (t.kept.items) |k| result_kept = result_kept or (k.node == last and k.resets == resets0 + waves);
    if (!cat or !tk or j.to != last + 1 or !result_kept) {
        std.debug.print("prefill {s} {s}: the join's reset [{d}, {d}) (concat {}, take {}, result kept {})\n", .{ case, what, j.from, j.to, cat, tk, result_kept });
        return error.TestUnexpectedResult;
    }
}

fn testBank(t: *Trace, cap: c_int) !BankArrays(Trace.T) {
    return .{
        .gate = .{ .code = try t.ext("gate_proj.code", &.{ cap, 320, 144, 48 }, .int16), .rout = try t.ext("gate_proj.rout", &.{ cap, 2304 }, .float16), .rin = try t.ext("gate_proj.rin", &.{ cap, 5120 }, .float16) },
        .up = .{ .code = try t.ext("up_proj.code", &.{ cap, 320, 144, 48 }, .int16), .rout = try t.ext("up_proj.rout", &.{ cap, 2304 }, .float16), .rin = try t.ext("up_proj.rin", &.{ cap, 5120 }, .float16) },
        .down = .{ .code = try t.ext("down_proj.code", &.{ cap, 144, 320, 48 }, .int16), .rout = try t.ext("down_proj.rout", &.{ cap, 5120 }, .float16), .rin = try t.ext("down_proj.rin", &.{ cap, 2304 }, .float16) },
    };
}

test "dsv41 kernels ops: the prefill wave route replays the lane's own launches, evals and joins (lane samples)" {
    const a = testing.allocator;
    var reg = try testRegistry();
    defer reg.deinit();
    const parsed = try std.json.parseFromSlice(JSamples, a, prefill_samples, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try testing.expectEqualStrings("mlx-serve-exl3-prefill-wave-samples-v1", parsed.value.format);
    // both down stages: stock (the 128-row down text, then rot_widen1) and the fused arm
    for ([_]DigXPrefill(Trace).Down{ .chain, .fused }) |down| {
        var n_calls: usize = 0;
        var n_waves: usize = 0;
        var n_take2: usize = 0;
        var n_gemm: usize = 0;
        var n_w1: usize = 0;
        for (parsed.value.cases) |*cs| {
            var t: Trace = .{ .a = a };
            defer t.deinit();
            const shape: PrefillShape = .{ .wave = cs.shape.wave, .inflight = cs.shape.inflight, .row_budget = cs.shape.row_budget, .carry_rows = cs.shape.carry_rows };
            var r = try DigXPrefill(Trace).init(a, &reg, shape, null);
            defer r.deinit(&t);
            r.installDown(down);
            const bank = try testBank(&t, @intCast(cs.cap));
            var mark: usize = 0;
            for (cs.calls) |*cl| {
                const slots = try routeRows(a, cl.route.seed, cl.route.slots, cl.route.counts);
                defer a.free(slots);
                try testing.expectEqual(@as(usize, cl.a_rows), slots.len);
                const act = try t.ext("act", &.{ @intCast(slots.len), 5120 }, .bfloat16);
                const launches0 = t.launches.items.len;
                const nodes0 = t.nodes.items.len;
                const resets0 = t.freed.items.len;
                const res = try r.call(&t, act, .{ .slot = slots }, bank);
                // the route's events in the lane's text (the take2 retune, the 128-row GEMMs and the fused down GEMM
                // mapped, launch references renumbered); every other byte as sampled
                var lm = try LaneMap.init(a, &t, &reg);
                defer lm.deinit(a);
                try expectEventsVia(&t, mark, cl.events, cs.case, cl.name, &lm);
                n_take2 += lm.n_take2;
                n_gemm += lm.n_gemm;
                n_w1 += lm.n_w1;
                // the wave lifecycle: one mark / resetTo per wave and one around the join; nothing the
                // call built outlives it but the kept result and the waves still in flight
                const waves = (t.launches.items.len - launches0) / r.wave_launches;
                try testing.expectEqual(waves + 1, t.freed.items.len - resets0);
                try expectLifecycle(&t, mark, resets0, waves, r.wave_launches, cs.case, cl.name);
                for (nodes0..t.nodes.items.len) |x| {
                    if (!t.leaked(@intCast(x))) continue;
                    std.debug.print("prefill {s} {s}: node {d} outlives the call\n", .{ cs.case, cl.name, x });
                    return error.TestUnexpectedResult;
                }
                try testing.expectEqual(@as(usize, 1 + r.flight.items.len), t.held.items.len);
                try testing.expect(std.mem.indexOfScalar(Trace.T, t.held.items, res) != null);
                mark = t.log.items.len;
                var buf: std.ArrayList(u8) = .empty;
                defer buf.deinit(a);
                try traceRef(&t, res, &buf);
                try testing.expectEqualStrings(cl.ret, buf.items);
                const rs = t.shapeOf(res);
                for (cl.ret_shape, rs.slice()) |w, d| try testing.expectEqual(w, @as(i64, d));
                n_calls += 1;
                n_waves += waves;
                t.release(res);
            }
            try r.finish(&t);
            var lf = try LaneMap.init(a, &t, &reg);
            defer lf.deinit(a);
            try expectEventsVia(&t, mark, cs.finish, cs.case, "finish", &lf);
            try testing.expectEqual(@as(isize, 0), t.keeps);
        }
        try testing.expect(n_calls >= 10 and n_waves >= 60);
        // every sampled wave's take2 went through the retune and its GEMMs through the 128-row texts, the down GEMM
        // and rot_widen1 through the fused down GEMM on the fused arm
        try testing.expectEqual(n_waves, n_take2);
        try testing.expectEqual(@as(usize, if (down == .fused) 1 else 2) * n_waves, n_gemm);
        try testing.expectEqual(if (down == .fused) n_waves else 0, n_w1);
    }
}

test "dsv41 kernels ops: L1: a K16 group call packs the tier's 8 experts per wave, half of Record 3's waves, every row once" {
    const a = testing.allocator;
    var reg = try testRegistry();
    defer reg.deinit();
    // One group call of a K16 layer at 16K: 48 experts (max_route_ids) of 256 rows each (98,304 / 384), the
    // row budget (7,168) far above 8 x 256.
    const n_experts: u32 = 48;
    const per: u32 = 256;
    const slots = try a.alloc(u32, n_experts * per);
    defer a.free(slots);
    for (slots, 0..) |*s, i| s.* = @intCast(i % n_experts);
    try testing.expectEqual(@as(u32, 8), PrefillShape.tier.wave);
    for ([_]struct { wave: u32, waves: usize }{ .{ .wave = PrefillShape.record3.wave, .waves = 12 }, .{ .wave = PrefillShape.tier.wave, .waves = 6 } }) |c| {
        var t: Trace = .{ .a = a };
        defer t.deinit();
        var shape = PrefillShape.tier;
        shape.wave = c.wave;
        var r = try DigXPrefill(Trace).init(a, &reg, shape, null);
        defer r.deinit(&t);
        const bank = try testBank(&t, @intCast(n_experts));
        const act = try t.ext("act", &.{ @intCast(slots.len), 5120 }, .bfloat16);
        const l0 = t.launches.items.len;
        const res = try r.call(&t, act, .{ .slot = slots }, bank);
        // five launches per wave, and one wave per `wave` experts
        const launches = t.launches.items.len - l0;
        try testing.expectEqual(@as(usize, 0), launches % r.wave_launches);
        try testing.expectEqual(c.waves, launches / r.wave_launches);
        // the result keeps every assignment row, in assignment order (the join), whatever the waves
        const rsh = t.shapeOf(res);
        const rs = rsh.slice();
        try testing.expectEqual(@as(usize, 2), rs.len);
        try testing.expectEqual(@as(i64, @intCast(slots.len)), @as(i64, rs[0]));
        try testing.expectEqual(@as(i64, 5120), @as(i64, rs[1]));
        t.release(res);
        try r.finish(&t);
    }
}

// DSV41_PHASE0B_MLX=1 + DSV41_BANK=<bank>, on a GPU machine (any MLX array creates the Metal device): L1's
// first device run and its exactness proof, seconds long, no model load. A handful of real records (layer 0's
// first 16 experts, loaded by the stream into MLX slot rows) through the real DIG-X prefill launches, one K16-shaped
// group call at 8 experts per wave against the same call at Record 3's 4: the outputs equal, bit for bit.
test "dsv41 smoke 0b: L1: DIG-X prefill waves at 8 experts per wave equal Record 3's 4, bit for bit, on real records" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_BANK) orelse return error.SkipZigTest);
    const ops = @import("../../../weights/streamed/ops.zig");
    const es = @import("../../../weights/streamed/expert/stream.zig");
    const eb = @import("../../../weights/streamed/expert/bank.zig");
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try ops.MlxOps.init(a, s);
    defer g.deinit();
    var kd: xk.Diag = .{};
    const set = ks.Set.init(a, .{ .device = .{ .stream = s } }, &kd) catch |e| {
        std.debug.print("kernel set refused: {s}\n", .{kd.message()});
        return e;
    };
    defer set.deinit();
    set.install(ops.MlxOps, &g);
    defer ks.Set.uninstall(ops.MlxOps, &g);
    // Layer 0's first 16 experts, read by the stream into 16 persistent MLX rows (the base bank).
    var bdiag: eb.Diag = .{};
    var bank = eb.Bank.open(a, io, dir, eb.dsv41, &bdiag) catch |e| {
        std.debug.print("bank refused: {s}\n", .{bdiag.message()});
        return e;
    };
    defer bank.deinit();
    const n_experts = 16;
    var rows: [40]u32 = @splat(0);
    rows[0] = n_experts;
    const st = try es.Stream.init(a, &bank, .{ .rows = &rows, .max_route_ids = n_experts, .transient_rows = n_experts, .slot_memory = .{ .mlx = s } });
    defer st.deinit();
    var ids: [n_experts]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(i);
    const r = try st.route(0, &ids, &.{});
    defer st.release(r);
    for (0..r.n_parts) |p| {
        try st.waitGu(r, @intCast(p));
        try st.waitDown(r, @intCast(p));
    }
    var refs: [@import("../../../weights/streamed/sdk_ext.zig").expert.policy.max_route_ids]es.SlotRef = undefined;
    const rf = st.refsOf(r, &refs);
    try testing.expectEqual(@as(usize, n_experts), rf.len);
    for (rf) |x| try testing.expectEqual(es.BankKind.base, x.bank);
    const ba = st.bankArrays(0, .base) orelse return error.TestUnexpectedResult;
    const bank_arrays: BankArrays(ops.MlxOps.T) = .{
        .gate = .{ .code = ba.gate.code, .rout = ba.gate.rout, .rin = ba.gate.rin },
        .up = .{ .code = ba.up.code, .rout = ba.up.rout, .rin = ba.up.rin },
        .down = .{ .code = ba.down.code, .rout = ba.down.rout, .rin = ba.down.rin },
    };
    // A K16-shaped group call at a handful of records: 16 experts x 64 rows, rows interleaved by expert.
    const per = 64;
    const slots = try a.alloc(u32, n_experts * per);
    defer a.free(slots);
    for (slots, 0..) |*sl, i| sl.* = rf[i % n_experts].row;
    const xs = try a.alloc(f32, slots.len * 5120);
    defer a.free(xs);
    for (xs, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 23)) - 11)) / 32.0;
    const act = try g.astype(try g.hostArray(std.mem.sliceAsBytes(xs), &.{ @intCast(slots.len), 5120 }, .float32), .bfloat16);
    var outs: [2][]f32 = undefined;
    var n_out: usize = 0;
    defer for (outs[0..n_out]) |o| a.free(o);
    for ([_]PrefillShape{ PrefillShape.record3, PrefillShape.tier }) |shape| {
        var w = try DigXPrefill(ops.MlxOps).init(a, &set.reg, shape, &kd);
        defer w.deinit(&g);
        const y = try w.call(&g, act, .{ .slot = slots }, bank_arrays);
        defer g.release(y);
        try w.finish(&g);
        outs[n_out] = try a.alloc(f32, slots.len * 5120);
        n_out += 1;
        _ = try g.hostF32(y, outs[n_out - 1]);
    }
    try testing.expect(PrefillShape.tier.wave == 8 and PrefillShape.record3.wave == 4);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(outs[0]), std.mem.sliceAsBytes(outs[1]));
    var nonzero: usize = 0;
    for (outs[1]) |v| nonzero += @intFromBool(v != 0);
    std.debug.print("\nL1 smoke: {d} rows x 5120 over {d} real records: wave 8 == wave 4, bit for bit ({d} nonzero values)\n", .{ slots.len, n_experts, nonzero });
    try testing.expect(nonzero > slots.len);
}

test "dsv41 kernels ops: the prefill wave route refuses by name, before any launch" {
    const a = testing.allocator;
    var reg = try testRegistry();
    defer reg.deinit();
    var diag: xk.Diag = .{};
    const tier = PrefillShape.tier;
    var bads = [_]PrefillShape{ tier, tier, tier, tier };
    bads[0].wave = 0;
    bads[1].wave = 17;
    bads[2].inflight = 1;
    bads[3].row_budget = 0;
    for (bads) |s| try testing.expectError(error.RouteInput, DigXPrefill(Trace).init(a, &reg, s, &diag));
    var t: Trace = .{ .a = a };
    defer t.deinit();
    var r = try DigXPrefill(Trace).init(a, &reg, tier, &diag);
    defer r.deinit(&t);
    const bank = try testBank(&t, 8);
    try testing.expectEqual(@as(usize, 0), t.launches.items.len);
    try testing.expectEqual(@as(usize, 0), t.log.items.len);
    // a bank above the kernels' slot bound is refused where the model binds it
    const big: ProjArrays(Trace.T) = .{ .code = try t.ext("c", &.{ 4097, 320, 144, 48 }, .int16), .rout = try t.ext("r", &.{ 4097, 2304 }, .float16), .rin = try t.ext("i", &.{ 4097, 5120 }, .float16) };
    try testing.expectError(error.RouteInput, checkBank(Trace, &t, &reg, .gate, big, &diag));
    try testing.expect(std.mem.indexOf(u8, diag.message(), "a bank of 4097 slots") != null);
    try checkBank(Trace, &t, &reg, .gate, bank.gate, &diag);
    try checkBank(Trace, &t, &reg, .down, bank.down, &diag);
}

test "dsv41 kernels ops: prefill rows read by act_row take the same act words (tokens, position / top_k)" {
    const a = testing.allocator;
    var reg = try testRegistry();
    defer reg.deinit();
    const counts = [_]u32{ 900, 700, 300, 297, 40, 1 };
    const slots = try routeRows(a, 77, &.{ 5, 0, 2, 7, 3, 6 }, &counts);
    defer a.free(slots);
    const n = slots.len;
    try testing.expectEqual(@as(usize, 2238), n);
    const act_row = try a.alloc(u32, n);
    defer a.free(act_row);
    for (act_row, 0..) |*v, i| v.* = @intCast(i / 6);
    var ta: Trace = .{ .a = a };
    defer ta.deinit();
    var tb: Trace = .{ .a = a };
    defer tb.deinit();
    var ra = try DigXPrefill(Trace).init(a, &reg, .tier, null);
    defer ra.deinit(&ta);
    var rb = try DigXPrefill(Trace).init(a, &reg, .tier, null);
    defer rb.deinit(&tb);
    _ = try ra.call(&ta, try ta.ext("act", &.{ @intCast(n), 5120 }, .bfloat16), .{ .slot = slots }, try testBank(&ta, 8));
    _ = try rb.call(&tb, try tb.ext("tokens", &.{ @intCast(n / 6), 5120 }, .bfloat16), .{ .slot = slots, .act_row = act_row }, try testBank(&tb, 8));
    try testing.expectEqual(ta.launches.items.len, tb.launches.items.len);
    try testing.expectEqual(ta.log.items.len, tb.log.items.len);
    for (ta.launches.items, tb.launches.items) |la, lb| {
        try testing.expectEqual(la.k, lb.k);
        try testing.expectEqual(la.cfg.grid, lb.cfg.grid);
        if (la.k != .dsv41_prefill_dig_take2v_5120) continue;
        // ridx: assignment row p in A reads act row p; in B, act_row[p] = p / 6
        const ba, const bb = .{ ta.nodes.items[la.inputs[1]].bytes, tb.nodes.items[lb.inputs[1]].bytes };
        try testing.expectEqual(ba.len, bb.len);
        var k: usize = 0;
        while (k < ba.len) : (k += 4) try testing.expectEqual(@divTrunc(std.mem.readInt(i32, ba[k..][0..4], .little), 6), std.mem.readInt(i32, bb[k..][0..4], .little));
    }
}

// The take2 retune's exactness argument, on the host in f32. The lane text's 128-point butterflies (lane l holds
// block elements l + 32q: element bits 0..4 across lanes, then 5 and 6 in registers) and the retune's (lane l holds
// 4l + q: bits 0 and 1 in registers, then 2..6 across lanes) combine the same pairs in the same bit order, each as
// (lower + upper, lower - upper), so every output is the same f32. A reordered stage fails here.
test "dsv41 kernels ops: the take2 retune's butterflies are the lane text's, bit for bit (both lane layouts emulated on the host)" {
    const Emu = struct {
        const scale: f32 = @bitCast(@as(u32, 1035273459));
        /// One simd_shuffle_xor stage over the 32 lanes: a lower lane keeps own + partner, an upper one partner - own.
        fn lanes(w: *[32][4]f32, h: usize) void {
            const old = w.*;
            for (0..32) |l| for (0..4) |q| {
                const o = old[l ^ h][q];
                w[l][q] = if (l & h != 0) o - old[l][q] else old[l][q] + o;
            };
        }
        /// The two in-register stages over a lane's 4 values: q bit 0, then q bit 1.
        fn regs(v: [4]f32) [4]f32 {
            const a0 = v[0] + v[1];
            const a1 = v[0] - v[1];
            const a2 = v[2] + v[3];
            const a3 = v[2] - v[3];
            return .{ a0 + a2, a1 + a3, a0 - a2, a1 - a3 };
        }
        fn laneText(x: *const [128]f32) [128]f32 {
            var w: [32][4]f32 = undefined;
            for (0..32) |l| for (0..4) |q| {
                w[l][q] = x[l + 32 * q];
            };
            var h: usize = 1;
            while (h < 32) : (h <<= 1) lanes(&w, h);
            var out: [128]f32 = undefined;
            for (0..32) |l| {
                const v = regs(w[l]);
                for (0..4) |q| out[l + 32 * q] = v[q] * scale;
            }
            return out;
        }
        fn retune(x: *const [128]f32) [128]f32 {
            var w: [32][4]f32 = undefined;
            for (0..32) |l| w[l] = regs(x[4 * l ..][0..4].*);
            var h: usize = 1;
            while (h < 32) : (h <<= 1) lanes(&w, h);
            var out: [128]f32 = undefined;
            for (0..32) |l| for (0..4) |q| {
                out[4 * l + q] = w[l][q] * scale;
            };
            return out;
        }
    };
    var prng = std.Random.DefaultPrng.init(0x7a4e2);
    const rnd = prng.random();
    var x: [128]f32 = undefined;
    for (0..4000) |i| {
        for (&x, 0..) |*v, j| v.* = switch (i % 4) {
            // a wide dynamic range: cancellation and absorption at every stage
            0 => (rnd.float(f32) - 0.5) * std.math.pow(f32, 2.0, @floatFromInt(rnd.intRangeAtMost(i32, -24, 24))),
            // the products' own scale (act about 1 x rin about 0.05)
            1 => (rnd.float(f32) - 0.5) * 0.1,
            // exact cancellations: equal magnitudes, alternating signs
            2 => if (j % 2 == 0) 0.375 else -0.375,
            // one outlier per block over small values
            else => if (j == i % 128) 1.0e3 else 1.0e-3 * (rnd.float(f32) + 0.5),
        };
        const want = Emu.laneText(&x);
        const got = Emu.retune(&x);
        try testing.expectEqualSlices(u32, @as(*const [128]u32, @ptrCast(&want)), @as(*const [128]u32, @ptrCast(&got)));
    }
}

// The 128-row DIG-X GEMMs' exactness argument, on the host: the texts' (threadgroup, simdgroup) -> output-block
// mapping, emulated from their own index arithmetic over the wave tables. At 64 rows per M tile (4 simdgroups) and at
// 128 (8), the blocks with rows to compute are the same list (op, first row, valid rows, 32-column block), and each
// (op, column block) tiles the wave's rows exactly once. Each stage's decode jobs cover the 16 x 16 trellis tiles'
// 256 (tile, column half, column) jobs exactly once at either simdgroup count. The per-simdgroup text is unchanged, so
// the blocks' words are the 64-row texts'.
test "dsv41 kernels ops: the 128-row DIG-X GEMMs compute every output block as the 64-row texts do (row-to-simdgroup mapping, host emulation)" {
    const a = testing.allocator;
    const Block = struct { op: u32, row: u32, valid: u32, col: u32 };
    const Emu = struct {
        /// The texts' mapping over one table: every (threadgroup, simdgroup) that has rows to compute.
        fn blocks(al: Allocator, tbl: *const [80]i32, tgs: u32, bm: i32, nsg: i32, ntiles: i32, out: *std.ArrayList(Block)) !void {
            const sm: i32 = 32;
            const sn: i32 = 32;
            const wn: i32 = 2;
            for (0..tgs) |gi| {
                const g: i32 = @intCast(gi);
                var j: usize = 0;
                for (1..16) |q| j = if (tbl[48 + q] <= g) q else j;
                const row0 = tbl[16 + j];
                const rows = tbl[32 + j];
                const local = g - tbl[48 + j];
                const tiles_m = @divTrunc(rows + bm - 1, bm);
                const nt_all = @divTrunc(local, tiles_m);
                const m0 = (local - nt_all * tiles_m) * bm;
                const op = @divTrunc(nt_all, ntiles);
                const n0 = (nt_all - op * ntiles) * 64;
                var sg: i32 = 0;
                while (sg < nsg) : (sg += 1) {
                    const tm = sm * @divTrunc(sg, wn);
                    const tn = sn * @mod(sg, wn);
                    const valid = @min(sm, @max(0, @min(bm, rows - m0) - tm));
                    if (valid > 0) try out.append(al, .{ .op = @intCast(op), .row = @intCast(row0 + m0 + tm), .valid = @intCast(valid), .col = @intCast(n0 + tn) });
                }
            }
        }
        fn lessThan(_: void, x: Block, y: Block) bool {
            if (x.op != y.op) return x.op < y.op;
            if (x.col != y.col) return x.col < y.col;
            return x.row < y.row;
        }
    };
    const Gemm = struct { tiles: u32, ntiles: i32, n_ops: u32, cols: u32 };
    const gemms = [_]Gemm{ .{ .tiles = 72, .ntiles = 36, .n_ops = 2, .cols = 2304 }, .{ .tiles = 80, .ntiles = 80, .n_ops = 1, .cols = 5120 } };
    const waves = [_][]const u32{ &.{ 1, 17, 64, 127, 128, 129, 270 }, &.{ 256, 256, 256, 256, 256, 256, 256, 256 }, &.{ 270, 270, 270, 270 }, &.{ 32, 32, 32, 32, 32, 32, 32, 32 } };
    var n_blocks: usize = 0;
    for (waves) |per| for (gemms) |gm| {
        var ex: [wave_max]WaveExpert = undefined;
        var total: u32 = 0;
        for (per, 0..) |r, j| {
            ex[j] = .{ .slot = @intCast(j), .rows = r };
            total += r;
        }
        var lists: [2]std.ArrayList(Block) = .{ .empty, .empty };
        defer for (&lists) |*l| l.deinit(a);
        for ([_]u32{ 64, 128 }, 0..) |bm, i| {
            const t = digTableBm(ex[0..per.len], gm.tiles, bm);
            try Emu.blocks(a, &t.table, t.tgs, @intCast(bm), if (bm == 64) 4 else 8, gm.ntiles, &lists[i]);
            std.mem.sort(Block, lists[i].items, {}, Emu.lessThan);
        }
        try testing.expectEqualSlices(Block, lists[0].items, lists[1].items);
        // each (op, column block) tiles the rows [0, total) exactly once
        var at: usize = 0;
        for (0..gm.n_ops) |op| {
            var col: u32 = 0;
            while (col < gm.cols) : (col += 32) {
                var next: u32 = 0;
                while (at < lists[0].items.len and lists[0].items[at].op == op and lists[0].items[at].col == col) : (at += 1) {
                    try testing.expectEqual(next, lists[0].items[at].row);
                    next += lists[0].items[at].valid;
                }
                try testing.expectEqual(total, next);
            }
        }
        try testing.expectEqual(lists[0].items.len, at);
        n_blocks += at;
    };
    try testing.expect(n_blocks > 10_000);
    // each stage's decode jobs: the 256 (trellis tile, column half, column) jobs exactly once, at 4 and 8 simdgroups
    for ([_]u32{ 4, 8 }) |nsg| {
        var seen: [256]u8 = @splat(0);
        const jobs = 8 / nsg;
        for (0..nsg) |sg| for (0..32) |lane| for (0..jobs) |jj| {
            const t = (lane >> 3) + 4 * ((sg >> 1) + (nsg / 2) * jj);
            seen[(t * 2 + (sg & 1)) * 8 + (lane & 7)] += 1;
        };
        for (seen) |x| try testing.expectEqual(@as(u8, 1), x);
    }
}

// The fused down GEMM's exactness argument, on the host (1): its (threadgroup, simdgroup) -> 32 x 32 accumulator blocks
// are the 128-row down text's (the same per-simdgroup code, so the same z words); the epilogue writes every output word
// of the wave exactly once; each stage's decode jobs cover the 32 trellis tiles' 512 jobs exactly once at 16 simdgroups.
test "dsv41 kernels ops: the fused down GEMM computes the 128-row down text's blocks and writes each output word once (host emulation)" {
    const a = testing.allocator;
    const Block = struct { row: u32, valid: u32, col: u32 };
    const Emu = struct {
        /// The text's mapping at BN `bn` over `wn` column simdgroups: every (threadgroup, simdgroup) with rows to compute.
        fn blocks(al: Allocator, tbl: *const [80]i32, tgs: u32, bn: i32, wn: i32, nsg: i32, ntiles: i32, out: *std.ArrayList(Block)) !void {
            for (0..tgs) |gi| {
                const g: i32 = @intCast(gi);
                var j: usize = 0;
                for (1..16) |q| j = if (tbl[48 + q] <= g) q else j;
                const row0 = tbl[16 + j];
                const rows = tbl[32 + j];
                const local = g - tbl[48 + j];
                const tiles_m = @divTrunc(rows + 127, 128);
                const nt_all = @divTrunc(local, tiles_m);
                const m0 = (local - nt_all * tiles_m) * 128;
                const n0 = @mod(nt_all, ntiles) * bn;
                var sg: i32 = 0;
                while (sg < nsg) : (sg += 1) {
                    const tm = 32 * @divTrunc(sg, wn);
                    const valid = @min(32, @max(0, @min(128, rows - m0) - tm));
                    if (valid > 0) try out.append(al, .{ .row = @intCast(row0 + m0 + tm), .valid = @intCast(valid), .col = @intCast(n0 + 32 * @mod(sg, wn)) });
                }
            }
        }
        fn lessThan(_: void, x: Block, y: Block) bool {
            return if (x.col != y.col) x.col < y.col else x.row < y.row;
        }
    };
    const waves = [_][]const u32{ &.{ 1, 17, 64, 127, 128, 129, 270 }, &.{ 256, 256, 256, 256, 256, 256, 256, 256 }, &.{ 33, 160, 97 } };
    for (waves) |per| {
        var ex: [wave_max]WaveExpert = undefined;
        var total: u32 = 0;
        for (per, 0..) |r, j| {
            ex[j] = .{ .slot = @intCast(j), .rows = r };
            total += r;
        }
        var lists: [2]std.ArrayList(Block) = .{ .empty, .empty };
        defer for (&lists) |*l| l.deinit(a);
        const t128 = digTableBm(ex[0..per.len], 80, 128);
        const tw = digTableBm(ex[0..per.len], 40, 128);
        try Emu.blocks(a, &t128.table, t128.tgs, 64, 2, 8, 80, &lists[0]);
        try Emu.blocks(a, &tw.table, tw.tgs, 128, 4, 16, 40, &lists[1]);
        for (&lists) |*l| std.mem.sort(Block, l.items, {}, Emu.lessThan);
        try testing.expectEqualSlices(Block, lists[0].items, lists[1].items);
        // The epilogue's passes: tile rows 32 at a time, rows sg and sg + 16 per simdgroup, columns n0 + 4 lane + q.
        const seen = try a.alloc(u8, @as(usize, total) * 5120);
        defer a.free(seen);
        @memset(seen, 0);
        for (0..tw.tgs) |gi| {
            const g: i32 = @intCast(gi);
            var j: usize = 0;
            for (1..16) |q| j = if (tw.table[48 + q] <= g) q else j;
            const rows = tw.table[32 + j];
            const local = g - tw.table[48 + j];
            const tiles_m = @divTrunc(rows + 127, 128);
            const nt_all = @divTrunc(local, tiles_m);
            const m0 = (local - nt_all * tiles_m) * 128;
            const n0: usize = @intCast(@mod(nt_all, 40) * 128);
            const valid = @min(128, rows - m0);
            var p: i32 = 0;
            while (p * 32 < valid) : (p += 1) for (0..16) |sg| {
                var rr: i32 = @intCast(sg);
                while (rr < 32 and p * 32 + rr < valid) : (rr += 16) {
                    const row: usize = @intCast(tw.table[16 + j] + m0 + p * 32 + rr);
                    for (0..32) |lane| for (0..4) |q| {
                        seen[row * 5120 + n0 + 4 * lane + q] += 1;
                    };
                }
            };
        }
        for (seen) |x| try testing.expectEqual(@as(u8, 1), x);
    }
    // each stage's decode jobs at 16 simdgroups and 8 column tiles: the 512 (trellis tile, column half, column) jobs once
    var seen: [512]u8 = @splat(0);
    const nt = 8;
    const jobs = 4 * nt / (2 * 16);
    for (0..16) |sg| for (0..32) |lane| for (0..jobs) |jj| {
        const t = (lane >> 3) + 4 * ((sg >> 1) + 8 * jj);
        seen[(t * 2 + (sg & 1)) * 8 + (lane & 7)] += 1;
    };
    for (seen) |x| try testing.expectEqual(@as(u8, 1), x);
}

// The fused down GEMM's exactness argument, on the host (2): its epilogue (lane l holds columns 4 l .. 4 l + 3; bits 0
// and 1 in registers, then 2 .. 6 across lanes) is rot_widen1's arithmetic (lane l holds l + 32 q; bits 0 .. 4 across
// lanes, then 5 and 6 in registers): the same pairs in the same bit order, then (w * SCALE) * rout, every word equal.
test "dsv41 kernels ops: the fused down GEMM's epilogue is rot_widen1's arithmetic, bit for bit (both lane layouts emulated, rout included)" {
    const Emu = struct {
        const scale: f32 = @bitCast(@as(u32, 1035273459));
        fn lanes(w: *[32][4]f32, h: usize) void {
            const old = w.*;
            for (0..32) |l| for (0..4) |q| {
                const o = old[l ^ h][q];
                w[l][q] = if (l & h != 0) o - old[l][q] else old[l][q] + o;
            };
        }
        fn regs(v: [4]f32) [4]f32 {
            const a0 = v[0] + v[1];
            const a1 = v[0] - v[1];
            const a2 = v[2] + v[3];
            const a3 = v[2] - v[3];
            return .{ a0 + a2, a1 + a3, a0 - a2, a1 - a3 };
        }
        /// rot_widen1: column l + 32 q in lane l.
        fn widen1(z: *const [128]f32, ro: *const [128]f16) [128]f32 {
            var w: [32][4]f32 = undefined;
            for (0..32) |l| for (0..4) |q| {
                w[l][q] = z[l + 32 * q];
            };
            var h: usize = 1;
            while (h < 32) : (h <<= 1) lanes(&w, h);
            var out: [128]f32 = undefined;
            for (0..32) |l| {
                const v = regs(w[l]);
                for (0..4) |q| out[l + 32 * q] = (v[q] * scale) * @as(f32, ro[l + 32 * q]);
            }
            return out;
        }
        /// The fused epilogue: columns 4 l .. 4 l + 3 in lane l.
        fn fused(z: *const [128]f32, ro: *const [128]f16) [128]f32 {
            var w: [32][4]f32 = undefined;
            for (0..32) |l| w[l] = regs(z[4 * l ..][0..4].*);
            var h: usize = 1;
            while (h < 32) : (h <<= 1) lanes(&w, h);
            var out: [128]f32 = undefined;
            for (0..32) |l| for (0..4) |q| {
                out[4 * l + q] = (w[l][q] * scale) * @as(f32, ro[4 * l + q]);
            };
            return out;
        }
    };
    var prng = std.Random.DefaultPrng.init(0x3e1d7);
    const rnd = prng.random();
    var z: [128]f32 = undefined;
    var ro: [128]f16 = undefined;
    for (0..4000) |i| {
        for (&z, &ro, 0..) |*v, *r, j| {
            v.* = switch (i % 3) {
                // the down GEMM's own scale, then a wide dynamic range, then exact cancellations
                0 => (rnd.float(f32) - 0.5) * 8.0,
                1 => (rnd.float(f32) - 0.5) * std.math.pow(f32, 2.0, @floatFromInt(rnd.intRangeAtMost(i32, -24, 24))),
                else => if (j % 2 == 0) 0.375 else -0.375,
            };
            r.* = @floatCast(0.02 + 0.06 * rnd.float(f32));
        }
        const want = Emu.widen1(&z, &ro);
        const got = Emu.fused(&z, &ro);
        try testing.expectEqualSlices(u32, @as(*const [128]u32, @ptrCast(&want)), @as(*const [128]u32, @ptrCast(&got)));
    }
}

// The table-codebook gate|up text's exactness argument, on the host. dig_cb's half for a 16-bit window w is the half fma
// of h(bits 0x6400 + s), h(0x1EEE) and h(0xC931), s the byte sum of w * 2212286765 (the upper half of t * 0x10001 + 0x64000000,
// t its two byte pairs' sums); the table's entry s is the same fma of the same three halves, filled for s = 0 .. 1023. So the
// text reads dig_cb's value when it indexes the table at the upper half of t * 0x10001: checked over every window, the fma
// emulated (the exact product and sum in f64, rounded once to f16).
test "dsv41 kernels ops: the table-codebook text indexes dig_cb's half for every 16-bit window" {
    const c1: f16 = @bitCast(@as(u16, 0x1EEE));
    const c2: f16 = @bitCast(@as(u16, 0xC931));
    var lut: [1024]f16 = undefined;
    for (&lut, 0..) |*v, i| {
        const hs: f16 = @bitCast(@as(u16, @intCast(0x6400 + i)));
        v.* = @floatCast(@as(f64, hs) * @as(f64, c1) + @as(f64, c2));
    }
    var max_s: u32 = 0;
    for (0..65536) |wi| {
        const x: u32 = @as(u32, @intCast(wi)) *% 2212286765;
        const t = (x & 0x00ff00ff) +% ((x >> 8) & 0x00ff00ff);
        const hs: f16 = @bitCast(@as(u16, @truncate((t *% 0x00010001 +% 0x64000000) >> 16)));
        const want: f16 = @floatCast(@as(f64, hs) * @as(f64, c1) + @as(f64, c2));
        const s = (t *% 0x00010001) >> 16;
        try testing.expect(s < lut.len);
        max_s = @max(max_s, s);
        try testing.expectEqual(@as(u16, @bitCast(want)), @as(u16, @bitCast(lut[s])));
    }
    try testing.expect(max_s <= 1020);
}

// ── 3. Move invariance: the C2 entries launch what today's EXL3 entries launch ──

/// A trace's log from `from` on as text: every launch with its kernel, grid, threadgroup,
/// template, inputs (by origin), outputs and prepared flag; evals, joins and graph ops as
/// `kt.traceEvent` renders them.
fn renderLog(t: *const Trace, from: usize, out: *std.ArrayList(u8)) !void {
    const a = t.a;
    for (t.log.items[from..]) |e| {
        switch (e) {
            .launch => |li| {
                const l = &t.launches.items[li];
                const c = &l.cfg;
                try out.print(a, "launch {t} g={d},{d},{d} t={d},{d},{d} prepared={} tmpl=", .{ l.k, c.grid[0], c.grid[1], c.grid[2], c.threadgroup[0], c.threadgroup[1], c.threadgroup[2], l.prepared });
                for (c.template) |x| switch (x.value) {
                    .int => |v| try out.print(a, "{s}:{d},", .{ x.name, v }),
                    .dtype => |v| try out.print(a, "{s}:{t},", .{ x.name, v }),
                };
                try out.appendSlice(a, " in=");
                try traceRefs(t, l.inputs[0..l.n_in], out);
                try out.appendSlice(a, " out=");
                for (0..c.n_out) |i| {
                    try out.print(a, "{t}", .{c.out_dtypes[i]});
                    try shapeStr(out, a, c.out_shapes[i][0..c.out_ranks[i]]);
                }
            },
            else => try traceEvent(t, e, out),
        }
        try out.append(a, '\n');
    }
}

/// The EXL3 slot bank as named caller arrays (their references render by name).
fn namedBank(t: *Trace, cap: c_int) !BankArrays(Trace.T) {
    return testBank(t, cap);
}

/// What today's EXL3 entries (exl3_kernel_ops.zig at e777dc5: RinChain's composition over its Gemv
/// and RinPrep, DigXPrefill call / finish) launched over the cases below, rendered by `renderLog`:
/// compared with the C2 entries byte for byte at the move commit aad8e67, then pinned here when the
/// old file was deleted.
const moved_log_lines = 1021;
const moved_log_sha256 = "e1f27114d1cbda8b71a4c9e990754c151bdabf1b9bf3075bc8aa61747578657d";

/// The move's cases on `acc` (accepted on `t`), from `t`'s log end: decode at every M, then the lane samples' prefill
/// calls at the lane's shape (Record 3: the move's pinned launches) and the boundary. Appends the log in the lane's
/// text (`renderLog`, then `LaneMap.lane`) to `out`; returns the map's take2 / 128-row GEMM / fused down counts.
fn movedLaneLog(a: Allocator, t: *Trace, reg: *const xk.Registry, acc: *Accepted(Trace), bb: BankArrays(Trace.T), out: *std.ArrayList(u8)) ![3]usize {
    var lb: std.ArrayList(u8) = .empty;
    defer lb.deinit(a);
    const b0 = t.log.items.len;
    for (1..49) |m| {
        const mc: c_int = @intCast(m);
        const xb, const ib = .{ try t.ext("x", &.{ mc, 5120 }, .bfloat16), try t.ext("ids", &.{mc}, .uint32) };
        const hb = try acc.gateUp(t, xb, ib, bb.gate, bb.up);
        _ = try acc.down(t, hb, ib, bb.down);
    }
    for (acc.waves) |*w| {
        w.deinit(t);
        w.* = try DigXPrefill(Trace).init(a, reg, PrefillShape.record3, null);
        if (acc.fused_down) w.installDown(.fused);
    }
    const parsed = try std.json.parseFromSlice(JSamples, a, prefill_samples, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    for (parsed.value.cases) |*cs| for (cs.calls) |*cl| {
        const slots = try routeRows(a, cl.route.seed, cl.route.slots, cl.route.counts);
        defer a.free(slots);
        for (slots) |*s| s.* %= 64;
        const n: c_int = @intCast(slots.len);
        const yb = try acc.prefill(t, 3, try t.ext("act", &.{ n, 5120 }, .bfloat16), .{ .slot = slots }, bb);
        t.release(yb);
    };
    try acc.finishPrefill(t);
    try renderLog(t, b0, &lb);
    var lane_arena = std.heap.ArenaAllocator.init(a);
    defer lane_arena.deinit();
    var lm = try LaneMap.init(a, t, reg);
    defer lm.deinit(a);
    var lines = std.mem.splitScalar(u8, lb.items, '\n');
    while (lines.next()) |line| {
        try out.appendSlice(a, try lm.lane(lane_arena.allocator(), line));
        if (lines.index != null) try out.append(a, '\n');
    }
    return .{ lm.n_take2, lm.n_gemm, lm.n_w1 };
}

test "dsv41 kernels c2: move invariance: gateUp / down / prefill / finishPrefill launch what today's EXL3 entries launched (pinned at the move)" {
    const a = testing.allocator;
    var diag: Diag = .{};
    var tb: Trace = .{ .a = a };
    defer tb.deinit();
    const set = try ks.Set.init(a, .{ .device = .{ .stub = .{} } }, &diag);
    defer set.deinit();
    set.install(Trace, &tb);
    const acc = try accept(Trace, a, &tb, .{ .kernels = set.ref() }, v41_spec, &diag);
    defer acc.deinit(&tb);
    const bb = try namedBank(&tb, 64);
    var lane_log: std.ArrayList(u8) = .empty;
    defer lane_log.deinit(a);
    // The route launches the take2 retune and the 128-row GEMMs (same inputs and output words; names, grids and the
    // wave tables' first-threadgroup column differ): their lines back in the lane's text, then every byte as pinned
    // at the move.
    const n = try movedLaneLog(a, &tb, &set.reg, acc, bb, &lane_log);
    try testing.expect(n[0] > 0 and n[1] == 2 * n[0] and n[2] == 0);
    try testing.expectEqual(@as(usize, moved_log_lines), std.mem.count(u8, lane_log.items, "\n"));
    try testing.expectEqualStrings(moved_log_sha256, &kset.shaHex(lane_log.items));
    // The fused arm: its self-checks join the report, and its waves launch the fused down GEMM (the down GEMM's and
    // rot_widen1's words in one launch; later launch indices one fewer per fused launch): the same pinned log.
    {
        var tf: Trace = .{ .a = a };
        defer tf.deinit();
        set.install(Trace, &tf);
        const accf = try accept(Trace, a, &tf, .{ .kernels = set.ref() }, v41_spec, &diag);
        defer accf.deinit(&tf);
        try accf.routeFusedDown(set, &diag);
        try testing.expect(accf.fused_down and !acc.fused_down);
        try testing.expectEqual(acc.report.results.items.len + w1_texts.len, accf.report.results.items.len); // its construction plan: one compile per text
        var fused_log: std.ArrayList(u8) = .empty;
        defer fused_log.deinit(a);
        const nf = try movedLaneLog(a, &tf, &set.reg, accf, try namedBank(&tf, 64), &fused_log);
        try testing.expect(nf[0] > 0 and nf[1] == nf[0] and nf[2] == nf[0]);
        try testing.expectEqualStrings(moved_log_sha256, &kset.shaHex(fused_log.items));
    }
    // checkBank: the moved per-projection check's refusals, through the quant's entry
    for ([_]struct { cap: c_int, last: c_int, rin_dt: Dtype, what: []const u8 }{
        .{ .cap = 1, .last = 40, .rin_dt = .float16, .what = "input code" },
        .{ .cap = 64, .last = 48, .rin_dt = .float32, .what = "input rg" },
        .{ .cap = 5000, .last = 48, .rin_dt = .float16, .what = "a bank of 5000 slots" },
    }) |c| {
        const arr = ProjArrays(Trace.T){ .code = try tb.ext("code", &.{ c.cap, 320, 144, c.last }, .int16), .rout = try tb.ext("rout", &.{ c.cap, 2304 }, .float16), .rin = try tb.ext("rin", &.{ c.cap, 5120 }, c.rin_dt) };
        try testing.expectError(error.RouteInput, acc.checkBank(&tb, .{ .gate = arr, .up = bb.up, .down = bb.down }, &diag));
        if (std.mem.indexOf(u8, diag.message(), c.what) == null) {
            std.debug.print("checkBank refused with: {s}\n", .{diag.message()});
            return error.TestUnexpectedResult;
        }
    }
    try acc.checkBank(&tb, bb, &diag);
}

// DSV41_PHASE0B_MLX=1 + DSV41_BANK=<bank>, on a GPU machine (any MLX array creates the Metal device); seconds,
// no model load. The take2 retune on the device, and the per-wave prices the drain note left open:
// 1. both take2 texts' construction self-checks on the device (compile, row invariance, the bitwise mlx_chain);
// 2. the retune against the lane's take2, each through its own entry (whichever the route launches), over real
//    records (layer 0's first 16 experts, their rin rows read by the stream), a K16 wave of 8 experts x 256 rows and
//    a ragged one (1 / 17 / 270 rows): every output word equal, bit for bit. Fail-closed: a mismatch, an all-zero
//    output, or DSV41_PHASE0B_MLX without the bank fails the test;
// 3. one DIGX_WAVE_MICROBENCH line per kernel and wave (take2 lane and retune, onepass, widen1) at the tier's
//    2,048-row wave and the DIG-X probe's 4 x 270 = 1,080: per launch, the best of 5 runs of 8 back-to-back launches
//    and one eval ("us": the eval's encode + GPU + wait; "host_us": the launches' graph build).
test "dsv41 smoke 0b: take2 retune: the lane's take2 words on real records, bit for bit; per-wave microbench of take2, onepass, widen1" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_BANK) orelse {
        std.debug.print("\ntake2 retune smoke: DSV41_PHASE0B_MLX without DSV41_BANK (the real records): refused\n", .{});
        return error.TestUnexpectedResult;
    });
    const ops = @import("../../../weights/streamed/ops.zig");
    const es = @import("../../../weights/streamed/expert/stream.zig");
    const eb = @import("../../../weights/streamed/expert/bank.zig");
    const G = ops.MlxOps;
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var kd: xk.Diag = .{};
    const set = ks.Set.init(a, .{ .device = .{ .stream = s } }, &kd) catch |e| {
        std.debug.print("kernel set refused: {s}\n", .{kd.message()});
        return e;
    };
    defer set.deinit();
    set.install(G, &g);
    defer ks.Set.uninstall(G, &g);
    // 1. both texts' construction self-checks, on the device
    {
        var report: selfcheck.Report = .{};
        defer report.deinit(a);
        set.selfCheck(a, &.{ .q3_prefill_dig_rot_take2_5120, .dsv41_prefill_dig_take2v_5120 }, .full, &report, &kd) catch |e| {
            std.debug.print("take2 self-checks refused: {s}\n", .{kd.message()});
            return e;
        };
        var chains: usize = 0;
        for (report.results.items) |r| chains += @intFromBool(r.check == .mlx_chain and r.ok and r.bad == 0 and r.words > 0);
        try testing.expectEqual(@as(usize, 2), chains);
    }
    // layer 0's first 16 experts, read by the stream into 16 persistent MLX rows (the base bank)
    var bdiag: eb.Diag = .{};
    var bank = eb.Bank.open(a, io, dir, eb.dsv41, &bdiag) catch |e| {
        std.debug.print("bank refused: {s}\n", .{bdiag.message()});
        return e;
    };
    defer bank.deinit();
    const n_experts = 16;
    var base_rows: [40]u32 = @splat(0);
    base_rows[0] = n_experts;
    const st = try es.Stream.init(a, &bank, .{ .rows = &base_rows, .max_route_ids = n_experts, .transient_rows = n_experts, .slot_memory = .{ .mlx = s } });
    defer st.deinit();
    var ids: [n_experts]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(i);
    const route = try st.route(0, &ids, &.{});
    defer st.release(route);
    for (0..route.n_parts) |p| {
        try st.waitGu(route, @intCast(p));
        try st.waitDown(route, @intCast(p));
    }
    var refs: [@import("../../../weights/streamed/sdk_ext.zig").expert.policy.max_route_ids]es.SlotRef = undefined;
    const rf = st.refsOf(route, &refs);
    try testing.expectEqual(@as(usize, n_experts), rf.len);
    const ba = st.bankArrays(0, .base) orelse return error.TestUnexpectedResult;
    const dig = DigX(G).init(&set.reg);
    const Wave = struct {
        rows: u32,
        act: G.T,
        ridx: G.T,
        rhs: G.T,
        tbl_gu: G.T,
        tbl_dn: G.T,

        /// A wave's rows grouped by expert in order (expert j at real record `refs_[j]`), act row i for row i, act a
        /// deterministic spread of bf16 values (a sum of two uniforms in [-2, 2)).
        fn init(al: Allocator, gg: *G, per: []const u32, refs_: []const es.SlotRef, d: DigX(G), seed: u64) !@This() {
            var ex: [wave_max]WaveExpert = undefined;
            var n: u32 = 0;
            for (per, 0..) |c, j| {
                ex[j] = .{ .slot = refs_[j].row, .rows = c };
                n += c;
            }
            const ridx = try al.alloc(i32, n);
            defer al.free(ridx);
            const rhs = try al.alloc(u32, n);
            defer al.free(rhs);
            var i: usize = 0;
            for (per, 0..) |c, j| for (0..c) |_| {
                ridx[i] = @intCast(i);
                rhs[i] = @intCast(j);
                i += 1;
            };
            const xs = try al.alloc(f32, @as(usize, n) * 5120);
            defer al.free(xs);
            var h: u64 = seed;
            for (xs) |*v| {
                h = h *% 6364136223846793005 +% 1442695040888963407;
                const u_1: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(h >> 40)))) / 16777216.0;
                const u_2: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(h >> 16)) & 0xffffff)) / 16777216.0;
                v.* = (u_1 + u_2 - 1.0) * 2.0;
            }
            const tg = digTableBm(ex[0..per.len], d.digTiles(.gate_up), 64);
            const td = digTableBm(ex[0..per.len], d.digTiles(.down), 64);
            const ni: c_int = @intCast(n);
            return .{
                .rows = n,
                .act = try gg.astype(try gg.hostArray(std.mem.sliceAsBytes(xs), &.{ ni, 5120 }, .float32), .bfloat16),
                .ridx = try gg.hostArray(std.mem.sliceAsBytes(ridx), &.{ni}, .int32),
                .rhs = try gg.hostArray(std.mem.sliceAsBytes(rhs), &.{ni}, .uint32),
                .tbl_gu = try gg.hostArray(std.mem.sliceAsBytes(&tg.table), &.{80}, .int32),
                .tbl_dn = try gg.hostArray(std.mem.sliceAsBytes(&td.table), &.{80}, .int32),
            };
        }
    };
    const k16 = [_]u32{ 256, 256, 256, 256, 256, 256, 256, 256 };
    // 2. the retune against the lane's take2 on the real rin rows: every output word, bit for bit
    for ([_][]const u32{ &k16, &.{ 1, 17, 270 } }, 0..) |per, wi| {
        const m = g.mark();
        defer g.resetTo(m);
        const w = try Wave.init(a, &g, per, rf, dig, 0x5eed + wi);
        const vars = rowsVars(w.rows);
        var lane: [2]G.T = undefined;
        try launchRule(G, &g, dig.take2_e, &vars, &.{ w.act, w.ridx, w.rhs, w.tbl_gu, ba.gate.rin, ba.up.rin }, &lane);
        var retuned: [2]G.T = undefined;
        try launchRule(G, &g, dig.take2v_e, &vars, &.{ w.act, w.ridx, w.rhs, w.tbl_gu, ba.gate.rin, ba.up.rin }, &retuned);
        try g.evalAll(&.{ lane[0], lane[1], retuned[0], retuned[1] });
        const n_words = @as(usize, w.rows) * 5120;
        var mismatch: usize = 0;
        var nonzero: usize = 0;
        for (lane, retuned) |x, y| {
            const px = mlx.mlx_array_data_uint8(x) orelse return error.TestUnexpectedResult;
            const py = mlx.mlx_array_data_uint8(y) orelse return error.TestUnexpectedResult;
            for (0..n_words) |k| {
                const wx = @as(u16, px[2 * k]) | @as(u16, px[2 * k + 1]) << 8;
                const wy = @as(u16, py[2 * k]) | @as(u16, py[2 * k + 1]) << 8;
                mismatch += @intFromBool(wx != wy);
                nonzero += @intFromBool(wx & 0x7fff != 0);
            }
        }
        std.debug.print("\nTAKE2_RETUNE_SMOKE {{\"rows\": {d}, \"experts\": {d}, \"words\": {d}, \"mismatch\": {d}, \"nonzero\": {d}}}\n", .{ w.rows, per.len, 2 * n_words, mismatch, nonzero });
        try testing.expectEqual(@as(usize, 0), mismatch);
        try testing.expect(nonzero > n_words);
    }
    // 3. per-wave time, one line per kernel and wave
    for ([_][]const u32{ &k16, &.{ 270, 270, 270, 270 } }) |per| {
        const m = g.mark();
        defer g.resetTo(m);
        const w = try Wave.init(a, &g, per, rf, dig, 0xbe7c);
        const n: usize = w.rows;
        const ni: c_int = @intCast(n);
        // the GEMM outputs' stand-ins (onepass reads z_g / z_u, widen1 z_d), uniform in [-3, 3)
        const zs = try a.alloc(f32, n * 5120);
        defer a.free(zs);
        var h: u64 = 0x2d;
        for (zs) |*v| {
            h = h *% 6364136223846793005 +% 1442695040888963407;
            v.* = (@as(f32, @floatFromInt(@as(u32, @truncate(h >> 40)))) / 16777216.0 - 0.5) * 6.0;
        }
        const zg = try g.hostArray(std.mem.sliceAsBytes(zs[0 .. n * 2304]), &.{ ni, 2304 }, .float32);
        const zu = try g.hostArray(std.mem.sliceAsBytes(zs[n * 2304 .. n * 4608]), &.{ ni, 2304 }, .float32);
        const zd = try g.hostArray(std.mem.sliceAsBytes(zs), &.{ ni, 5120 }, .float32);
        try g.evalAll(&.{ w.act, w.ridx, w.rhs, w.tbl_gu, w.tbl_dn, zg, zu, zd });
        const vars = rowsVars(w.rows);
        const Kind = enum { take2_lane, take2_retune, onepass, widen1 };
        for ([_]Kind{ .take2_lane, .take2_retune, .onepass, .widen1 }) |kind| {
            var best_eval: u64 = std.math.maxInt(u64);
            var best_host: u64 = std.math.maxInt(u64);
            // run 0 warms (the pipeline's first compile); runs 1..5 are timed
            for (0..6) |run| {
                const mr = g.mark();
                defer g.resetTo(mr);
                var outs: [16]G.T = undefined;
                var n_out: usize = 0;
                const t0 = std.Io.Timestamp.now(io, .boot);
                for (0..8) |_| switch (kind) {
                    .take2_lane => {
                        var o: [2]G.T = undefined;
                        try launchRule(G, &g, dig.take2_e, &vars, &.{ w.act, w.ridx, w.rhs, w.tbl_gu, ba.gate.rin, ba.up.rin }, &o);
                        outs[n_out] = o[0];
                        outs[n_out + 1] = o[1];
                        n_out += 2;
                    },
                    .take2_retune => {
                        var o: [2]G.T = undefined;
                        try launchRule(G, &g, dig.take2v_e, &vars, &.{ w.act, w.ridx, w.rhs, w.tbl_gu, ba.gate.rin, ba.up.rin }, &o);
                        outs[n_out] = o[0];
                        outs[n_out + 1] = o[1];
                        n_out += 2;
                    },
                    .onepass => {
                        outs[n_out] = try dig.onePass(&g, zg, zu, w.rhs, w.tbl_gu, ba.gate.rout, ba.up.rout, ba.down.rin);
                        n_out += 1;
                    },
                    .widen1 => {
                        outs[n_out] = try dig.widen1(&g, zd, w.rhs, w.tbl_dn, ba.down.rout);
                        n_out += 1;
                    },
                };
                const host: u64 = @intCast(t0.untilNow(io, .boot).nanoseconds);
                const t1 = std.Io.Timestamp.now(io, .boot);
                try g.evalAll(outs[0..n_out]);
                const eval: u64 = @intCast(t1.untilNow(io, .boot).nanoseconds);
                if (run > 0) {
                    best_eval = @min(best_eval, eval);
                    best_host = @min(best_host, host);
                }
            }
            const kname = switch (kind) {
                .take2_lane => @tagName(Kernel.q3_prefill_dig_rot_take2_5120),
                .take2_retune => @tagName(Kernel.dsv41_prefill_dig_take2v_5120),
                .onepass => @tagName(Kernel.q3_prefill_dig2_swiglu_2304_x),
                .widen1 => @tagName(Kernel.q3_prefill_dig_rot_widen1_5120),
            };
            std.debug.print("DIGX_WAVE_MICROBENCH {{\"kernel\": \"{s}\", \"rows\": {d}, \"us\": {d:.1}, \"host_us\": {d:.1}}}\n", .{ kname, w.rows, @as(f64, @floatFromInt(best_eval)) / 8000.0, @as(f64, @floatFromInt(best_host)) / 8000.0 });
        }
    }
}

// DSV41_PHASE0B_MLX=1 + DSV41_BANK=<bank>, on a GPU machine (any MLX array creates the Metal device); seconds,
// no model load. The 128-row DIG-X GEMMs on the device, and the decode share's price:
// 1. both 128-row texts' construction self-checks on the device (compile, composition, f64, twin);
// 2. the 64- and 128-row texts on real records (layer 0's first 16 experts' codes, read by the stream), each wave's
//    x / hd from the routed take2 / gate|up / onepass chain: a wave at every partial-tile edge (1 / 17 / 64 / 127 /
//    128 / 129 / 270 rows) and a K16 wave (8 x 256), every z word of gate, up and down equal, bit for bit.
//    Fail-closed: a mismatch, an all-zero output, or DSV41_PHASE0B_MLX without the bank fails the test;
// 3. DIGX_WAVE_MICROBENCH lines for both texts' GEMMs at the tier's 2,048-row wave and the DIG-X probe's 4 x 270 =
//    1,080, and DIGX_GEMM_SWEEP lines for today's texts at 8 experts x 32 / 64 / 256 rows (the decode share's fit:
//    per 64-row tile T64 / 8 = d + m, T32 / 8 = d + m / 2). Per launch: the best of 5 runs of 8 back-to-back
//    launches and one eval ("us": the eval's encode + GPU + wait; "host_us": the launches' graph build).
test "dsv41 smoke 0b: m128: the 128-row DIG-X GEMMs' z words equal the 64-row texts' on real records, bit for bit; GEMM microbench and sweep" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_BANK) orelse {
        std.debug.print("\nm128 smoke: DSV41_PHASE0B_MLX without DSV41_BANK (the real records): refused\n", .{});
        return error.TestUnexpectedResult;
    });
    const ops = @import("../../../weights/streamed/ops.zig");
    const es = @import("../../../weights/streamed/expert/stream.zig");
    const eb = @import("../../../weights/streamed/expert/bank.zig");
    const G = ops.MlxOps;
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var kd: xk.Diag = .{};
    const set = ks.Set.init(a, .{ .device = .{ .stream = s } }, &kd) catch |e| {
        std.debug.print("kernel set refused: {s}\n", .{kd.message()});
        return e;
    };
    defer set.deinit();
    set.install(G, &g);
    defer ks.Set.uninstall(G, &g);
    // 1. both 128-row texts' construction self-checks, on the device
    {
        var report: selfcheck.Report = .{};
        defer report.deinit(a);
        set.selfCheck(a, &m128_texts, .full, &report, &kd) catch |e| {
            std.debug.print("m128 self-checks refused: {s}\n", .{kd.message()});
            return e;
        };
        var twins: usize = 0;
        for (report.results.items) |r| twins += @intFromBool(r.check == .twin and r.ok and r.bad == 0 and r.words > 0);
        try testing.expectEqual(@as(usize, 2), twins);
    }
    // layer 0's first 16 experts, read by the stream into 16 persistent MLX rows (the base bank)
    var bdiag: eb.Diag = .{};
    var bank = eb.Bank.open(a, io, dir, eb.dsv41, &bdiag) catch |e| {
        std.debug.print("bank refused: {s}\n", .{bdiag.message()});
        return e;
    };
    defer bank.deinit();
    const n_experts = 16;
    var base_rows: [40]u32 = @splat(0);
    base_rows[0] = n_experts;
    const st = try es.Stream.init(a, &bank, .{ .rows = &base_rows, .max_route_ids = n_experts, .transient_rows = n_experts, .slot_memory = .{ .mlx = s } });
    defer st.deinit();
    var ids: [n_experts]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(i);
    const route = try st.route(0, &ids, &.{});
    defer st.release(route);
    for (0..route.n_parts) |p| {
        try st.waitGu(route, @intCast(p));
        try st.waitDown(route, @intCast(p));
    }
    var refs: [@import("../../../weights/streamed/sdk_ext.zig").expert.policy.max_route_ids]es.SlotRef = undefined;
    const rf = st.refsOf(route, &refs);
    try testing.expectEqual(@as(usize, n_experts), rf.len);
    const ba = st.bankArrays(0, .base) orelse return error.TestUnexpectedResult;
    const dig = DigX(G).init(&set.reg);
    const tiles_gu = dig.digTiles(.gate_up);
    const tiles_dn = dig.digTiles(.down);
    const Wave = struct {
        rows: u32,
        x0: G.T,
        x1: G.T,
        hd: G.T,
        gu: [2]DigTable,
        dn: [2]DigTable,

        /// A wave's rows grouped by expert in order (expert j at real record `refs_[j]`), its GEMM inputs from the
        /// routed chain: x0 / x1 = take2(act), hd = onepass(the 64-row gate|up); tables at 64 and 128 rows.
        fn init(al: Allocator, gg: *G, per: []const u32, refs_: []const es.SlotRef, d: DigX(G), bank_: anytype, t_gu: u32, t_dn: u32, seed: u64) !@This() {
            var ex: [wave_max]WaveExpert = undefined;
            var n: u32 = 0;
            for (per, 0..) |c, j| {
                ex[j] = .{ .slot = refs_[j].row, .rows = c };
                n += c;
            }
            const ridx = try al.alloc(i32, n);
            defer al.free(ridx);
            const rhs = try al.alloc(u32, n);
            defer al.free(rhs);
            var i: usize = 0;
            for (per, 0..) |c, j| for (0..c) |_| {
                ridx[i] = @intCast(i);
                rhs[i] = @intCast(j);
                i += 1;
            };
            const xs = try al.alloc(f32, @as(usize, n) * 5120);
            defer al.free(xs);
            var h: u64 = seed;
            for (xs) |*v| {
                h = h *% 6364136223846793005 +% 1442695040888963407;
                const u_1: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(h >> 40)))) / 16777216.0;
                const u_2: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(h >> 16)) & 0xffffff)) / 16777216.0;
                v.* = (u_1 + u_2 - 1.0) * 2.0;
            }
            const ni: c_int = @intCast(n);
            const act = try gg.astype(try gg.hostArray(std.mem.sliceAsBytes(xs), &.{ ni, 5120 }, .float32), .bfloat16);
            const ridx_a = try gg.hostArray(std.mem.sliceAsBytes(ridx), &.{ni}, .int32);
            const rhs_a = try gg.hostArray(std.mem.sliceAsBytes(rhs), &.{ni}, .uint32);
            var w: @This() = .{ .rows = n, .x0 = undefined, .x1 = undefined, .hd = undefined, .gu = undefined, .dn = undefined };
            for ([_]u32{ 64, 128 }, 0..) |bm, k| {
                w.gu[k] = digTableBm(ex[0..per.len], t_gu, bm);
                w.dn[k] = digTableBm(ex[0..per.len], t_dn, bm);
            }
            const tbl = try gg.hostArray(std.mem.sliceAsBytes(&w.gu[0].table), &.{80}, .int32);
            const x = try d.take2(gg, act, ridx_a, rhs_a, tbl, bank_.gate.rin, bank_.up.rin);
            w.x0 = x[0];
            w.x1 = x[1];
            // the 64-row gate|up through its own entry, whichever text the route launches
            const z = try w.gemm(gg, d, bank_, 0, 0);
            w.hd = try d.onePass(gg, z[0], z[1], rhs_a, tbl, bank_.gate.rout, bank_.up.rout, bank_.down.rin);
            try gg.evalAll(&.{ w.x0, w.x1, w.hd });
            return w;
        }

        /// The gate|up (k = 0) or down (k = 1) GEMM at M tile `bm` (index 0: 64 rows, 1: 128).
        fn gemm(w: *const @This(), gg: *G, d: DigX(G), bank_: anytype, k: usize, bi: usize) ![2]G.T {
            var out: [2]G.T = undefined;
            if (k == 0) {
                const e = if (bi == 0) d.gemm_gu else d.gemm_gu128;
                var vars = rowsVars(w.rows);
                vars.set(.tgs, w.gu[bi].tgs);
                const tbl = try gg.hostArray(std.mem.sliceAsBytes(&w.gu[bi].table), &.{80}, .int32);
                try launchRule(G, gg, e, &vars, &.{ w.x0, w.x1, bank_.gate.code, bank_.up.code, tbl }, &out);
            } else {
                const e = if (bi == 0) d.gemm_dn else d.gemm_dn128;
                var vars = rowsVars(w.rows);
                vars.set(.tgs, w.dn[bi].tgs);
                const tbl = try gg.hostArray(std.mem.sliceAsBytes(&w.dn[bi].table), &.{80}, .int32);
                try launchRule(G, gg, e, &vars, &.{ w.hd, bank_.down.code, tbl }, out[0..1]);
                out[1] = out[0];
            }
            return out;
        }
    };
    const k16 = [_]u32{ 256, 256, 256, 256, 256, 256, 256, 256 };
    // 2. the 64- and 128-row texts on the same inputs: every z word
    for ([_][]const u32{ &.{ 1, 17, 64, 127, 128, 129, 270 }, &k16 }, 0..) |per, wi| {
        const m = g.mark();
        defer g.resetTo(m);
        const w = try Wave.init(a, &g, per, rf, dig, ba, tiles_gu, tiles_dn, 0x3128 + wi);
        var words: usize = 0;
        var mismatch: usize = 0;
        var nonzero: usize = 0;
        for (0..2) |k| {
            const z64 = try w.gemm(&g, dig, ba, k, 0);
            const z128 = try w.gemm(&g, dig, ba, k, 1);
            try g.evalAll(&.{ z64[0], z64[1], z128[0], z128[1] });
            const n_out: usize = if (k == 0) 2 else 1;
            const cols: usize = if (k == 0) 2304 else 5120;
            for (0..n_out) |o| {
                const px = mlx.mlx_array_data_uint8(z64[o]) orelse return error.TestUnexpectedResult;
                const py = mlx.mlx_array_data_uint8(z128[o]) orelse return error.TestUnexpectedResult;
                const n_words = @as(usize, w.rows) * cols;
                for (0..n_words) |q| {
                    const wx = std.mem.readInt(u32, px[4 * q ..][0..4], .little);
                    const wy = std.mem.readInt(u32, py[4 * q ..][0..4], .little);
                    mismatch += @intFromBool(wx != wy);
                    nonzero += @intFromBool(wx & 0x7fffffff != 0);
                }
                words += n_words;
            }
        }
        std.debug.print("\nM128_GEMM_SMOKE {{\"rows\": {d}, \"experts\": {d}, \"words\": {d}, \"mismatch\": {d}, \"nonzero\": {d}}}\n", .{ w.rows, per.len, words, mismatch, nonzero });
        try testing.expectEqual(@as(usize, 0), mismatch);
        try testing.expect(nonzero > words / 2);
    }
    // 3. per-launch time: both texts at 2,048 and 1,080 rows (DIGX_WAVE_MICROBENCH), today's at 8 x 32 / 64 / 256
    //    (DIGX_GEMM_SWEEP)
    const Time = struct {
        /// The best of 5 timed runs (after a warm one) of 8 launches of GEMM `k` at M tile index `bi`, per launch.
        fn best(gg: *G, io_: std.Io, w: anytype, d: DigX(G), bank_: anytype, k: usize, bi: usize) !struct { eval: u64, host: u64 } {
            var best_eval: u64 = std.math.maxInt(u64);
            var best_host: u64 = std.math.maxInt(u64);
            for (0..6) |run| {
                const mr = gg.mark();
                defer gg.resetTo(mr);
                var outs: [16]G.T = undefined;
                const t0 = std.Io.Timestamp.now(io_, .boot);
                for (0..8) |q| {
                    const o = try w.gemm(gg, d, bank_, k, bi);
                    outs[2 * q] = o[0];
                    outs[2 * q + 1] = o[1];
                }
                const host: u64 = @intCast(t0.untilNow(io_, .boot).nanoseconds);
                const t1 = std.Io.Timestamp.now(io_, .boot);
                try gg.evalAll(&outs);
                const eval: u64 = @intCast(t1.untilNow(io_, .boot).nanoseconds);
                if (run > 0) {
                    best_eval = @min(best_eval, eval);
                    best_host = @min(best_host, host);
                }
            }
            return .{ .eval = best_eval / 8, .host = best_host / 8 };
        }
    };
    for ([_][]const u32{ &k16, &.{ 270, 270, 270, 270 } }) |per| {
        const m = g.mark();
        defer g.resetTo(m);
        const w = try Wave.init(a, &g, per, rf, dig, ba, tiles_gu, tiles_dn, 0xbe7c);
        for (0..2) |k| for (0..2) |bi| {
            const t = try Time.best(&g, io, &w, dig, ba, k, bi);
            const kname = switch (k * 2 + bi) {
                0 => @tagName(Kernel.q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3),
                1 => @tagName(Kernel.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128),
                2 => @tagName(Kernel.q3_prefill_dig_gemm_2304x5120_xmul1hk3),
                else => @tagName(Kernel.dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128),
            };
            std.debug.print("DIGX_WAVE_MICROBENCH {{\"kernel\": \"{s}\", \"rows\": {d}, \"us\": {d:.1}, \"host_us\": {d:.1}}}\n", .{ kname, w.rows, @as(f64, @floatFromInt(t.eval)) / 1000.0, @as(f64, @floatFromInt(t.host)) / 1000.0 });
        };
    }
    for ([_]u32{ 32, 64, 256 }) |r| {
        const m = g.mark();
        defer g.resetTo(m);
        const per = [_]u32{ r, r, r, r, r, r, r, r };
        const w = try Wave.init(a, &g, &per, rf, dig, ba, tiles_gu, tiles_dn, 0x5eeb);
        for (0..2) |k| {
            const t = try Time.best(&g, io, &w, dig, ba, k, 0);
            std.debug.print("DIGX_GEMM_SWEEP {{\"gemm\": \"{s}\", \"experts\": 8, \"rows_per_expert\": {d}, \"us\": {d:.1}}}\n", .{ if (k == 0) "gate_up" else "down", r, @as(f64, @floatFromInt(t.eval)) / 1000.0 });
        }
    }
}

// DSV41_PHASE0B_MLX=1 and DSV41_BANK, lock-held (seconds, under 2 GB of device memory: 16 slot rows and one wave's
// act / hd / z / outputs). The fused down GEMM against its two-kernel chain on real records.
test "dsv41 smoke 0b: fused down: the fused down GEMM's words equal the 128-row down text then widen1 on real records, bit for bit; microbench and sweep" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_BANK) orelse {
        std.debug.print("\nfused down smoke: DSV41_PHASE0B_MLX without DSV41_BANK (the real records): refused\n", .{});
        return error.TestUnexpectedResult;
    });
    const ops = @import("../../../weights/streamed/ops.zig");
    const es = @import("../../../weights/streamed/expert/stream.zig");
    const eb = @import("../../../weights/streamed/expert/bank.zig");
    const G = ops.MlxOps;
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var kd: xk.Diag = .{};
    const set = ks.Set.init(a, .{ .device = .{ .stream = s } }, &kd) catch |e| {
        std.debug.print("kernel set refused: {s}\n", .{kd.message()});
        return e;
    };
    defer set.deinit();
    set.install(G, &g);
    defer ks.Set.uninstall(G, &g);
    // 1. the fused text's construction self-checks, on the device
    {
        var report: selfcheck.Report = .{};
        defer report.deinit(a);
        set.selfCheck(a, &w1_texts, .full, &report, &kd) catch |e| {
            std.debug.print("fused down self-checks refused: {s}\n", .{kd.message()});
            return e;
        };
        var fused: usize = 0;
        for (report.results.items) |r| fused += @intFromBool(r.check == .fused and r.ok and r.bad == 0 and r.words > 0);
        try testing.expectEqual(@as(usize, 1), fused);
    }
    // layer 0's first 16 experts, read by the stream into 16 persistent MLX rows (the base bank)
    var bdiag: eb.Diag = .{};
    var bank = eb.Bank.open(a, io, dir, eb.dsv41, &bdiag) catch |e| {
        std.debug.print("bank refused: {s}\n", .{bdiag.message()});
        return e;
    };
    defer bank.deinit();
    const n_experts = 16;
    var base_rows: [40]u32 = @splat(0);
    base_rows[0] = n_experts;
    const st = try es.Stream.init(a, &bank, .{ .rows = &base_rows, .max_route_ids = n_experts, .transient_rows = n_experts, .slot_memory = .{ .mlx = s } });
    defer st.deinit();
    var ids: [n_experts]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(i);
    const route = try st.route(0, &ids, &.{});
    defer st.release(route);
    for (0..route.n_parts) |p| {
        try st.waitGu(route, @intCast(p));
        try st.waitDown(route, @intCast(p));
    }
    var refs: [@import("../../../weights/streamed/sdk_ext.zig").expert.policy.max_route_ids]es.SlotRef = undefined;
    const rf = st.refsOf(route, &refs);
    try testing.expectEqual(@as(usize, n_experts), rf.len);
    const ba = st.bankArrays(0, .base) orelse return error.TestUnexpectedResult;
    const dig = DigX(G).init(&set.reg);
    const Wave = struct {
        rows: u32,
        hd: G.T,
        rhs: G.T,
        dn: DigTable,
        w1: DigTable,

        /// A wave's rows grouped by expert in order (expert j at real record `refs_[j]`), hd from the routed chain
        /// (take2, the gate|up GEMM, onepass); the down tables at 128 rows for the two-kernel chain and the fused text.
        fn init(al: Allocator, gg: *G, per: []const u32, refs_: []const es.SlotRef, d: DigX(G), bank_: anytype, seed: u64) !@This() {
            var ex: [wave_max]WaveExpert = undefined;
            var n: u32 = 0;
            for (per, 0..) |c, j| {
                ex[j] = .{ .slot = refs_[j].row, .rows = c };
                n += c;
            }
            const ridx = try al.alloc(i32, n);
            defer al.free(ridx);
            const rhs = try al.alloc(u32, n);
            defer al.free(rhs);
            var i: usize = 0;
            for (per, 0..) |c, j| for (0..c) |_| {
                ridx[i] = @intCast(i);
                rhs[i] = @intCast(j);
                i += 1;
            };
            const xs = try al.alloc(f32, @as(usize, n) * 5120);
            defer al.free(xs);
            var h: u64 = seed;
            for (xs) |*v| {
                h = h *% 6364136223846793005 +% 1442695040888963407;
                const u_1: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(h >> 40)))) / 16777216.0;
                const u_2: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(h >> 16)) & 0xffffff)) / 16777216.0;
                v.* = (u_1 + u_2 - 1.0) * 2.0;
            }
            const ni: c_int = @intCast(n);
            const act = try gg.astype(try gg.hostArray(std.mem.sliceAsBytes(xs), &.{ ni, 5120 }, .float32), .bfloat16);
            const ridx_a = try gg.hostArray(std.mem.sliceAsBytes(ridx), &.{ni}, .int32);
            const rhs_a = try gg.hostArray(std.mem.sliceAsBytes(rhs), &.{ni}, .uint32);
            const gu = digTableBm(ex[0..per.len], d.digTiles(.gate_up), DigX(G).m_tile);
            const tbl_gu = try gg.hostArray(std.mem.sliceAsBytes(&gu.table), &.{80}, .int32);
            const x = try d.take2(gg, act, ridx_a, rhs_a, tbl_gu, bank_.gate.rin, bank_.up.rin);
            const z = try d.gemmGateUp(gg, x[0], x[1], bank_.gate.code, bank_.up.code, tbl_gu, gu.tgs);
            const hd = try d.onePass(gg, z[0], z[1], rhs_a, tbl_gu, bank_.gate.rout, bank_.up.rout, bank_.down.rin);
            try gg.evalAll(&.{hd});
            return .{ .rows = n, .hd = hd, .rhs = rhs_a, .dn = digTableBm(ex[0..per.len], d.digTiles(.down), DigX(G).m_tile), .w1 = digTableBm(ex[0..per.len], d.widenTiles(), DigX(G).m_tile) };
        }

        /// The two-kernel chain (the 128-row down text, then widen1) or the fused text: [rows, 5120] f32 words.
        fn run(w: *const @This(), gg: *G, d: DigX(G), bank_: anytype, fused: bool) !G.T {
            if (fused) {
                const tbl = try gg.hostArray(std.mem.sliceAsBytes(&w.w1.table), &.{80}, .int32);
                return d.gemmDownWiden(gg, w.hd, bank_.down.code, bank_.down.rout, tbl, w.w1.tgs);
            }
            const tbl = try gg.hostArray(std.mem.sliceAsBytes(&w.dn.table), &.{80}, .int32);
            const z = try d.gemmDown(gg, w.hd, bank_.down.code, tbl, w.dn.tgs);
            return d.widen1(gg, z, w.rhs, tbl, bank_.down.rout);
        }
    };
    const k16 = [_]u32{ 256, 256, 256, 256, 256, 256, 256, 256 };
    // 2. both paths on the same hd at every partial-tile edge and a K16 wave: every output word
    for ([_][]const u32{ &.{ 1, 17, 64, 127, 128, 129, 270 }, &k16 }, 0..) |per, wi| {
        const m = g.mark();
        defer g.resetTo(m);
        const w = try Wave.init(a, &g, per, rf, dig, ba, 0x51de + wi);
        const chain = try w.run(&g, dig, ba, false);
        const fused = try w.run(&g, dig, ba, true);
        try g.evalAll(&.{ chain, fused });
        const px = mlx.mlx_array_data_uint8(chain) orelse return error.TestUnexpectedResult;
        const py = mlx.mlx_array_data_uint8(fused) orelse return error.TestUnexpectedResult;
        const words = @as(usize, w.rows) * 5120;
        var mismatch: usize = 0;
        var nonzero: usize = 0;
        for (0..words) |q| {
            const wx = std.mem.readInt(u32, px[4 * q ..][0..4], .little);
            const wy = std.mem.readInt(u32, py[4 * q ..][0..4], .little);
            mismatch += @intFromBool(wx != wy);
            nonzero += @intFromBool(wx & 0x7fffffff != 0);
        }
        std.debug.print("\nFUSED_W1_SMOKE {{\"rows\": {d}, \"experts\": {d}, \"words\": {d}, \"mismatch\": {d}, \"nonzero\": {d}}}\n", .{ w.rows, per.len, words, mismatch, nonzero });
        try testing.expectEqual(@as(usize, 0), mismatch);
        try testing.expect(nonzero > words / 2);
    }
    // 3. per-wave time of both paths at 2,048 and 1,080 rows (DIGX_WAVE_MICROBENCH) and at 8 x 32 / 64 / 256 rows
    //    (DIGX_GEMM_SWEEP)
    const Time = struct {
        /// The best of 5 timed runs (after a warm one) of 8 waves of one path, per wave.
        fn best(gg: *G, io_: std.Io, w: anytype, d: DigX(G), bank_: anytype, fused: bool) !struct { eval: u64, host: u64 } {
            var best_eval: u64 = std.math.maxInt(u64);
            var best_host: u64 = std.math.maxInt(u64);
            for (0..6) |run| {
                const mr = gg.mark();
                defer gg.resetTo(mr);
                var outs: [8]G.T = undefined;
                const t0 = std.Io.Timestamp.now(io_, .boot);
                for (&outs) |*o| o.* = try w.run(gg, d, bank_, fused);
                const host: u64 = @intCast(t0.untilNow(io_, .boot).nanoseconds);
                const t1 = std.Io.Timestamp.now(io_, .boot);
                try gg.evalAll(&outs);
                const eval: u64 = @intCast(t1.untilNow(io_, .boot).nanoseconds);
                if (run > 0) {
                    best_eval = @min(best_eval, eval);
                    best_host = @min(best_host, host);
                }
            }
            return .{ .eval = best_eval / 8, .host = best_host / 8 };
        }
    };
    const chain_name = @tagName(Kernel.dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128) ++ "+" ++ @tagName(Kernel.q3_prefill_dig_rot_widen1_5120);
    for ([_][]const u32{ &k16, &.{ 270, 270, 270, 270 } }) |per| {
        const m = g.mark();
        defer g.resetTo(m);
        const w = try Wave.init(a, &g, per, rf, dig, ba, 0xfd3e);
        for ([_]bool{ false, true }) |fused| {
            const t = try Time.best(&g, io, &w, dig, ba, fused);
            const kname = if (fused) @tagName(Kernel.dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1) else chain_name;
            std.debug.print("DIGX_WAVE_MICROBENCH {{\"kernel\": \"{s}\", \"rows\": {d}, \"us\": {d:.1}, \"host_us\": {d:.1}}}\n", .{ kname, w.rows, @as(f64, @floatFromInt(t.eval)) / 1000.0, @as(f64, @floatFromInt(t.host)) / 1000.0 });
        }
    }
    for ([_]u32{ 32, 64, 256 }) |r| {
        const m = g.mark();
        defer g.resetTo(m);
        const per = [_]u32{ r, r, r, r, r, r, r, r };
        const w = try Wave.init(a, &g, &per, rf, dig, ba, 0x5eeb);
        for ([_]bool{ false, true }) |fused| {
            const t = try Time.best(&g, io, &w, dig, ba, fused);
            std.debug.print("DIGX_GEMM_SWEEP {{\"gemm\": \"{s}\", \"experts\": 8, \"rows_per_expert\": {d}, \"us\": {d:.1}}}\n", .{ if (fused) "down_w1" else "down+widen1", r, @as(f64, @floatFromInt(t.eval)) / 1000.0 });
        }
    }
}

// DSV41_PHASE0B_MLX=1 and DSV41_BANK, lock-held (seconds, under 2 GB of device memory: 16 slot rows and one wave's
// act / x / z). The table-codebook gate|up text against the 128-row text on real records; the gate|up texts' tail tiles
// priced (lever 3).
test "dsv41 smoke 0b: lut: the table-codebook gate|up GEMM's z words equal the 128-row text's on real records, bit for bit; microbench and sweep" {
    _ = env.get(.DSV41_PHASE0B_MLX) orelse return error.SkipZigTest;
    const dir = std.mem.span(env.get(.DSV41_BANK) orelse {
        std.debug.print("\nlut smoke: DSV41_PHASE0B_MLX without DSV41_BANK (the real records): refused\n", .{});
        return error.TestUnexpectedResult;
    });
    const ops = @import("../../../weights/streamed/ops.zig");
    const es = @import("../../../weights/streamed/expert/stream.zig");
    const eb = @import("../../../weights/streamed/expert/bank.zig");
    const G = ops.MlxOps;
    const a = testing.allocator;
    const io = testing.io;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var g = try G.init(a, s);
    defer g.deinit();
    var kd: xk.Diag = .{};
    const set = ks.Set.init(a, .{ .device = .{ .stream = s } }, &kd) catch |e| {
        std.debug.print("kernel set refused: {s}\n", .{kd.message()});
        return e;
    };
    defer set.deinit();
    set.install(G, &g);
    defer ks.Set.uninstall(G, &g);
    // 1. the table text's construction self-checks, on the device
    {
        var report: selfcheck.Report = .{};
        defer report.deinit(a);
        set.selfCheck(a, &lut_texts, .full, &report, &kd) catch |e| {
            std.debug.print("lut self-checks refused: {s}\n", .{kd.message()});
            return e;
        };
        var twins: usize = 0;
        for (report.results.items) |r| twins += @intFromBool(r.check == .twin and r.ok and r.bad == 0 and r.words > 0);
        try testing.expectEqual(@as(usize, 1), twins);
    }
    // layer 0's first 16 experts, read by the stream into 16 persistent MLX rows (the base bank)
    var bdiag: eb.Diag = .{};
    var bank = eb.Bank.open(a, io, dir, eb.dsv41, &bdiag) catch |e| {
        std.debug.print("bank refused: {s}\n", .{bdiag.message()});
        return e;
    };
    defer bank.deinit();
    const n_experts = 16;
    var base_rows: [40]u32 = @splat(0);
    base_rows[0] = n_experts;
    const st = try es.Stream.init(a, &bank, .{ .rows = &base_rows, .max_route_ids = n_experts, .transient_rows = n_experts, .slot_memory = .{ .mlx = s } });
    defer st.deinit();
    var ids: [n_experts]u16 = undefined;
    for (&ids, 0..) |*e, i| e.* = @intCast(i);
    const route = try st.route(0, &ids, &.{});
    defer st.release(route);
    for (0..route.n_parts) |p| {
        try st.waitGu(route, @intCast(p));
        try st.waitDown(route, @intCast(p));
    }
    var refs: [@import("../../../weights/streamed/sdk_ext.zig").expert.policy.max_route_ids]es.SlotRef = undefined;
    const rf = st.refsOf(route, &refs);
    try testing.expectEqual(@as(usize, n_experts), rf.len);
    const ba = st.bankArrays(0, .base) orelse return error.TestUnexpectedResult;
    const dig = DigX(G).init(&set.reg);
    // the gate|up GEMM texts: the 64-row text, the 128-row text, the table-codebook text
    const Text = enum { m64, m128, lut };
    const Wave = struct {
        rows: u32,
        x0: G.T,
        x1: G.T,
        gu: DigTable,
        gu64: DigTable,

        /// A wave's rows grouped by expert in order (expert j at real record `refs_[j]`), its gate|up inputs from the
        /// routed chain: x0 / x1 = take2(act); its tables at 128 and 64 rows.
        fn init(al: Allocator, gg: *G, per: []const u32, refs_: []const es.SlotRef, d: DigX(G), bank_: anytype, seed: u64) !@This() {
            var ex: [wave_max]WaveExpert = undefined;
            var n: u32 = 0;
            for (per, 0..) |c, j| {
                ex[j] = .{ .slot = refs_[j].row, .rows = c };
                n += c;
            }
            const ridx = try al.alloc(i32, n);
            defer al.free(ridx);
            const rhs = try al.alloc(u32, n);
            defer al.free(rhs);
            var i: usize = 0;
            for (per, 0..) |c, j| for (0..c) |_| {
                ridx[i] = @intCast(i);
                rhs[i] = @intCast(j);
                i += 1;
            };
            const xs = try al.alloc(f32, @as(usize, n) * 5120);
            defer al.free(xs);
            var h: u64 = seed;
            for (xs) |*v| {
                h = h *% 6364136223846793005 +% 1442695040888963407;
                const u_1: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(h >> 40)))) / 16777216.0;
                const u_2: f32 = @as(f32, @floatFromInt(@as(u32, @truncate(h >> 16)) & 0xffffff)) / 16777216.0;
                v.* = (u_1 + u_2 - 1.0) * 2.0;
            }
            const ni: c_int = @intCast(n);
            const act = try gg.astype(try gg.hostArray(std.mem.sliceAsBytes(xs), &.{ ni, 5120 }, .float32), .bfloat16);
            const ridx_a = try gg.hostArray(std.mem.sliceAsBytes(ridx), &.{ni}, .int32);
            const rhs_a = try gg.hostArray(std.mem.sliceAsBytes(rhs), &.{ni}, .uint32);
            const gu = digTableBm(ex[0..per.len], d.digTiles(.gate_up), DigX(G).m_tile);
            const tbl = try gg.hostArray(std.mem.sliceAsBytes(&gu.table), &.{80}, .int32);
            const x = try d.take2(gg, act, ridx_a, rhs_a, tbl, bank_.gate.rin, bank_.up.rin);
            try gg.evalAll(&.{ x[0], x[1] });
            return .{ .rows = n, .x0 = x[0], .x1 = x[1], .gu = gu, .gu64 = digTableBm(ex[0..per.len], d.digTiles(.gate_up), 64) };
        }

        /// The gate|up GEMM through `text`, over its own table.
        fn gemm(w: *const @This(), gg: *G, d: DigX(G), bank_: anytype, text: Text) ![2]G.T {
            const t = if (text == .m64) &w.gu64 else &w.gu;
            const tbl = try gg.hostArray(std.mem.sliceAsBytes(&t.table), &.{80}, .int32);
            switch (text) {
                .m64 => {
                    var vars = rowsVars(w.rows);
                    vars.set(.tgs, t.tgs);
                    var out: [2]G.T = undefined;
                    try launchRule(G, gg, d.gemm_gu, &vars, &.{ w.x0, w.x1, bank_.gate.code, bank_.up.code, tbl }, &out);
                    return out;
                },
                .m128 => return d.gemmGateUp(gg, w.x0, w.x1, bank_.gate.code, bank_.up.code, tbl, t.tgs),
                .lut => return d.gemmGateUpLut(gg, w.x0, w.x1, bank_.gate.code, bank_.up.code, tbl, t.tgs),
            }
        }
    };
    const k16 = [_]u32{ 256, 256, 256, 256, 256, 256, 256, 256 };
    // 2. both texts on the same inputs at every partial-tile edge and a K16 wave: every z word
    for ([_][]const u32{ &.{ 1, 17, 64, 127, 128, 129, 270 }, &k16 }, 0..) |per, wi| {
        const m = g.mark();
        defer g.resetTo(m);
        const w = try Wave.init(a, &g, per, rf, dig, ba, 0x7a17 + wi);
        const z = try w.gemm(&g, dig, ba, .m128);
        const zl = try w.gemm(&g, dig, ba, .lut);
        try g.evalAll(&.{ z[0], z[1], zl[0], zl[1] });
        var words: usize = 0;
        var mismatch: usize = 0;
        var nonzero: usize = 0;
        for (0..2) |o| {
            const px = mlx.mlx_array_data_uint8(z[o]) orelse return error.TestUnexpectedResult;
            const py = mlx.mlx_array_data_uint8(zl[o]) orelse return error.TestUnexpectedResult;
            const n_words = @as(usize, w.rows) * 2304;
            for (0..n_words) |q| {
                const wx = std.mem.readInt(u32, px[4 * q ..][0..4], .little);
                const wy = std.mem.readInt(u32, py[4 * q ..][0..4], .little);
                mismatch += @intFromBool(wx != wy);
                nonzero += @intFromBool(wx & 0x7fffffff != 0);
            }
            words += n_words;
        }
        std.debug.print("\nLUT_GU_SMOKE {{\"rows\": {d}, \"experts\": {d}, \"words\": {d}, \"mismatch\": {d}, \"nonzero\": {d}}}\n", .{ w.rows, per.len, words, mismatch, nonzero });
        try testing.expectEqual(@as(usize, 0), mismatch);
        try testing.expect(nonzero > words / 2);
    }
    // 3. per-launch time of both texts at 2,048 and 1,080 rows (DIGX_WAVE_MICROBENCH) and at 8 x 32 / 64 / 256 rows
    //    (DIGX_GEMM_SWEEP)
    const Time = struct {
        /// The best of 5 timed runs (after a warm one) of 8 launches of one text, per launch.
        fn best(gg: *G, io_: std.Io, w: anytype, d: DigX(G), bank_: anytype, text: Text) !struct { eval: u64, host: u64 } {
            var best_eval: u64 = std.math.maxInt(u64);
            var best_host: u64 = std.math.maxInt(u64);
            for (0..6) |run| {
                const mr = gg.mark();
                defer gg.resetTo(mr);
                var outs: [16]G.T = undefined;
                const t0 = std.Io.Timestamp.now(io_, .boot);
                for (0..8) |q| {
                    const o = try w.gemm(gg, d, bank_, text);
                    outs[2 * q] = o[0];
                    outs[2 * q + 1] = o[1];
                }
                const host: u64 = @intCast(t0.untilNow(io_, .boot).nanoseconds);
                const t1 = std.Io.Timestamp.now(io_, .boot);
                try gg.evalAll(&outs);
                const eval: u64 = @intCast(t1.untilNow(io_, .boot).nanoseconds);
                if (run > 0) {
                    best_eval = @min(best_eval, eval);
                    best_host = @min(best_host, host);
                }
            }
            return .{ .eval = best_eval / 8, .host = best_host / 8 };
        }
    };
    for ([_][]const u32{ &k16, &.{ 270, 270, 270, 270 } }) |per| {
        const m = g.mark();
        defer g.resetTo(m);
        const w = try Wave.init(a, &g, per, rf, dig, ba, 0xbe7c);
        for ([_]Text{ .m128, .lut }) |text| {
            const t = try Time.best(&g, io, &w, dig, ba, text);
            const kname = if (text == .lut) @tagName(Kernel.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut) else @tagName(Kernel.dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128);
            std.debug.print("DIGX_WAVE_MICROBENCH {{\"kernel\": \"{s}\", \"rows\": {d}, \"us\": {d:.1}, \"host_us\": {d:.1}}}\n", .{ kname, w.rows, @as(f64, @floatFromInt(t.eval)) / 1000.0, @as(f64, @floatFromInt(t.host)) / 1000.0 });
        }
    }
    for ([_]u32{ 32, 64, 256 }) |r| {
        const m = g.mark();
        defer g.resetTo(m);
        const per = [_]u32{ r, r, r, r, r, r, r, r };
        const w = try Wave.init(a, &g, &per, rf, dig, ba, 0x5eeb);
        for ([_]Text{ .m128, .lut }) |text| {
            const t = try Time.best(&g, io, &w, dig, ba, text);
            std.debug.print("DIGX_GEMM_SWEEP {{\"gemm\": \"{s}\", \"experts\": 8, \"rows_per_expert\": {d}, \"us\": {d:.1}}}\n", .{ if (text == .lut) "gate_up_lut" else "gate_up", r, @as(f64, @floatFromInt(t.eval)) / 1000.0 });
        }
    }
    // 4. tail tiles (lever 3's pricing): the 64-row, 128-row and table texts at 8 experts x 33 / 48 / 64 / 96 / 128 / 256
    //    rows (DIGX_TAIL_SWEEP); a tail tile's cost = (the wave - the full tiles at the 8 x 256 wave's per-tile cost) / tails
    for ([_]u32{ 33, 48, 64, 96, 128, 256 }) |r| {
        const m = g.mark();
        defer g.resetTo(m);
        const per = [_]u32{ r, r, r, r, r, r, r, r };
        const w = try Wave.init(a, &g, &per, rf, dig, ba, 0x7a11);
        for ([_]Text{ .m64, .m128, .lut }) |text| {
            const t = try Time.best(&g, io, &w, dig, ba, text);
            const gname = switch (text) {
                .m64 => "gate_up_m64",
                .m128 => "gate_up",
                .lut => "gate_up_lut",
            };
            std.debug.print("DIGX_TAIL_SWEEP {{\"gemm\": \"{s}\", \"experts\": 8, \"rows_per_expert\": {d}, \"us\": {d:.1}}}\n", .{ gname, r, @as(f64, @floatFromInt(t.eval)) / 1000.0 });
        }
    }
}

// The served construction's self-checks alone, on the device (a guarded step: the GPU lock held). The two plans the
// served module judges at construction, the EXL3 quant's accept subset and the V4.1 trunk's, each result printed
// (PASS / FAIL, words / bad, metric / limit, the reference side), then the module's verdict. No model, no bank.
test "dsv41 pre-ship gate: the kernels' full self-check plan on the device, every result printed (DSV41_SELFCHECK_DEVICE)" {
    if (env.get(.DSV41_SELFCHECK_DEVICE) == null) return error.SkipZigTest;
    const trunk = @import("kernel_routes.zig");
    @import("sdk").log.enableStderr(); // the results are this test's output
    const a = testing.allocator;
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var kd: xk.Diag = .{};
    const set = ks.Set.init(a, .{ .device = .{ .stream = s } }, &kd) catch |e| {
        std.debug.print("DSV41_SELFCHECK_DEVICE set refused: {t} {s}\n", .{ e, kd.message() });
        return e;
    };
    defer set.deinit();
    var failed: usize = 0;
    // Both depths, each timed: the construction runs `.compile`; `.full` is the manifest's whole plan (the numeric probes).
    inline for (.{ selfcheck.Depth.compile, selfcheck.Depth.full }) |depth| {
        inline for (.{ .{ "exl3_quant_accept", &checked_at_accept }, .{ "trunk_routes", &trunk.kernels } }) |plan| {
            var report: selfcheck.Report = .{};
            defer report.deinit(a);
            const t0 = std.Io.Timestamp.now(testing.io, .awake);
            selfcheck.runSubset(a, &set.reg, &set.bound, plan[1], depth, &report) catch |e| {
                std.debug.print("DSV41_SELFCHECK_DEVICE plan {s} {t} raised {t}\n", .{ plan[0], depth, e });
                return e;
            };
            const ms = @as(f64, @floatFromInt(@max(t0.untilNow(testing.io, .awake).nanoseconds, 0))) / 1e6;
            std.debug.print("DSV41_SELFCHECK_DEVICE plan {s} {t}\n", .{ plan[0], depth });
            selfcheck.logResults(&report, true);
            const n = report.failures();
            failed += n;
            std.debug.print("DSV41_SELFCHECK_DEVICE verdict {s} {t} results={d} failed={d} ms={d:.1}\n", .{ plan[0], depth, report.results.items.len, n, ms });
        }
    }
    std.debug.print("DSV41_SELFCHECK_DEVICE done failed={d}\n", .{failed});
    // The pre-ship gate (the served construction runs compiles only): every numeric probe of the manifest passes here.
    try testing.expectEqual(@as(usize, 0), failed);
}
