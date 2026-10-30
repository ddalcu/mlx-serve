//! The self-checks a GPU run runs before the EXL3 kernel registry is accepted: one
//! executor per (kernel, check) of the manifest's plan. Every trellis state decoded through
//! the GEMVs, golden tiles through the DIG-X decode, row invariance of the decode kernels,
//! bitwise equality with the eager MLX chains the fused kernels replace, float64 parity of
//! the rounding-class kernels, wave composition of the DIG GEMMs, and the rcproj layout guard
//! against MLX's own mxfp8 route. GPU only (DSV41_KERNELS_GPU=1 in a lock-holding window).

const std = @import("std");
const env = @import("mlx_host").env;
const mlx = @import("sdk").mlx;
const log = @import("sdk").log;
const xk = @import("kernels.zig");
const kset = @import("../../../weights/streamed/sdk_ext/kernel_set.zig");

const Allocator = std.mem.Allocator;
const Kernel = xk.Kernel;
const Check = xk.Check;
const Entry = xk.Entry;
const Vars = xk.Vars;
const Var = xk.Var;

const scaffold = kset.Checks(Kernel, Check);
pub const Result = scaffold.Result;
pub const Report = scaffold.Report;

/// Which (kernel, check) pairs this executor implements; the host test holds the manifest to it.
pub fn implemented(k: Kernel, c: Check) bool {
    return switch (c) {
        .compile, .row_invariance => true,
        .join_equiv => k == .q3jl_combine or k == .dsv41_jl_combine_bf16,
        .twin => twinOf(k) != null or formTwinOf(k) != null or bankedTwinOf(k) != null,
        .fused => fusedOf(k) != null,
        .decode_table => k == .dsv41_exl3_mul1h_k3_2304 or k == .dsv41_exl3_mul1h_k3_5120 or k == .dsv41_exl3_mul1h_k2_2304 or k == .dsv41_exl3_mul1h_k2_5120,
        .golden_tiles => std.mem.startsWith(u8, @tagName(k), "q3_exl3_dig_decmat_"),
        .composition => isDigGemm(k) or fusedOf(k) != null,
        .layout_guard => k == .q3rc_mxfp8_fma or k == .q3drc_mxfp8_fma_f32x or k == .q3rc_mxfp8_fma__draft,
        .mlx_chain => switch (k) {
            .q3_exl3_prep_in_rin, .q3_exl3_prep_din_rin, .q3_moeprep_dpost, .q3_prefill_dig_rot_take2_5120, .dsv41_prefill_dig_take2v_5120, .q3_prefill_dig_rot_roundx_2304, .q3_prefill_dig_rot_widen2_2304, .q3_prefill_dig_rot_widen1_5120, .dsv41_prefill_dig_rot_widen1_5120_obf16, .q3_prefill_fused_exl3x3_mul1lut_k3_bf16 => true,
            else => false,
        },
        .f64 => switch (k) {
            .q3_exl3_prep_gu_epi, .q3rc_router_tail, .q3rc_premix_fin, .q3dk_sinkhorn16_hc4_it20, .q3ht_combine, .q3ht_collapse_norm, .q3ht_combine_collapse_norm, .q3ht_mixfin, .q3_prefill_dig2_swiglu_2304_x => true,
            .mtplx_dsv4_sinkhorn_hc4_it20, .mtplx_dsv41_fp_rmsnorm_tg128_d1280, .mtplx_dsv41_fp_rmsnorm_rope_tg128_d512_rd64, .mtplx_dsv41_fp_rope_h64_hd512_rd64_fwd, .mtplx_dsv41_fp_rope_h64_hd512_rd64_inv => true,
            .q3rc_router_tail__n128_top3, .q3ht_combine__f32, .q3ht_collapse_norm__f32, .q3ht_combine_collapse_norm__f32, .q3ht_combine_collapse_norm__f32_rbf16 => true,
            else => isDigGemm(k),
        },
    };
}

fn isDigGemm(k: Kernel) bool {
    return k == .q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3 or k == .q3_prefill_dig_gemm_2304x5120_xmul1hk3 or twinOf(k) != null or
        k == .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128 or k == .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128;
}

/// The fused down GEMM's GEMM text (its `fused` check's reference is that text, then rot_widen1); null otherwise.
fn fusedOf(k: Kernel) ?Kernel {
    return switch (k) {
        .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128w1 => .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128,
        .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128w1 => .dsv41_prefill_dig_gemm_2304x5120_xmul1hk2_m128,
        else => null,
    };
}

/// A DIG-X GEMM's `twin` check reference: a 128-row text's 64-row text, the table-codebook text's 128-row text; null
/// for every other kernel.
fn twinOf(k: Kernel) ?Kernel {
    return switch (k) {
        .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128 => .q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3,
        .dsv41_prefill_dig_gemm_2304x5120_xmul1hk3_m128 => .q3_prefill_dig_gemm_2304x5120_xmul1hk3,
        .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut => .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128,
        else => null,
    };
}

/// A banked text's `twin` reference: the stock text it reads three banks for; null for every other kernel.
fn bankedTwinOf(k: Kernel) ?Kernel {
    return switch (k) {
        .dsv41_exl3_b3_mul1h_k3_2304 => .dsv41_exl3_mul1h_k3_2304,
        .dsv41_exl3_b3_mul1h_k3_5120 => .dsv41_exl3_mul1h_k3_5120,
        .dsv41_exl3_b3_prep_in_rin => .q3_exl3_prep_in_rin,
        .dsv41_exl3_b3_prep_gu_epi => .q3_exl3_prep_gu_epi,
        .dsv41_exl3_b3_prep_din_rin => .q3_exl3_prep_din_rin,
        .dsv41_exl3_b3_moeprep_dpost => .q3_moeprep_dpost,
        .dsv41_exl3_b3_pair_k3_5120 => .dsv41_exl3_pair_k3_5120,
        .dsv41_exl3_b3_guone_k3_2304 => .dsv41_exl3_guone_k3_2304,
        .dsv41_exl3_b3_mul1h_k2_2304 => .dsv41_exl3_mul1h_k2_2304,
        .dsv41_exl3_b3_mul1h_k2_5120 => .dsv41_exl3_mul1h_k2_5120,
        .dsv41_exl3_b3_pair_k2_5120 => .dsv41_exl3_pair_k2_5120,
        .dsv41_exl3_b3_guone_k2_2304 => .dsv41_exl3_guone_k2_2304,
        else => null,
    };
}

/// A banked text against its stock text, once per bank: the generated slot ids are packed with bank b (b<<24 | row),
/// and the stock call takes bank b's arrays plus the shared inputs; every output word of every bank pass.
fn checkBankedTwin(h: *H, k: Kernel) !void {
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    const e = h.reg.get(k);
    const stock = bankedTwinOf(k).?;
    const es = h.reg.get(stock);
    var vars = defaultVars(e);
    vars.set(.rows, 6);
    vars.set(.cap, 16);
    var svars = defaultVars(es);
    svars.set(.rows, 6);
    svars.set(.cap, 16);
    const wave = Wave.even(1, 6, 16);
    var ins = try genAll(h, &sc, e, &vars, null, &wave);
    const ids_at = for (e.inputs, 0..) |ea, jj| {
        if (std.mem.eql(u8, ea.name, "ids")) break jj;
    } else return error.BankedTwinInput;
    const ids_bytes = try hostCopy(h, ins[ids_at]);
    defer h.a.free(ids_bytes);
    const n_ids = ids_bytes.len / 4;
    if (n_ids == 0 or n_ids > 64) return error.BankedTwinInput;
    var plain: [64]u32 = undefined;
    @memcpy(std.mem.sliceAsBytes(plain[0..n_ids]), ids_bytes[0 .. n_ids * 4]);
    const shape = [_]c_int{@intCast(n_ids)};
    var bad: u64 = 0;
    var words: u64 = 0;
    for (0..3) |b| {
        var packed_ids: [64]u32 = undefined;
        for (0..n_ids) |r| packed_ids[r] = (@as(u32, @intCast(b)) << 24) | plain[r];
        ins[ids_at] = try fromHost(&sc, std.mem.sliceAsBytes(packed_ids[0..n_ids]), &shape, .uint32);
        var sins: [inputs_max]mlx.mlx_array = @splat(.{});
        for (es.inputs, 0..) |arg, i| {
            var nb: [64]u8 = undefined;
            const bank_name = std.fmt.bufPrint(&nb, "{s}{d}", .{ arg.name, b }) catch unreachable;
            const j = for (e.inputs, 0..) |ea, jj| {
                if (std.mem.eql(u8, ea.name, arg.name) or (ea.role == .bank and std.mem.eql(u8, ea.name, bank_name))) break jj;
            } else return error.BankedTwinInput;
            sins[i] = if (j == ids_at) try fromHost(&sc, std.mem.sliceAsBytes(plain[0..n_ids]), &shape, .uint32) else ins[j];
        }
        const got = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, null);
        const want = try launch(h, &sc, stock, sins[0..es.inputs.len], &svars, null);
        for (0..e.outputs.len) |o| {
            const g_ = try hostCopy(h, got[o]);
            defer h.a.free(g_);
            const w_ = try hostCopy(h, want[o]);
            defer h.a.free(w_);
            words += g_.len / 4;
            bad += if (g_.len == w_.len) countDiff(w_, g_, 4) else g_.len / 4;
        }
    }
    try h.record(.{ .kernel = k, .check = .twin, .words = words, .bad = bad, .ok = bad == 0 and words > 0 });
}

/// A routed decode form's `twin` reference: the stock mul1h text of its projection; null for every other kernel.
fn formTwinOf(k: Kernel) ?Kernel {
    return switch (k) {
        .dsv41_exl3_pair_k3_5120 => .dsv41_exl3_mul1h_k3_5120,
        .dsv41_exl3_guone_k3_2304 => .dsv41_exl3_mul1h_k3_2304,
        .dsv41_exl3_pair_k2_5120 => .dsv41_exl3_mul1h_k2_5120,
        .dsv41_exl3_guone_k2_2304 => .dsv41_exl3_mul1h_k2_2304,
        else => null,
    };
}

/// A wave's slot runs for the form twins: single rows, pairs, odd and even runs.
const form_twin_runs = [_]u32{ 1, 2, 3, 6, 1, 4, 1, 5 };

/// A routed form against the stock mul1h text on the same inputs with ids grouped in runs: every output word (the
/// pair text: its one output; gate + up in one launch: each half against its own stock launch).
fn checkFormTwin(h: *H, k: Kernel) !void {
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    const e = h.reg.get(k);
    const stock = formTwinOf(k).?;
    const es = h.reg.get(stock);
    var vars = defaultVars(e);
    var rows: u64 = 0;
    for (form_twin_runs) |r| rows += r;
    vars.set(.rows, rows);
    vars.set(.cap, 16);
    var svars = defaultVars(es);
    svars.set(.rows, rows);
    svars.set(.cap, 16);
    const wave = Wave.even(1, rows, 16);
    var ins = try genAll(h, &sc, e, &vars, null, &wave);
    var sins = try genAll(h, &sc, es, &svars, null, &wave);
    var ids: [64]u32 = undefined;
    var n: usize = 0;
    for (form_twin_runs, 0..) |r, j| for (0..r) |_| {
        ids[n] = @intCast((j * 5 + 3) % 16);
        n += 1;
    };
    const shape = [_]c_int{@intCast(rows)};
    const id_arr = try fromHost(&sc, std.mem.sliceAsBytes(ids[0..n]), &shape, .uint32);
    var bad: u64 = 0;
    var words: u64 = 0;
    if (k == .dsv41_exl3_pair_k3_5120 or k == .dsv41_exl3_pair_k2_5120) {
        ins[1] = id_arr;
        sins[0] = ins[0];
        sins[1] = id_arr;
        sins[2] = ins[2];
        const got = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, null);
        const want = try launch(h, &sc, stock, sins[0..es.inputs.len], &svars, null);
        const g_ = try hostCopy(h, got[0]);
        defer h.a.free(g_);
        const w_ = try hostCopy(h, want[0]);
        defer h.a.free(w_);
        words += g_.len / 4;
        bad += if (g_.len == w_.len) countDiff(w_, g_, 4) else g_.len / 4;
    } else {
        // inputs xh_g, xh_u, ids, code_g, code_u
        ins[2] = id_arr;
        const got = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, null);
        for ([_]usize{ 0, 1 }, [_]usize{ 3, 4 }, 0..) |xi, ci, o| {
            sins[0] = ins[xi];
            sins[1] = id_arr;
            sins[2] = ins[ci];
            const want = try launch(h, &sc, stock, sins[0..es.inputs.len], &svars, null);
            const g_ = try hostCopy(h, got[o]);
            defer h.a.free(g_);
            const w_ = try hostCopy(h, want[0]);
            defer h.a.free(w_);
            words += g_.len / 4;
            bad += if (g_.len == w_.len) countDiff(w_, g_, 4) else g_.len / 4;
        }
    }
    try h.record(.{ .kernel = k, .check = .twin, .words = words, .bad = bad, .ok = bad == 0 and words > 0 });
}

/// How much of each kernel's manifest plan runs.
/// - `compile` (the construction's acceptance): every kernel compiles, binds and completes one launch on generated inputs;
///   no numeric comparison (David 10-05: the server's startup runs no numeric probes). A kernel that does not compile or
///   bind refuses the load by name (SelfCheckFailed).
/// - `full` (the device test, `DSV41_SELFCHECK_DEVICE`, and the window smoke tests): every check of the manifest: the
///   exact references (host f64, the MLX chain, the decode table, the join, golden tiles, compositions), the per-site row
///   invariance and the twins. The numeric probes live there only.
pub const Depth = kset.CheckDepth;

/// The checks of `want`'s entries at `depth`, per kernel, in registry order (`stubPlan` runs the same plan).
pub fn plan(reg: *const xk.Registry, want: std.EnumSet(Kernel), depth: Depth) std.EnumArray(Kernel, std.EnumSet(Check)) {
    var out = scaffold.plan(&reg.entries, want, depth);
    // layout_guard is never run: its oracle was MLX's own mxfp8 quantized_matmul (MLX's to verify, not ours);
    // the rcproj kernels' f64 checks cover them.
    for (&reg.entries) |*e| {
        var cs = out.get(e.kernel);
        cs.remove(.layout_guard);
        out.set(e.kernel, cs);
    }
    return out;
}

/// Runs every check of `subset`'s plan (one consumer's kernels: `kernel_set.Set.selfCheck`) at `depth`, in registry
/// order; a failing check is recorded (with the latched MLX message) and the run continues, so one window reports
/// every kernel.
pub fn runSubset(a: Allocator, reg: *const xk.Registry, bound: *const xk.Bound, subset: []const Kernel, depth: Depth, report: *Report) !void {
    var want: std.EnumSet(Kernel) = .empty;
    for (subset) |k| want.insert(k);
    return runOver(a, reg, bound, want, depth, report);
}

fn runOver(a: Allocator, reg: *const xk.Registry, bound: *const xk.Bound, want: std.EnumSet(Kernel), depth: Depth, report: *Report) !void {
    const table = try a.create([65536]u16);
    defer a.destroy(table);
    xk.mul1Table(table);
    var h: H = .{ .a = a, .reg = reg, .bound = bound, .s = bound.stream, .rng = .init(20260928), .report = report, .table = table };
    const checks = plan(reg, want, depth);
    for (&reg.entries) |*e| {
        if (!want.contains(e.kernel)) continue;
        var it = checks.get(e.kernel).iterator();
        while (it.next()) |c| {
            const before = report.results.items.len;
            h.dispatch(e.kernel, c) catch |err| {
                var buf: [512]u8 = undefined;
                const msg = mlx.takeError(&buf) orelse "";
                log.err("[exl3 selfcheck] {t} {t}: {t} {s}\n", .{ e.kernel, c, err, msg });
                try report.appendRaised(a, e.kernel, c, err, msg);
            };
            if (report.results.items.len == before)
                try report.results.append(a, .{ .kernel = e.kernel, .check = c, .ok = false, .err = "no result" });
        }
    }
}


/// The acceptance verdict over a finished plan: refused (error.SelfCheckFailed, `diag` naming
/// the first failing kernel / check / site) unless every result passed and the plan ran.
pub fn judge(report: *const Report, diag: ?*xk.Diag) error{SelfCheckFailed}!void {
    logResults(report, if (env.get(.DSV41_SELFCHECK_REPORT)) |v| v[0] == '1' else false);
    return scaffold.judge(report, "exl3 kernels", diag);
}

/// One line per failed result (every result when `all`): kernel, check, site, words / bad, metric / limit, the error
/// and the reference side. Construction only (the acceptance gate), never in a measured path.
pub fn logResults(report: *const Report, all: bool) void {
    var n_fail: usize = 0;
    for (report.results.items) |r| {
        n_fail += @intFromBool(!r.ok);
        if (r.ok and !all) continue;
        if (r.ok) log.info("[exl3 selfcheck] {s} {t} {t} site={s} words={d} bad={d} metric={e} limit={e} err={s} msg={s} reference={s}\n", .{ if (r.ok) "PASS" else "FAIL", r.kernel, r.check, r.site, r.words, r.bad, r.metric, r.limit, r.err, r.msg, referenceOf(r.check) }) else log.err("[exl3 selfcheck] {s} {t} {t} site={s} words={d} bad={d} metric={e} limit={e} err={s} msg={s} reference={s}\n", .{ if (r.ok) "PASS" else "FAIL", r.kernel, r.check, r.site, r.words, r.bad, r.metric, r.limit, r.err, r.msg, referenceOf(r.check) });
    }
    if (n_fail > 0) log.err("[exl3 selfcheck] {d} results, {d} failed\n", .{ report.results.items.len, n_fail }) else if (all) log.info("[exl3 selfcheck] {d} results, {d} failed\n", .{ report.results.items.len, n_fail });
}

const inputs_max = xk.max_inputs;
const t128_scale: f32 = @bitCast(@as(u32, 0x3DB504F3));

const H = struct {
    a: Allocator,
    reg: *const xk.Registry,
    bound: *const xk.Bound,
    s: mlx.mlx_stream,
    rng: std.Random.DefaultPrng,
    report: *Report,
    table: *const [65536]u16,

    fn record(h: *H, r: Result) !void {
        try h.report.results.append(h.a, r);
    }

    fn dispatch(h: *H, k: Kernel, c: Check) !void {
        switch (c) {
            .compile => try checkCompile(h, k),
            .row_invariance => try checkRowInvariance(h, k),
            .decode_table => try checkDecodeTable(h, k),
            .golden_tiles => try checkGoldenTiles(h, k),
            .mlx_chain => try checkChain(h, k),
            .f64 => try checkF64(h, k),
            .layout_guard => unreachable, // never planned (runOver)
            .composition => try checkComposition(h, k),
            .join_equiv => try checkJoinEquiv(h, k),
            .twin => if (bankedTwinOf(k) != null) try checkBankedTwin(h, k) else if (formTwinOf(k) != null) try checkFormTwin(h, k) else try checkTwin(h, k),
            .fused => try checkFused(h, k),
        }
    }
};

/// MLX arrays created by one check, freed together.
const Scope = kset.Scope;

fn op(sc: *Scope, comptime f: anytype, args: anytype) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    mlx.check(@call(.auto, f, .{&r} ++ args)) catch |e| {
        _ = mlx.mlx_array_free(r);
        return e;
    };
    return sc.keep(r);
}

fn dtypeSize(dt: mlx.mlx_dtype) usize {
    return switch (dt) {
        .bool_, .uint8, .int8 => 1,
        .uint16, .int16, .float16, .bfloat16 => 2,
        .uint32, .int32, .float32 => 4,
        .uint64, .int64, .float64, .complex64 => 8,
    };
}

const bf16Bits = @import("sdk").io_util.bf16Rne;

fn putFloat(buf: []u8, i: usize, dt: mlx.mlx_dtype, v: f64) void {
    switch (dt) {
        .float32 => std.mem.writeInt(u32, buf[i * 4 ..][0..4], @bitCast(@as(f32, @floatCast(v))), .little),
        .float16 => std.mem.writeInt(u16, buf[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little),
        .bfloat16 => std.mem.writeInt(u16, buf[i * 2 ..][0..2], bf16Bits(@floatCast(v)), .little),
        else => unreachable,
    }
}

fn putInt(buf: []u8, i: usize, dt: mlx.mlx_dtype, v: i64) void {
    switch (dt) {
        .bool_ => buf[i] = @intFromBool(v != 0),
        .int32 => std.mem.writeInt(i32, buf[i * 4 ..][0..4], @intCast(v), .little),
        .uint32 => std.mem.writeInt(u32, buf[i * 4 ..][0..4], @intCast(v), .little),
        .int16 => std.mem.writeInt(i16, buf[i * 2 ..][0..2], @intCast(v), .little),
        .uint8 => buf[i] = @intCast(v),
        else => unreachable,
    }
}

fn getF64(bytes: []const u8, i: usize, dt: mlx.mlx_dtype) f64 {
    return switch (dt) {
        .float32 => @as(f32, @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little))),
        .float16 => @as(f16, @bitCast(std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little))),
        .bfloat16 => @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16)),
        .int32 => @floatFromInt(std.mem.readInt(i32, bytes[i * 4 ..][0..4], .little)),
        .uint32 => @floatFromInt(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little)),
        .int16 => @floatFromInt(std.mem.readInt(i16, bytes[i * 2 ..][0..2], .little)),
        .uint8 => @floatFromInt(bytes[i]),
        else => unreachable,
    };
}

/// A DIG wave: `n` experts, expert j at slot `slots[j]` with `rows[j]` consecutive rows.
const Wave = struct {
    n: usize = 0,
    slots: [16]u32 = @splat(0),
    rows: [16]u32 = @splat(0),

    fn even(experts: u64, rows: u64, cap: u64) Wave {
        const n: u64 = @max(1, @min(experts, 16));
        var w: Wave = .{ .n = @intCast(n) };
        for (0..w.n) |j| {
            const jj: u64 = @intCast(j);
            w.slots[j] = @intCast(jj % @max(cap, 1));
            w.rows[j] = @intCast(rows / n + @intFromBool(jj < rows % n));
        }
        return w;
    }

    fn total(w: *const Wave) u64 {
        var t: u64 = 0;
        for (w.rows[0..w.n]) |r| t += r;
        return t;
    }

    /// q3_prefill_dig_candidate.wave_table: slot, first row, rows, first threadgroup per expert
    /// (unused threadgroup entries INT32_MAX), then [n, threadgroups], at `bm`-row M tiles. Returns the threadgroups.
    fn table(w: *const Wave, tiles: u32, bm: u32, out: *[80]i32) u64 {
        @memset(out, 0);
        var row0: i64 = 0;
        var tg: i64 = 0;
        for (0..16) |j| {
            if (j >= w.n) {
                out[48 + j] = std.math.maxInt(i32);
                continue;
            }
            out[j] = @intCast(w.slots[j]);
            out[16 + j] = @intCast(row0);
            out[32 + j] = @intCast(w.rows[j]);
            out[48 + j] = @intCast(tg);
            row0 += w.rows[j];
            tg += @as(i64, @intCast((w.rows[j] + bm - 1) / bm)) * tiles;
        }
        out[64] = @intCast(w.n);
        out[65] = @intCast(tg);
        return @intCast(tg);
    }
};

fn defaultVars(e: *const Entry) Vars {
    var v: Vars = .initFill(0);
    v.set(.rows, if (e.rows_max > 0) e.rows_max else 8);
    v.set(.cap, 4);
    v.set(.m_tokens, 3);
    v.set(.experts, 2);
    v.set(.a_rows, 16);
    v.set(.seq, v.get(.rows));
    // decode batch 2: the attention's key count (the tier's 640, clamped into an entry's key
    // range: 512 on the ls 32 text, 640 on ls 128) and the index top-k at a 2,048-entry history
    // (k = width = 512, not all finite)
    inline for (.{ .{ Var.keys, 640 }, .{ Var.ncomp, 2048 }, .{ Var.topk, 512 }, .{ Var.width, 512 } }) |d| {
        const b = e.bounds.get(d[0]) orelse .{ 1, std.math.maxInt(u64) };
        v.set(d[0], std.math.clamp(@as(u64, d[1]), b[0], b[1]));
    }
    v.set(.allfin, @intFromBool(v.get(.topk) >= v.get(.ncomp)));
    // prefill batch 2: a 256-row window store, a 700-row compressed store (the lane's warm) and
    // the full 512-key selection (k = 640, the tier's common key count)
    v.set(.ring, 256);
    v.set(.store, 700);
    // JOINLESS: 24 sources of 97 rows each (an odd count: no source is a tile multiple)
    v.set(.src, 97);
    const kc = e.bounds.get(.kc) orelse .{ 1, 512 };
    v.set(.kc, std.math.clamp(@as(u64, 512), kc[0], kc[1]));
    return v;
}

fn fillArg(h: *H, buf: []u8, n: usize, arg: *const xk.Arg, vars: *const Vars, wave: *const Wave) void {
    const r = h.rng.random();
    const d = arg.domain;
    switch (d.kind) {
        .bits => r.bytes(buf),
        .zeros => @memset(buf, 0),
        .normal => for (0..n) |i| putFloat(buf, i, arg.dtype, r.floatNorm(f64) * d.scale),
        .uniform => for (0..n) |i| putFloat(buf, i, arg.dtype, d.lo + (d.hi - d.lo) * r.float(f64)),
        .signed_pow2 => for (0..n) |i| {
            const sign: f64 = if (r.boolean()) 1 else -1;
            putFloat(buf, i, arg.dtype, sign * std.math.exp2(d.lo + (d.hi - d.lo) * r.float(f64)));
        },
        .values => for (0..n) |i| {
            if (d.floats.len > 0) putFloat(buf, i, arg.dtype, d.floats[i % d.floats.len]) else putInt(buf, i, arg.dtype, d.ints[i % d.ints.len]);
        },
        .index => {
            const hi = @max(vars.get(d.of.?), 1);
            for (0..n) |i| putInt(buf, i, arg.dtype, @intCast(r.uintLessThan(u64, hi)));
        },
        .range => {
            const lo: i64 = @intFromFloat(d.lo);
            const span: u64 = @intFromFloat(d.hi - d.lo);
            for (0..n) |i| putInt(buf, i, arg.dtype, lo + @as(i64, @intCast(r.uintLessThan(u64, span))));
        },
        .wave_table => {
            var t: [80]i32 = undefined;
            _ = wave.table(d.tiles, d.bm, &t);
            for (0..n) |i| putInt(buf, i, arg.dtype, t[i]);
        },
        .slots => for (0..n) |i| putInt(buf, i, arg.dtype, wave.slots[if (i < wave.n) i else 0]),
        .@"var" => unreachable,
        // (source, row) pairs over the JOINLESS sources: every assignment a random row of a random source
        .join_table => for (0..n / 2) |j| {
            putInt(buf, 2 * j, arg.dtype, @intCast(r.uintLessThan(u64, join_sources)));
            putInt(buf, 2 * j + 1, arg.dtype, @intCast(r.uintLessThan(u64, @max(vars.get(.src), 1))));
        },
    }
}

/// JOINLESS binds this many call-output sources (the lane's NSRC; unused slots alias source 0).
const join_sources = 24;

fn argShape(arg: *const xk.Arg, vars: *const Vars, shape: *[xk.max_rank]c_int) usize {
    var n: usize = 1;
    for (arg.shape, 0..) |d, i| {
        shape[i] = @intCast(d.eval(vars));
        n *= @intCast(shape[i]);
    }
    return n;
}

fn genInput(h: *H, sc: *Scope, arg: *const xk.Arg, vars: *const Vars, wave: *const Wave) !mlx.mlx_array {
    if (arg.role == .scalar) return sc.keep(mlx.mlx_array_new_int(@intCast(vars.get(arg.domain.of.?))));
    var shape: [xk.max_rank]c_int = undefined;
    const n = argShape(arg, vars, &shape);
    // The input's host bytes on mapped pages, unmapped once MLX copied them: through libc malloc each freed large block
    // stayed dirty in the footprint (pass3eo's allocation trace: 2.0 GB of construction self-check inputs, one 1.32 GB).
    const buf = try std.heap.page_allocator.alloc(u8, n * dtypeSize(arg.dtype));
    defer std.heap.page_allocator.free(buf);
    fillArg(h, buf, n, arg, vars, wave);
    return sc.keep(mlx.mlx_array_new_data(buf.ptr, &shape, @intCast(arg.shape.len), arg.dtype));
}

fn fromHost(sc: *Scope, bytes: []const u8, shape: []const c_int, dt: mlx.mlx_dtype) !mlx.mlx_array {
    return sc.keep(mlx.mlx_array_new_data(bytes.ptr, shape.ptr, @intCast(shape.len), dt));
}

/// Inputs of `e` at `vars` (the site's vars and the GEMMs' threadgroups filled in first).
fn genAll(h: *H, sc: *Scope, e: *const Entry, vars: *Vars, site: ?*const xk.Site, wave: *const Wave) ![inputs_max]mlx.mlx_array {
    if (site) |s| xk.siteVars(s, vars);
    for (e.inputs) |*arg| {
        if (arg.domain.kind == .wave_table and arg.domain.tiles > 0) {
            var t: [80]i32 = undefined;
            vars.set(.tgs, wave.table(arg.domain.tiles, arg.domain.bm, &t));
        }
    }
    var ins: [inputs_max]mlx.mlx_array = @splat(.{});
    for (e.inputs, 0..) |*arg, i| ins[i] = try genInput(h, sc, arg, vars, wave);
    return ins;
}

fn inputIndex(e: *const Entry, name: []const u8) usize {
    for (e.inputs, 0..) |arg, i| if (std.mem.eql(u8, arg.name, name)) return i;
    unreachable;
}

fn siteName(site: ?*const xk.Site) ?[]const u8 {
    return if (site) |s| s.name else null;
}

/// One launch of `k`, evaluated; outputs belong to `sc`.
fn launch(h: *H, sc: *Scope, k: Kernel, ins: []const mlx.mlx_array, vars: *const Vars, site: ?[]const u8) ![xk.max_outputs]mlx.mlx_array {
    const e = h.reg.get(k);
    const cfg = try xk.launchFor(e, vars, site);
    var outs: [xk.max_outputs]mlx.mlx_array = @splat(.{});
    try h.bound.apply(k, ins, &cfg, outs[0..cfg.n_out]);
    for (outs[0..cfg.n_out]) |o| _ = try sc.keep(o);
    const v = mlx.mlx_vector_array_new_data(&outs, cfg.n_out);
    defer _ = mlx.mlx_vector_array_free(v);
    try mlx.check(mlx.mlx_eval(v));
    if (mlx.errorPending()) return error.MlxError;
    return outs;
}

/// A row-contiguous host copy of `arr` (caller frees).
fn hostCopy(h: *H, arr: mlx.mlx_array) ![]u8 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, arr, false, h.s));
    try mlx.check(mlx.mlx_array_eval(c));
    if (mlx.errorPending()) return error.MlxError;
    const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
    const p = mlx.mlx_array_data_uint8(c) orelse return error.MlxArrayDataNull;
    return h.a.dupe(u8, p[0..n]);
}

fn hostF64(h: *H, arr: mlx.mlx_array) ![]f64 {
    const bytes = try hostCopy(h, arr);
    defer h.a.free(bytes);
    const dt = mlx.mlx_array_dtype(arr);
    const out = try h.a.alloc(f64, bytes.len / dtypeSize(dt));
    for (out, 0..) |*o, i| o.* = getF64(bytes, i, dt);
    return out;
}

fn countDiff(a: []const u8, b: []const u8, size: usize) u64 {
    var bad: u64 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += size) bad += @intFromBool(!std.mem.eql(u8, a[i..][0..size], b[i..][0..size]));
    return bad;
}

// ── MLX op helpers (the eager chains the fused kernels replace) ──

fn take0(sc: *Scope, a: mlx.mlx_array, idx: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    return op(sc, mlx.mlx_take_axis, .{ a, idx, @as(c_int, 0), s });
}

fn f32Of(sc: *Scope, a: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    return op(sc, mlx.mlx_astype, .{ a, mlx.mlx_dtype.float32, s });
}

fn mul(sc: *Scope, a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    return op(sc, mlx.mlx_multiply, .{ a, b, s });
}

fn reshape(sc: *Scope, a: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return op(sc, mlx.mlx_reshape, .{ a, shape.ptr, shape.len, s });
}

/// tcq_runtime._t128 / hfold RotOps._t: f32, the last axis in 128-blocks through
/// mx.hadamard_transform(scale = 128^-0.5), back to `out_shape`.
fn t128(sc: *Scope, v: mlx.mlx_array, out_shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var lead: c_int = 1;
    for (out_shape[0 .. out_shape.len - 1]) |d| lead *= d;
    const n = out_shape[out_shape.len - 1];
    const blocks = try reshape(sc, try f32Of(sc, v, s), &.{ lead, @divExact(n, 128), 128 }, s);
    const hd = try op(sc, mlx.mlx_hadamard_transform, .{ blocks, mlx.mlx_optional_float.some(t128_scale), s });
    return reshape(sc, hd, out_shape, s);
}

/// take(bank, take(table, rhs)).f32 reshaped to [rows, 1, n]: a DIG row's slot values.
fn slotRows(sc: *Scope, bank: mlx.mlx_array, table: mlx.mlx_array, rhs: mlx.mlx_array, rows: c_int, n: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return reshape(sc, try f32Of(sc, try take0(sc, bank, try take0(sc, table, rhs, s), s), s), &.{ rows, 1, n }, s);
}

fn expectWords(h: *H, k: Kernel, c: Check, want: mlx.mlx_array, got: mlx.mlx_array, words: *u64, bad: *u64) !void {
    const w = try hostCopy(h, want);
    defer h.a.free(w);
    const g = try hostCopy(h, got);
    defer h.a.free(g);
    const size = dtypeSize(mlx.mlx_array_dtype(got));
    if (w.len != g.len or mlx.mlx_array_dtype(want) != mlx.mlx_array_dtype(got)) {
        log.err("[exl3 selfcheck] {t} {t}: reference {d} B {t} vs kernel {d} B {t}\n", .{ k, c, w.len, mlx.mlx_array_dtype(want), g.len, mlx.mlx_array_dtype(got) });
        bad.* += g.len / size + 1;
        return;
    }
    words.* += g.len / size;
    const n_bad = countDiff(w, g, size);
    bad.* += n_bad;
    if (n_bad > 0) logWordDiff(k, c, mlx.mlx_array_dtype(got), mlx.mlx_array_shape(got)[0..@intCast(mlx.mlx_array_ndim(got))], w, g, size, n_bad);
}

/// A failed bitwise comparison, once, at construction: the shape, how many words differ, the first one (reference
/// and kernel values) and the largest absolute difference. The reference side is the check's (`referenceOf`).
fn logWordDiff(k: Kernel, c: Check, dt: mlx.mlx_dtype, shape: []const c_int, want: []const u8, got: []const u8, size: usize, n_bad: u64) void {
    var first: ?usize = null;
    var max_abs: f64 = 0;
    const numeric = switch (dt) {
        .float32, .float16, .bfloat16, .int32, .uint32, .int16, .uint8 => true,
        else => false,
    };
    var i: usize = 0;
    while (i * size < got.len) : (i += 1) {
        if (std.mem.eql(u8, want[i * size ..][0..size], got[i * size ..][0..size])) continue;
        if (first == null) first = i;
        if (!numeric) continue;
        const d = @abs(getF64(got, i, dt) - getF64(want, i, dt));
        if (!(d <= max_abs)) max_abs = d; // NaN wins
    }
    const f = first.?;
    if (!numeric) {
        log.err("[exl3 selfcheck] {t} {t}: {d} of {d} words differ ({t} {any}); first at word {d}; reference = {s}\n", .{ k, c, n_bad, got.len / size, dt, shape, f, referenceOf(c) });
        return;
    }
    log.err("[exl3 selfcheck] {t} {t}: {d} of {d} words differ ({t} {any}); first at {d}: reference {e} kernel {e}; max abs error {e}; reference = {s}\n", .{ k, c, n_bad, got.len / size, dt, shape, f, getF64(want, f, dt), getF64(got, f, dt), max_abs, referenceOf(c) });
}

/// What a check compares the kernel against (the side a failure line names as the reference).
pub fn referenceOf(c: Check) []const u8 {
    return switch (c) {
        .compile => "a launch that completes",
        .row_invariance => "the kernel's own full-M call",
        .decode_table => "the stored trellis decode table",
        .golden_tiles => "the stored golden tiles",
        .mlx_chain => "the eager MLX op chain (bitwise)",
        .f64 => "a host float64 reference",
        .layout_guard => "MLX's mxfp8 quantized_matmul",
        .composition => "the composition of the registered kernels",
        .twin => "the registered twin kernel",
        .fused => "the unfused kernel text then rot_widen1",
        .join_equiv => "the registered q3sk_combine on the joined rows",
    };
}

// ── compile ──

/// JOINLESS == SMALLK over the joined array, every word: the sources, the (source, row) table,
/// the weights and the shared rows drawn once per shape; the reference joins the rows the table
/// names (`take(concatenate(sources), source * src + row)`) and runs the registered q3sk_combine
/// (itself proven == the stock multiply / col_reduce_small / add chain), at a partial chunk and a
/// full one.
fn checkJoinEquiv(h: *H, k: Kernel) !void {
    const e = h.reg.get(k);
    if (e.inputs.len != join_sources + 3) return error.SchemaInvalid;
    var words: u64 = 0;
    var bad: u64 = 0;
    for ([_]u64{ 183, 953 }) |rows| {
        var sc: Scope = .{ .a = h.a };
        defer sc.deinit();
        var vars = defaultVars(e);
        vars.set(.rows, rows);
        const wave = Wave.even(vars.get(.experts), rows, vars.get(.cap));
        const ins = try genAll(h, &sc, e, &vars, null, &wave);
        const got = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, null);
        // the reference: the rows the table names, joined, through SMALLK
        const loc = try hostCopy(h, ins[join_sources]);
        defer h.a.free(loc);
        const n_as: usize = @intCast(rows * 6);
        const glob = try h.a.alloc(i32, n_as);
        defer h.a.free(glob);
        const src: i32 = @intCast(vars.get(.src));
        for (glob, 0..) |*g, j| {
            const s_i = std.mem.readInt(i32, loc[8 * j ..][0..4], .little);
            const r_i = std.mem.readInt(i32, loc[8 * j + 4 ..][0..4], .little);
            g.* = s_i * src + r_i;
        }
        const v = mlx.mlx_vector_array_new_data(&ins, join_sources);
        defer _ = mlx.mlx_vector_array_free(v);
        const flat = try op(&sc, mlx.mlx_concatenate_axis, .{ v, @as(c_int, 0), h.s });
        const gi = try fromHost(&sc, std.mem.sliceAsBytes(glob), &.{@intCast(n_as)}, .int32);
        const routed = try reshape(&sc, try take0(&sc, flat, gi, h.s), &.{ @intCast(rows), 6, 5120 }, h.s);
        // (the bf16 sources' text against SMALLK's bf16-routed instantiation)
        const want = try launch(h, &sc, if (k == .dsv41_jl_combine_bf16) .q3sk_combine__rbf16 else .q3sk_combine, &.{ routed, ins[join_sources + 1], ins[join_sources + 2] }, &vars, null);
        const a = try hostCopy(h, got[0]);
        defer h.a.free(a);
        const b = try hostCopy(h, want[0]);
        defer h.a.free(b);
        words += a.len / 4;
        bad += countDiff(a, b, 4);
    }
    try h.record(.{ .kernel = k, .check = .join_equiv, .words = words, .bad = bad, .ok = bad == 0 and words > 0 });
}

fn checkCompile(h: *H, k: Kernel) !void {
    const e = h.reg.get(k);
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    var vars = defaultVars(e);
    const site: ?*const xk.Site = if (e.sites.len > 0) &e.sites[0] else null;
    const wave = Wave.even(vars.get(.experts), vars.get(.rows), vars.get(.cap));
    const ins = try genAll(h, &sc, e, &vars, site, &wave);
    _ = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, siteName(site));
    try h.record(.{ .kernel = k, .check = .compile, .site = if (site) |s| s.name else "", .ok = true });
}

// ── row invariance: every M = 1..rows_max at every slot == those rows of the full call ──

fn checkRowInvariance(h: *H, k: Kernel) !void {
    const e = h.reg.get(k);
    if (e.sites.len == 0) return rowInvariance(h, k, null, 1);
    for (e.sites) |*s| {
        // rcproj's budget: at least 2 sets, ~2^19 compared words per site (a last-bit f32
        // difference flips a bf16 word with probability ~2^-15).
        const per_set = @as(u64, e.rows_max) * s.G * s.N;
        try rowInvariance(h, k, s, @max(2, ((1 << 19) + per_set - 1) / per_set));
    }
}

fn sliceRows(h: *H, sc: *Scope, arr: mlx.mlx_array, arg: *const xk.Arg, vars: *const Vars, slot: usize, m: usize) !mlx.mlx_array {
    var start: [xk.max_rank]c_int = @splat(0);
    var stop: [xk.max_rank]c_int = undefined;
    _ = argShape(arg, vars, &stop);
    const strides: [xk.max_rank]c_int = @splat(1);
    start[arg.row_axis] = @intCast(slot);
    stop[arg.row_axis] = @intCast(slot + m);
    const nd = arg.shape.len;
    return op(sc, mlx.mlx_slice, .{ arr, &start, nd, &stop, nd, &strides, nd, h.s });
}

/// [words, mismatches] of `got` (an m-row call) against rows slot..slot+m of `full`.
fn compareRows(full: []const u8, got: []const u8, o: *const xk.Arg, vars: *const Vars, slot: usize, m: usize) [2]u64 {
    var shape: [xk.max_rank]c_int = undefined;
    _ = argShape(o, vars, &shape);
    const ax = o.row_axis;
    const rows: usize = @intCast(shape[ax]);
    const size = dtypeSize(o.dtype);
    var outer: usize = 1;
    for (shape[0..ax]) |d| outer *= @intCast(d);
    var inner: usize = size;
    for (shape[ax + 1 .. o.shape.len]) |d| inner *= @intCast(d);
    var bad: u64 = 0;
    for (0..outer) |b| {
        for (0..m) |r| bad += countDiff(full[(b * rows + slot + r) * inner ..][0..inner], got[(b * m + r) * inner ..][0..inner], size);
    }
    return .{ outer * m * inner / size, bad };
}

/// The weights / banks / tables are drawn once; each set draws new row inputs (rcproj's scheme).
fn rowInvariance(h: *H, k: Kernel, site: ?*const xk.Site, sets: u64) !void {
    const e = h.reg.get(k);
    const rmax: usize = e.rows_max;
    var words: u64 = 0;
    var bad: u64 = 0;
    var once: Scope = .{ .a = h.a };
    defer once.deinit();
    var vars = defaultVars(e);
    vars.set(.rows, rmax);
    const wave = Wave.even(vars.get(.experts), rmax, vars.get(.cap));
    var ins = try genAll(h, &once, e, &vars, site, &wave);
    for (0..sets) |set| {
        var sc: Scope = .{ .a = h.a };
        defer sc.deinit();
        if (set > 0) {
            for (e.inputs, 0..) |*arg, i| {
                if (arg.role == .rows) ins[i] = try genInput(h, &sc, arg, &vars, &wave);
            }
        }
        const full = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, siteName(site));
        var full_host: [xk.max_outputs][]u8 = undefined;
        var n_host: usize = 0;
        defer for (full_host[0..n_host]) |b| h.a.free(b);
        for (0..e.outputs.len) |oi| {
            full_host[oi] = try hostCopy(h, full[oi]);
            n_host += 1;
        }
        for (1..rmax + 1) |m| {
            for (0..rmax - m + 1) |slot| {
                var sc2: Scope = .{ .a = h.a };
                defer sc2.deinit();
                var ins2 = ins;
                for (e.inputs, 0..) |*arg, i| switch (arg.role) {
                    .rows => ins2[i] = try sliceRows(h, &sc2, ins[i], arg, &vars, slot, m),
                    // a row-count scalar (rows, or seq = rows here) follows the m-row call; any
                    // other scalar (the index top-k's N / K / W / flag) keeps its value
                    .scalar => ins2[i] = try sc2.keep(mlx.mlx_array_new_int(@intCast(switch (arg.domain.of.?) {
                        .rows, .seq => m,
                        else => vars.get(arg.domain.of.?),
                    }))),
                    else => {},
                };
                var v2 = vars;
                v2.set(.rows, m);
                const outs = try launch(h, &sc2, k, ins2[0..e.inputs.len], &v2, siteName(site));
                for (e.outputs, 0..) |*o, oi| {
                    const got = try hostCopy(h, outs[oi]);
                    defer h.a.free(got);
                    const r = compareRows(full_host[oi], got, o, &vars, slot, m);
                    words += r[0];
                    bad += r[1];
                }
            }
        }
    }
    try h.record(.{ .kernel = k, .check = .row_invariance, .site = if (site) |s| s.name else "", .words = words, .bad = bad, .ok = bad == 0 and words > 0 });
}

// ── decode_table: one-hot rows through the GEMV read W_hat exactly, every state covered ──

/// The golden plane of a code input: its projection (the input width) at its K (the tile width / 16).
fn goldenFor(h: *H, in_dim: u32, code_arg: *const xk.Arg) *const xk.GoldenPlane {
    return h.reg.golden.plane(in_dim, code_arg.shape[3].m / 16);
}

fn checkDecodeTable(h: *H, k: Kernel) !void {
    const e = h.reg.get(k);
    const g = goldenFor(h, e.inputs[inputIndex(e, "xh")].shape[1].m, &e.inputs[inputIndex(e, "code")]);
    var hp = try xk.HostPlane.init(h.a, g, h.table);
    defer hp.deinit(h.a);
    const rows: usize = g.onehot_rows;
    const in_dim: usize = g.in_dim;
    const out_dim: usize = g.out_dim;
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    var vars: Vars = .initFill(0);
    vars.set(.rows, rows);
    vars.set(.cap, 1);
    const xh = try h.a.alloc(f32, rows * in_dim);
    defer h.a.free(xh);
    @memset(xh, 0);
    for (0..rows) |r| xh[r * in_dim + r] = 1.0;
    const ids = try h.a.alloc(u32, rows);
    defer h.a.free(ids);
    @memset(ids, 0);
    const wave: Wave = .{};
    var ins: [inputs_max]mlx.mlx_array = @splat(.{});
    for (e.inputs, 0..) |*arg, i| {
        ins[i] = if (std.mem.eql(u8, arg.name, "xh"))
            try fromHost(&sc, std.mem.sliceAsBytes(xh), &.{ @intCast(rows), @intCast(in_dim) }, .float32)
        else if (std.mem.eql(u8, arg.name, "ids"))
            try fromHost(&sc, std.mem.sliceAsBytes(ids), &.{@intCast(rows)}, .uint32)
        else if (std.mem.eql(u8, arg.name, "code"))
            try fromHost(&sc, std.mem.sliceAsBytes(hp.code), &.{ 1, @intCast(in_dim / 16), @intCast(out_dim / 16), @intCast(16 * g.k) }, .int16)
        else
            try genInput(h, &sc, arg, &vars, &wave);
    }
    const outs = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, null);
    const got = try hostCopy(h, outs[0]);
    defer h.a.free(got);
    const n = rows * out_dim;
    var bad: u64 = 0;
    for (0..n) |i| {
        const want: u32 = @bitCast(@as(f32, @as(f16, @bitCast(hp.w[i]))));
        bad += @intFromBool(std.mem.readInt(u32, got[i * 4 ..][0..4], .little) != want);
    }
    var seen = try std.DynamicBitSet.initEmpty(h.a, 65536);
    defer seen.deinit();
    for (hp.states[0..n]) |s| seen.set(s);
    const covered = seen.count();
    try h.record(.{ .kernel = k, .check = .decode_table, .words = n, .bad = bad, .metric = @floatFromInt(covered), .limit = 65536, .ok = bad == 0 and covered == 65536 });
}

// ── golden_tiles: the DIG-X loader's decode (B^T) == W_hat^T, every word ──

fn checkGoldenTiles(h: *H, k: Kernel) !void {
    const e = h.reg.get(k);
    const n_out: usize = e.outputs[0].shape[0].m;
    const kd: usize = e.outputs[0].shape[1].m;
    const g = goldenFor(h, @intCast(kd), &e.inputs[0]);
    var hp = try xk.HostPlane.init(h.a, g, h.table);
    defer hp.deinit(h.a);
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    const vars: Vars = .initFill(0);
    const code = try fromHost(&sc, std.mem.sliceAsBytes(hp.code), &.{ 1, @intCast(kd / 16), @intCast(n_out / 16), @intCast(16 * g.k) }, .int16);
    const outs = try launch(h, &sc, k, &.{code}, &vars, null);
    const got = try hostCopy(h, outs[0]);
    defer h.a.free(got);
    var bad: u64 = 0;
    for (0..n_out) |nn| {
        for (0..kd) |kk| bad += @intFromBool(std.mem.readInt(u16, got[(nn * kd + kk) * 2 ..][0..2], .little) != hp.w[kk * n_out + nn]);
    }
    try h.record(.{ .kernel = k, .check = .golden_tiles, .words = n_out * kd, .bad = bad, .ok = bad == 0 });
}

// ── mlx_chain: bitwise equal to the eager MLX chain the kernel replaces ──

fn checkChain(h: *H, k: Kernel) !void {
    if (k == .q3_prefill_fused_exl3x3_mul1lut_k3_bf16) return chainRebuild(h, k);
    const e = h.reg.get(k);
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    const s = h.s;
    var vars = defaultVars(e);
    const wave = Wave.even(vars.get(.experts), vars.get(.rows), vars.get(.cap));
    const ins = try genAll(h, &sc, e, &vars, null, &wave);
    const outs = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, null);
    const rows: c_int = @intCast(vars.get(.rows));
    const in_ = struct {
        e: *const Entry,
        ins: [inputs_max]mlx.mlx_array,
        fn at(self: @This(), name: []const u8) mlx.mlx_array {
            return self.ins[inputIndex(self.e, name)];
        }
    }{ .e = e, .ins = ins };
    var refs: [2]mlx.mlx_array = undefined;
    var n_ref: usize = 0;
    switch (k) {
        // xg / xu = t128(take(x, tok).f32 * take(rin, ids).f32)   (Exl3PackedOps._project)
        .q3_exl3_prep_in_rin => {
            const xs = try f32Of(&sc, try take0(&sc, in_.at("x"), in_.at("tok"), s), s);
            for ([_][]const u8{ "rg", "ru" }) |name| {
                const r = try f32Of(&sc, try take0(&sc, in_.at(name), in_.at("ids"), s), s);
                refs[n_ref] = try t128(&sc, try mul(&sc, xs, r, s), &.{ rows, 5120 }, s);
                n_ref += 1;
            }
        },
        // t128(h * take(rin_d, ids).f32)
        .q3_exl3_prep_din_rin => {
            const r = try f32Of(&sc, try take0(&sc, in_.at("rn"), in_.at("ids"), s), s);
            refs[0] = try t128(&sc, try mul(&sc, try f32Of(&sc, in_.at("hid"), s), r, s), &.{ rows, 2304 }, s);
            n_ref = 1;
        },
        // t128(z) * take(rout_d, ids).f32
        .q3_moeprep_dpost => {
            const r = try f32Of(&sc, try take0(&sc, in_.at("rd"), in_.at("ids"), s), s);
            refs[0] = try mul(&sc, try t128(&sc, in_.at("zd"), &.{ rows, 5120 }, s), r, s);
            n_ref = 1;
        },
        // f16(t128(take(act, ridx).f32 * rin[slots[rhs]]))   (RotOpsX.take2; the retune checks against the same chain)
        .q3_prefill_dig_rot_take2_5120, .dsv41_prefill_dig_take2v_5120 => {
            const xs = try reshape(&sc, try f32Of(&sc, try take0(&sc, in_.at("act"), in_.at("ridx"), s), s), &.{ rows, 1, 5120 }, s);
            for ([_][]const u8{ "rin_g", "rin_u" }) |name| {
                const r = try slotRows(&sc, in_.at(name), in_.at("slots"), in_.at("rhs"), rows, 5120, s);
                refs[n_ref] = try op(&sc, mlx.mlx_astype, .{ try t128(&sc, try mul(&sc, xs, r, s), &.{ rows, 1, 5120 }, s), mlx.mlx_dtype.float16, s });
                n_ref += 1;
            }
        },
        // f16(t128(h * rin_d[slots[rhs]]))   (RotOpsX.roundx)
        .q3_prefill_dig_rot_roundx_2304 => {
            const r = try slotRows(&sc, in_.at("rin"), in_.at("slots"), in_.at("rhs"), rows, 2304, s);
            refs[0] = try op(&sc, mlx.mlx_astype, .{ try t128(&sc, try mul(&sc, try f32Of(&sc, in_.at("act"), s), r, s), &.{ rows, 1, 2304 }, s), mlx.mlx_dtype.float16, s });
            n_ref = 1;
        },
        // t128(z) * rout[slots[rhs]]   (hfold RotOps.widen2 / widen1)
        .q3_prefill_dig_rot_widen2_2304 => {
            for ([_][2][]const u8{ .{ "act_g", "rout_g" }, .{ "act_u", "rout_u" } }) |pair| {
                const r = try slotRows(&sc, in_.at(pair[1]), in_.at("slots"), in_.at("rhs"), rows, 2304, s);
                refs[n_ref] = try mul(&sc, try t128(&sc, in_.at(pair[0]), &.{ rows, 1, 2304 }, s), r, s);
                n_ref += 1;
            }
        },
        .q3_prefill_dig_rot_widen1_5120, .dsv41_prefill_dig_rot_widen1_5120_obf16 => {
            const r = try slotRows(&sc, in_.at("rout"), in_.at("slots"), in_.at("rhs"), rows, 5120, s);
            refs[0] = try mul(&sc, try t128(&sc, in_.at("act"), &.{ rows, 1, 5120 }, s), r, s);
            // kv16-opt: the bf16-out instantiation stores the same f32 word rounded to bf16 (MLX's astype)
            if (k == .dsv41_prefill_dig_rot_widen1_5120_obf16) refs[0] = try op(&sc, mlx.mlx_astype, .{ refs[0], mlx.mlx_dtype.bfloat16, s });
            n_ref = 1;
        },
        else => return error.NoChainForKernel,
    }
    var words: u64 = 0;
    var bad: u64 = 0;
    for (refs[0..n_ref], 0..) |r, i| try expectWords(h, k, .mlx_chain, r, outs[i], &words, &bad);
    try h.record(.{ .kernel = k, .check = .mlx_chain, .words = words, .bad = bad, .ok = bad == 0 and words > 0 });
}

/// The v2 eager chain of the rebuild (q3_prefill_exl3_rebuild_candidate.eager_weight_v2):
/// W = W_hat.f32 -> t128(W^T)^T -> t128(W) -> * rout[None, :] -> * rin[:, None] -> bf16,
/// W_hat from the host reconstruct, on a 4-slot bank, experts at slots (3, 0, 2).
fn chainRebuild(h: *H, k: Kernel) !void {
    const e = h.reg.get(k);
    const s = h.s;
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    var vars: Vars = .initFill(0);
    vars.set(.cap, 4);
    vars.set(.experts, 3);
    var wave: Wave = .{ .n = 3 };
    wave.slots[0] = 3;
    wave.slots[1] = 0;
    wave.slots[2] = 2;
    const ins = try genAll(h, &sc, e, &vars, null, &wave);
    const outs = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, null);
    const names = [3][3][]const u8{ .{ "code_g", "rout_g", "rin_g" }, .{ "code_u", "rout_u", "rin_u" }, .{ "code_d", "rout_d", "rin_d" } };
    var words: u64 = 0;
    var bad: u64 = 0;
    for (names, 0..) |nm, p| {
        const code_arg = &e.inputs[inputIndex(e, nm[0])];
        const n_i: usize = code_arg.shape[1].m;
        const n_j: usize = code_arg.shape[2].m;
        const in_dim: c_int = @intCast(n_i * 16);
        const out_dim: c_int = @intCast(n_j * 16);
        const k16: usize = code_arg.shape[3].m;
        const plane_words = n_i * n_j * k16;
        const code_host = try hostCopy(h, ins[inputIndex(e, nm[0])]);
        defer h.a.free(code_host);
        const got_all = try hostCopy(h, outs[p]);
        defer h.a.free(got_all);
        const plane = try h.a.alloc(i16, plane_words);
        defer h.a.free(plane);
        const w = try h.a.alloc(u16, n_i * n_j * 256);
        defer h.a.free(w);
        for (0..wave.n) |j| {
            var sj: Scope = .{ .a = h.a };
            defer sj.deinit();
            const slot: usize = wave.slots[j];
            @memcpy(std.mem.sliceAsBytes(plane), code_host[slot * plane_words * 2 ..][0 .. plane_words * 2]);
            xk.reconstruct(plane, n_i, n_j, k16 / 16, h.table, w, null);
            var wv = try f32Of(&sj, try fromHost(&sj, std.mem.sliceAsBytes(w), &.{ in_dim, out_dim }, .float16), s);
            const wt = try op(&sj, mlx.mlx_transpose, .{ wv, s });
            wv = try op(&sj, mlx.mlx_transpose, .{ try t128(&sj, wt, &.{ out_dim, in_dim }, s), s });
            wv = try t128(&sj, wv, &.{ in_dim, out_dim }, s);
            const one = [2]c_int{ @intCast(slot), 0 };
            const rout = try op(&sj, mlx.mlx_slice, .{ ins[inputIndex(e, nm[1])], &one, @as(usize, 2), &[2]c_int{ @intCast(slot + 1), out_dim }, @as(usize, 2), &[2]c_int{ 1, 1 }, @as(usize, 2), s });
            wv = try mul(&sj, wv, try f32Of(&sj, try reshape(&sj, rout, &.{ 1, out_dim }, s), s), s);
            const rin = try op(&sj, mlx.mlx_slice, .{ ins[inputIndex(e, nm[2])], &one, @as(usize, 2), &[2]c_int{ @intCast(slot + 1), in_dim }, @as(usize, 2), &[2]c_int{ 1, 1 }, @as(usize, 2), s });
            wv = try mul(&sj, wv, try f32Of(&sj, try reshape(&sj, rin, &.{ in_dim, 1 }, s), s), s);
            const ref = try hostCopy(h, try op(&sj, mlx.mlx_astype, .{ wv, mlx.mlx_dtype.bfloat16, s }));
            defer h.a.free(ref);
            const block = ref.len;
            words += block / 2;
            bad += countDiff(ref, got_all[j * block ..][0..block], 2);
        }
    }
    try h.record(.{ .kernel = k, .check = .mlx_chain, .words = words, .bad = bad, .ok = bad == 0 and words > 0 });
}

// ── f64: parity with a float64 host reference (the rounding-class kernels) ──

/// Natural-order (Sylvester) Walsh-Hadamard over one 128-block, scaled by 128^-0.5.
fn fwht128(v: []f64) void {
    var hh: usize = 1;
    while (hh < 128) : (hh *= 2) {
        var i: usize = 0;
        while (i < 128) : (i += 2 * hh) {
            for (i..i + hh) |j| {
                const x = v[j];
                const y = v[j + hh];
                v[j] = x + y;
                v[j + hh] = x - y;
            }
        }
    }
    for (v[0..128]) |*x| x.* /= @sqrt(128.0);
}

fn fwhtRow(v: []f64) void {
    var i: usize = 0;
    while (i < v.len) : (i += 128) fwht128(v[i..][0..128]);
}

fn silu(x: f64) f64 {
    return x / (1.0 + @exp(-x));
}

fn clampedSwiglu(g: f64, u: f64) f64 {
    return silu(@min(g, 10.0)) * std.math.clamp(u, -10.0, 10.0);
}

fn softplus(x: f64) f64 {
    return @max(x, 0.0) + std.math.log1p(@exp(-@abs(x)));
}

/// max |got - ref| / rms(ref) per row, the worst row.
fn maxErrOverRowRms(got: []const f64, ref: []const f64, row: usize) f64 {
    var worst: f64 = 0;
    var r: usize = 0;
    while (r < ref.len) : (r += row) {
        var ss: f64 = 0;
        var mx: f64 = 0;
        for (ref[r..][0..row], got[r..][0..row]) |a, b| {
            ss += a * a;
            mx = @max(mx, @abs(a - b));
        }
        worst = @max(worst, mx / (@sqrt(ss / @as(f64, @floatFromInt(row))) + 1e-30));
    }
    return worst;
}

fn maxRel(got: []const f64, ref: []const f64) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (got, ref) |a, b| {
        num = @max(num, @abs(a - b));
        den = @max(den, @abs(b));
    }
    return num / @max(den, 1e-30);
}

fn roundF16(x: f64) f64 {
    return @as(f16, @floatCast(x));
}

fn checkF64(h: *H, k: Kernel) !void {
    if (isDigGemm(k)) return digF64(h, k);
    const e = h.reg.get(k);
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    var vars = defaultVars(e);
    const wave = Wave.even(vars.get(.experts), vars.get(.rows), vars.get(.cap));
    const rows: usize = @intCast(vars.get(.rows));
    switch (k) {
        .q3_exl3_prep_gu_epi => {
            const ins = try genAll(h, &sc, e, &vars, null, &wave);
            const outs = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, null);
            const zg = try hostF64(h, ins[0]);
            defer h.a.free(zg);
            const zu = try hostF64(h, ins[1]);
            defer h.a.free(zu);
            const rg = try hostF64(h, ins[2]);
            defer h.a.free(rg);
            const ru = try hostF64(h, ins[3]);
            defer h.a.free(ru);
            const ids = try hostF64(h, ins[4]);
            defer h.a.free(ids);
            const got = try hostF64(h, outs[0]);
            defer h.a.free(got);
            const ref = try h.a.alloc(f64, got.len);
            defer h.a.free(ref);
            fwhtRow(zg);
            fwhtRow(zu);
            for (0..rows) |r| {
                const slot: usize = @intFromFloat(ids[r]);
                for (0..2304) |c| ref[r * 2304 + c] = clampedSwiglu(zg[r * 2304 + c] * rg[slot * 2304 + c], zu[r * 2304 + c] * ru[slot * 2304 + c]);
            }
            try recordTol(h, k, got.len, maxErrOverRowRms(got, ref, 2304), 1e-4);
        },
        .q3_prefill_dig2_swiglu_2304_x => {
            const ins = try genAll(h, &sc, e, &vars, null, &wave);
            const outs = try launch(h, &sc, k, ins[0..e.inputs.len], &vars, null);
            var v: [7][]f64 = undefined;
            for (0..7) |i| v[i] = try hostF64(h, ins[i]);
            defer for (v) |x| h.a.free(x);
            const got = try hostF64(h, outs[0]);
            defer h.a.free(got);
            const ref = try h.a.alloc(f64, got.len);
            defer h.a.free(ref);
            // inputs: z0, z1, rhs, tbl, rout0, rout1, rin2
            fwhtRow(v[0]);
            fwhtRow(v[1]);
            for (0..rows) |r| {
                const slot: usize = @intFromFloat(v[3][@intFromFloat(v[2][r])]);
                const hr = ref[r * 2304 ..][0..2304];
                for (hr, 0..) |*o, c| o.* = clampedSwiglu(v[0][r * 2304 + c] * v[4][slot * 2304 + c], v[1][r * 2304 + c] * v[5][slot * 2304 + c]) * v[6][slot * 2304 + c];
                fwhtRow(hr);
                for (hr) |*o| o.* = roundF16(o.*);
            }
            try recordTol(h, k, got.len, maxErrOverRowRms(got, ref, 2304), 1.0 / 256.0);
        },
        .q3rc_router_tail => try routerF64(h, &sc, .q3rc_gate_part, .q3rc_router_tail),
        .q3rc_router_tail__n128_top3 => try routerF64(h, &sc, .q3rc_gate_part__n128, .q3rc_router_tail__n128_top3),
        .q3ht_combine__f32, .q3ht_collapse_norm__f32, .q3ht_combine_collapse_norm__f32, .q3ht_combine_collapse_norm__f32_rbf16 => try hctapeF32(h, &sc, k),
        .q3rc_premix_fin => try premixF64(h, &sc),
        .q3dk_sinkhorn16_hc4_it20, .mtplx_dsv4_sinkhorn_hc4_it20 => try sinkhornF64(h, &sc, k),
        .mtplx_dsv41_fp_rmsnorm_tg128_d1280, .mtplx_dsv41_fp_rmsnorm_rope_tg128_d512_rd64, .mtplx_dsv41_fp_rope_h64_hd512_rd64_fwd, .mtplx_dsv41_fp_rope_h64_hd512_rd64_inv => try k36F64(h, &sc, k),
        .q3ht_combine, .q3ht_collapse_norm, .q3ht_combine_collapse_norm, .q3ht_mixfin => try hctapeF64(h, &sc, k),
        else => return error.NoF64ForKernel,
    }
}

fn recordTol(h: *H, k: Kernel, words: usize, metric: f64, limit: f64) !void {
    try h.record(.{ .kernel = k, .check = .f64, .words = words, .metric = metric, .limit = limit, .ok = metric <= limit and words > 0 });
}

fn templateInt(e: *const Entry, name: []const u8) usize {
    for (e.template) |x| if (std.mem.eql(u8, x.name, name)) return @intCast(x.value.int);
    unreachable;
}

/// gate part then router tail (the verify router, or the draft's N 128 / top-3 variants) vs
/// float64: indices equal (rows whose top-k-th / next biased scores are within 1e-5 skipped),
/// weights within 1e-5 relative.
fn routerF64(h: *H, sc: *Scope, part_k: Kernel, tail_k: Kernel) !void {
    const pe = h.reg.get(part_k);
    const te = h.reg.get(tail_k);
    var vars = defaultVars(pe);
    const wave: Wave = .{};
    const pins = try genAll(h, sc, pe, &vars, null, &wave);
    const part = try launch(h, sc, part_k, pins[0..pe.inputs.len], &vars, null);
    const bias = try genInput(h, sc, &te.inputs[1], &vars, &wave);
    const outs = try launch(h, sc, tail_k, &.{ part[0], bias }, &vars, null);
    const x = try hostF64(h, pins[0]);
    defer h.a.free(x);
    const w = try hostF64(h, pins[1]);
    defer h.a.free(w);
    const b = try hostF64(h, bias);
    defer h.a.free(b);
    const wt = try hostF64(h, outs[0]);
    defer h.a.free(wt);
    const ix = try hostF64(h, outs[1]);
    defer h.a.free(ix);
    const max_exp = 384;
    const n_exp = templateInt(te, "N");
    const topk = templateInt(te, "TOPK");
    std.debug.assert(n_exp <= max_exp and topk < n_exp);
    const kdim = 5120;
    const rows: usize = @intCast(vars.get(.rows));
    var bad: u64 = 0;
    var worst: f64 = 0;
    for (0..rows) |r| {
        var score: [max_exp]f64 = undefined;
        var biased: [max_exp]f64 = undefined;
        for (0..n_exp) |n| {
            var acc: f64 = 0;
            for (0..kdim) |c| acc += x[r * kdim + c] * w[n * kdim + c];
            score[n] = @sqrt(softplus(acc));
            biased[n] = score[n] + b[n];
        }
        var order_buf: [max_exp]u16 = undefined;
        const order = order_buf[0..n_exp];
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u16, order, @as(*const [max_exp]f64, &biased), struct {
            fn lt(bs: *const [max_exp]f64, a: u16, c: u16) bool {
                return bs[a] > bs[c] or (bs[a] == bs[c] and a < c);
            }
        }.lt);
        if (biased[order[topk - 1]] - biased[order[topk]] < 1e-5) continue;
        var sum: f64 = 0;
        for (order[0..topk]) |n| sum += score[n];
        for (0..topk) |t| {
            if (@as(usize, @intFromFloat(ix[r * topk + t])) != order[t]) bad += 1;
            const want = score[order[t]] / (sum + 1e-20) * 1.5;
            worst = @max(worst, @abs(wt[r * topk + t] - want) / @abs(want));
        }
    }
    try h.record(.{ .kernel = tail_k, .check = .f64, .words = rows * topk, .bad = bad, .metric = worst, .limit = 1e-5, .ok = bad == 0 and worst < 1e-5 });
}

/// q3rc_premix_part then q3rc_premix_fin vs float64: |out - x w^T| / (|x| |w|^T) < 1e-5.
fn premixF64(h: *H, sc: *Scope) !void {
    const pe = h.reg.get(.q3rc_premix_part);
    var vars = defaultVars(pe);
    const wave: Wave = .{};
    const pins = try genAll(h, sc, pe, &vars, null, &wave);
    const part = try launch(h, sc, .q3rc_premix_part, pins[0..pe.inputs.len], &vars, null);
    const outs = try launch(h, sc, .q3rc_premix_fin, &.{part[0]}, &vars, null);
    const x = try hostF64(h, pins[0]);
    defer h.a.free(x);
    const w = try hostF64(h, pins[1]);
    defer h.a.free(w);
    const got = try hostF64(h, outs[0]);
    defer h.a.free(got);
    const n_out = 24;
    const kdim = 20480;
    const rows: usize = @intCast(vars.get(.rows));
    var worst: f64 = 0;
    for (0..rows) |r| {
        for (0..n_out) |n| {
            var acc: f64 = 0;
            var mag: f64 = 0;
            for (0..kdim) |c| {
                acc += x[r * kdim + c] * w[n * kdim + c];
                mag += @abs(x[r * kdim + c] * w[n * kdim + c]);
            }
            worst = @max(worst, @abs(got[r * n_out + n] - acc) / @max(mag, 1e-30));
        }
    }
    try recordTol(h, .q3rc_premix_fin, rows * n_out, worst, 1e-5);
}

/// The 16-lane Sinkhorn vs its algorithm in float64 (hc 4, 20 iterations, eps 1e-6).
fn sinkhornF64(h: *H, sc: *Scope, k: Kernel) !void {
    const e = h.reg.get(k);
    var vars = defaultVars(e);
    const wave: Wave = .{};
    const ins = try genAll(h, sc, e, &vars, null, &wave);
    const outs = try launch(h, sc, k, ins[0..e.inputs.len], &vars, null);
    const comb = try hostF64(h, ins[0]);
    defer h.a.free(comb);
    const got = try hostF64(h, outs[0]);
    defer h.a.free(got);
    const eps = 1e-6;
    var worst: f64 = 0;
    var mat: usize = 0;
    while (mat * 16 < comb.len) : (mat += 1) {
        var c: [4][4]f64 = undefined;
        for (0..4) |i| {
            var m: f64 = -std.math.inf(f64);
            for (0..4) |j| m = @max(m, comb[mat * 16 + i * 4 + j]);
            var s: f64 = 0;
            for (0..4) |j| {
                c[i][j] = @exp(comb[mat * 16 + i * 4 + j] - m);
                s += c[i][j];
            }
            for (0..4) |j| c[i][j] = c[i][j] / s + eps;
        }
        sinkCols(&c, eps);
        for (0..19) |_| {
            for (0..4) |i| {
                var s: f64 = 0;
                for (0..4) |j| s += c[i][j];
                for (0..4) |j| c[i][j] /= s + eps;
            }
            sinkCols(&c, eps);
        }
        for (0..4) |i| {
            for (0..4) |j| worst = @max(worst, @abs(got[mat * 16 + i * 4 + j] - c[i][j]));
        }
    }
    try recordTol(h, k, comb.len, worst, 1e-5);
}

fn sinkCols(c: *[4][4]f64, eps: f64) void {
    for (0..4) |j| {
        var s: f64 = 0;
        for (0..4) |i| s += c[i][j];
        for (0..4) |i| c[i][j] /= s + eps;
    }
}

fn bf16Value(w: u16) f64 {
    return @as(f32, @bitCast(@as(u32, w) << 16));
}

/// The operands of one q3ht_combine output (row m, stream k, column c): x, r_j, post_k, comb[j, k].
const HcTerms = struct { x: f32, r: [4]f32, post: f32, comb: [4]f32 };

/// The stored word of q3ht_combine's written f32 chain: fma(c0, r0, -0), fma(cj, rj, .) for
/// j = 1..3, fma(post, x, .), then the RNE bf16 store.
fn hcChainWord(t: HcTerms) u16 {
    var mk: f32 = @mulAdd(f32, t.comb[0], t.r[0], -0.0);
    for (1..4) |j| mk = @mulAdd(f32, t.comb[j], t.r[j], mk);
    return bf16Bits(@mulAdd(f32, t.post, t.x, mk));
}

/// The output in float64 (post x + sum_j comb_j r_j, in that order) and the sum of |terms|.
const HcExact = struct { value: f64, mag: f64 };

fn hcExact(t: HcTerms) HcExact {
    var acc = @as(f64, t.post) * t.x;
    var mag = @abs(acc);
    for (0..4) |j| {
        const v = @as(f64, t.comb[j]) * t.r[j];
        acc += v;
        mag += @abs(v);
    }
    return .{ .value = acc, .mag = mag };
}

/// |chain result - value| <= 6u x sum |terms|: five f32 roundings, each <= u x a partial sum
/// (<= (1 + u)^5 sum |terms|), plus the f64 reference's own.
const hc_chain_slack = 6.0 / 16777216.0;

/// A stored combine word is the RNE bf16 store of some f32 chain result: within half a bf16 ulp
/// (at the word) plus the chain's slack of the f64 value. NaN / inf never pass.
fn hcWordWithinChain(word: u16, e: HcExact) bool {
    return bf16Ratio(word, e.value, hc_chain_slack * e.mag) <= 1.0;
}

/// |word - value| over (half a bf16 ulp at the word + `slack`): <= 1 iff the word is the RNE
/// bf16 store of a value within `slack` of `value`. NaN / inf give NaN / inf (never <= 1).
fn bf16Ratio(word: u16, value: f64, slack: f64) f64 {
    const eb: u64 = (word >> 7) & 0xFF;
    const half_ulp: f64 = @bitCast((@max(eb, 1) + 888) << 52);
    return @abs(bf16Value(word) - value) / (half_ulp + slack);
}

/// HCTAPE vs q3_decode_hctape_candidate._f64_ref and its split statements; the combine words
/// also bitwise against the written chain (h_twin), as the lane's probe checks the device.
fn hctapeF64(h: *H, sc: *Scope, k: Kernel) !void {
    const e = h.reg.get(k);
    var vars = defaultVars(e);
    const wave: Wave = .{};
    const ins = try genAll(h, sc, e, &vars, null, &wave);
    const outs = try launch(h, sc, k, ins[0..e.inputs.len], &vars, null);
    const rows: usize = @intCast(vars.get(.rows));
    const d = 5120;
    const hc = 4;
    var in_host: [6][]f64 = undefined;
    var n_in: usize = 0;
    defer for (in_host[0..n_in]) |x| h.a.free(x);
    for (0..e.inputs.len) |i| {
        in_host[i] = try hostF64(h, ins[i]);
        n_in += 1;
    }
    switch (k) {
        .q3ht_mixfin => {
            const mm = in_host[0];
            const ssq = in_host[1];
            const scale = in_host[2];
            const base = in_host[3];
            var worst: f64 = 0;
            for (0..3) |o| {
                const got = try hostF64(h, outs[o]);
                defer h.a.free(got);
                const width: usize = if (o == 2) hc * hc else hc;
                const off: usize = if (o == 0) 0 else if (o == 1) hc else 2 * hc;
                const ref = try h.a.alloc(f64, got.len);
                defer h.a.free(ref);
                for (0..rows) |r| {
                    const rs = 1.0 / @sqrt(ssq[r] / (hc * d) + 1e-20);
                    for (0..width) |c| {
                        const z = mm[r * 24 + off + c] * rs * scale[o] + base[off + c];
                        ref[r * width + c] = switch (o) {
                            0 => 1.0 / (1.0 + @exp(-z)) + 1e-6,
                            1 => 2.0 / (1.0 + @exp(-z)),
                            else => z,
                        };
                    }
                }
                worst = @max(worst, maxRel(got, ref));
            }
            try recordTol(h, k, rows * 24, worst, 1e-5);
        },
        .q3ht_collapse_norm => {
            const s = in_host[0];
            const got = try hostF64(h, outs[1]);
            defer h.a.free(got);
            const ref = try h.a.alloc(f64, rows);
            defer h.a.free(ref);
            for (0..rows) |r| {
                var acc: f64 = 0;
                for (s[r * hc * d ..][0 .. hc * d]) |v| acc += v * v;
                ref[r] = acc;
            }
            try recordTol(h, k, rows, maxRel(got, ref), 1e-5);
        },
        .q3ht_combine, .q3ht_combine_collapse_norm => {
            const x = in_host[0];
            const r_ = in_host[1];
            const post = in_host[2];
            const comb = in_host[3];
            const hb = try h.a.alloc(f64, rows * hc * d);
            defer h.a.free(hb);
            const got_h = try hostCopy(h, outs[0]);
            defer h.a.free(got_h);
            const fused = k == .q3ht_combine_collapse_norm;
            const got_hf = if (fused) try hostCopy(h, outs[1]) else &[_]u8{};
            defer if (fused) h.a.free(got_hf);
            var twin_bad: u64 = 0;
            var beyond: u64 = 0;
            var differing: u64 = 0;
            for (0..rows) |m| {
                for (0..hc) |kk| {
                    for (0..d) |c| {
                        var t: HcTerms = .{ .x = @floatCast(x[m * d + c]), .r = undefined, .post = @floatCast(post[m * hc + kk]), .comb = undefined };
                        for (0..hc) |j| {
                            t.r[j] = @floatCast(r_[(m * hc + j) * d + c]);
                            t.comb[j] = @floatCast(comb[m * 16 + j * hc + kk]);
                        }
                        const at = (m * hc + kk) * d + c;
                        const have = std.mem.readInt(u16, got_h[at * 2 ..][0..2], .little);
                        const twin = hcChainWord(t);
                        twin_bad += @intFromBool(have != twin);
                        if (fused) twin_bad += @intFromBool(std.mem.readInt(u32, got_hf[at * 4 ..][0..4], .little) != @as(u32, twin) << 16);
                        const ex = hcExact(t);
                        const want = bf16Bits(@floatCast(ex.value));
                        hb[at] = bf16Value(want);
                        differing += @intFromBool(have != want);
                        beyond += @intFromBool(!hcWordWithinChain(have, ex));
                    }
                }
            }
            const n = rows * hc * d;
            const frac = @as(f64, @floatFromInt(differing)) / @as(f64, @floatFromInt(n));
            try h.record(.{ .kernel = k, .check = .f64, .site = "h_twin", .words = if (fused) 2 * n else n, .bad = twin_bad, .ok = twin_bad == 0 });
            try h.record(.{ .kernel = k, .check = .f64, .site = "h", .words = n, .bad = beyond, .metric = frac, .limit = 2e-3, .ok = beyond == 0 and frac <= 2e-3 });
            if (k == .q3ht_combine_collapse_norm) {
                const pre = in_host[4];
                const w = in_host[5];
                const ssq_got = try hostF64(h, outs[2]);
                defer h.a.free(ssq_got);
                const y_got = try hostF64(h, outs[3]);
                defer h.a.free(y_got);
                const ssq_ref = try h.a.alloc(f64, rows);
                defer h.a.free(ssq_ref);
                const y_ref = try h.a.alloc(f64, rows * d);
                defer h.a.free(y_ref);
                const cbuf = try h.a.alloc(f64, d);
                defer h.a.free(cbuf);
                for (0..rows) |m| {
                    var acc: f64 = 0;
                    for (hb[m * hc * d ..][0 .. hc * d]) |v| acc += v * v;
                    ssq_ref[m] = acc;
                    var var_: f64 = 0;
                    for (0..d) |c| {
                        var col: f64 = 0;
                        for (0..hc) |kk| col += pre[m * hc + kk] * hb[(m * hc + kk) * d + c];
                        cbuf[c] = @as(f32, @bitCast(@as(u32, bf16Bits(@floatCast(col))) << 16));
                        var_ += cbuf[c] * cbuf[c];
                    }
                    const inv = 1.0 / @sqrt(var_ / d + 1e-20);
                    for (0..d) |c| y_ref[m * d + c] = w[c] * cbuf[c] * inv;
                }
                try h.record(.{ .kernel = k, .check = .f64, .site = "ssq", .words = rows, .metric = maxRel(ssq_got, ssq_ref), .limit = 1e-5, .ok = maxRel(ssq_got, ssq_ref) < 1e-5 });
                var num: f64 = 0;
                var den: f64 = 0;
                for (y_got, y_ref) |a, b| {
                    num += (a - b) * (a - b);
                    den += b * b;
                }
                const rel = @sqrt(num / @max(den, 1e-30));
                try h.record(.{ .kernel = k, .check = .f64, .site = "y", .words = rows * d, .metric = rel, .limit = 5e-3, .ok = rel < 5e-3 });
            }
        },
        else => return error.NoF64ForKernel,
    }
}

/// The draft's f32-stream HCTAPE variants vs float64: h (combine / fused) within 8 f32 ulps of
/// its terms' magnitude of post x + sum_j comb[j, k] r_j, the fused hf == h word for word (an
/// f32 stream has no rounding in between), ssq within 1e-5 relative, the fused y within 1e-5
/// rms relative of w * col / rms(col) (col = sum_k pre[k] h_k).
fn hctapeF32(h: *H, sc: *Scope, k: Kernel) !void {
    const e = h.reg.get(k);
    var vars = defaultVars(e);
    const wave: Wave = .{};
    const ins = try genAll(h, sc, e, &vars, null, &wave);
    const outs = try launch(h, sc, k, ins[0..e.inputs.len], &vars, null);
    const rows: usize = @intCast(vars.get(.rows));
    const d = 5120;
    const hc = 4;
    var in_host: [6][]f64 = undefined;
    var n_in: usize = 0;
    defer for (in_host[0..n_in]) |x| h.a.free(x);
    for (0..e.inputs.len) |i| {
        in_host[i] = try hostF64(h, ins[i]);
        n_in += 1;
    }
    if (k == .q3ht_collapse_norm__f32) {
        const s = in_host[0];
        const got = try hostF64(h, outs[1]);
        defer h.a.free(got);
        const ref = try h.a.alloc(f64, rows);
        defer h.a.free(ref);
        for (0..rows) |r| {
            var acc: f64 = 0;
            for (s[r * hc * d ..][0 .. hc * d]) |v| acc += v * v;
            ref[r] = acc;
        }
        return recordTol(h, k, rows, maxRel(got, ref), 1e-5);
    }
    const x, const r_, const post, const comb = .{ in_host[0], in_host[1], in_host[2], in_host[3] };
    const n = rows * hc * d;
    const href = try h.a.alloc(f64, n);
    defer h.a.free(href);
    const got_h = try hostF64(h, outs[0]);
    defer h.a.free(got_h);
    var beyond: u64 = 0;
    var worst: f64 = 0;
    for (0..rows) |m| for (0..hc) |kk| for (0..d) |c| {
        var v: f64 = post[m * hc + kk] * x[m * d + c];
        var mag: f64 = @abs(v);
        for (0..hc) |j| {
            const t = comb[m * 16 + j * hc + kk] * r_[(m * hc + j) * d + c];
            v += t;
            mag += @abs(t);
        }
        const at = (m * hc + kk) * d + c;
        href[at] = v;
        const err = @abs(got_h[at] - v);
        const lim = 0x1p-21 * mag;
        beyond += @intFromBool(err > lim);
        if (mag > 0) worst = @max(worst, err / mag);
    };
    try h.record(.{ .kernel = k, .check = .f64, .site = "h", .words = n, .bad = beyond, .metric = worst, .limit = 0x1p-21, .ok = beyond == 0 });
    if (k == .q3ht_combine__f32) return;
    const hw = try hostCopy(h, outs[0]);
    defer h.a.free(hw);
    const hfw = try hostCopy(h, outs[1]);
    defer h.a.free(hfw);
    try h.record(.{ .kernel = k, .check = .f64, .site = "hf", .words = n, .bad = countDiff(hw, hfw, 4), .ok = std.mem.eql(u8, hw, hfw) });
    const pre, const w = .{ in_host[4], in_host[5] };
    const ssq_got = try hostF64(h, outs[2]);
    defer h.a.free(ssq_got);
    const y_got = try hostF64(h, outs[3]);
    defer h.a.free(y_got);
    const ssq_ref = try h.a.alloc(f64, rows);
    defer h.a.free(ssq_ref);
    const y_ref = try h.a.alloc(f64, rows * d);
    defer h.a.free(y_ref);
    const cbuf = try h.a.alloc(f64, d);
    defer h.a.free(cbuf);
    for (0..rows) |m| {
        var acc: f64 = 0;
        for (href[m * hc * d ..][0 .. hc * d]) |v| acc += v * v;
        ssq_ref[m] = acc;
        var var_: f64 = 0;
        for (0..d) |c| {
            var col: f64 = 0;
            for (0..hc) |kk| col += pre[m * hc + kk] * href[(m * hc + kk) * d + c];
            cbuf[c] = col;
            var_ += col * col;
        }
        const inv = 1.0 / @sqrt(var_ / d + 1e-20);
        for (0..d) |c| y_ref[m * d + c] = w[c] * cbuf[c] * inv;
    }
    const ssq_rel = maxRel(ssq_got, ssq_ref);
    try h.record(.{ .kernel = k, .check = .f64, .site = "ssq", .words = rows, .metric = ssq_rel, .limit = 1e-5, .ok = ssq_rel < 1e-5 });
    var num: f64 = 0;
    var den: f64 = 0;
    for (y_got, y_ref) |a, b| {
        num += (a - b) * (a - b);
        den += b * b;
    }
    const rel = @sqrt(num / @max(den, 1e-30));
    try h.record(.{ .kernel = k, .check = .f64, .site = "y", .words = rows * d, .metric = rel, .limit = 1e-5, .ok = rel < 1e-5 });
}

/// |f32 statements - f64 value| bound for K36, relative to the terms' magnitude (a tree sum of
/// 1280 squares and an rsqrt, or two products and a sum: ~1e-6).
const k36_slack = 1.0 / 65536.0;

/// K36 vs float64 (the kernels' statements): every bf16 word within half a bf16 ulp (at the
/// word) + k36_slack x |its terms| of the f64 value.
fn k36F64(h: *H, sc: *Scope, k: Kernel) !void {
    const e = h.reg.get(k);
    var vars = defaultVars(e);
    const wave: Wave = .{};
    const ins = try genAll(h, sc, e, &vars, null, &wave);
    const outs = try launch(h, sc, k, ins[0..e.inputs.len], &vars, null);
    const rows: usize = @intCast(vars.get(.rows));
    const got = try hostCopy(h, outs[0]);
    defer h.a.free(got);
    var in_host: [5][]f64 = undefined;
    var n_in: usize = 0;
    defer for (in_host[0..n_in]) |x| h.a.free(x);
    for (e.inputs) |*arg| {
        if (arg.role == .scalar) break;
        in_host[n_in] = try hostF64(h, ins[n_in]);
        n_in += 1;
    }
    const rope_only = k == .mtplx_dsv41_fp_rope_h64_hd512_rd64_fwd or k == .mtplx_dsv41_fp_rope_h64_hd512_rd64_inv;
    const x = in_host[0];
    const width: usize = if (rope_only) 64 * 512 else e.inputs[0].shape[1].m;
    const seg: usize = if (rope_only) 512 else width;
    const rd = 64;
    const val = try h.a.alloc(f64, width);
    defer h.a.free(val);
    const mag = try h.a.alloc(f64, width);
    defer h.a.free(mag);
    var beyond: u64 = 0;
    var worst: f64 = 0;
    for (0..rows) |r| {
        const xr = x[r * width ..][0..width];
        if (rope_only) {
            @memcpy(val, xr);
        } else {
            const w = in_host[1];
            const eps = in_host[2][0];
            var ss: f64 = 0;
            for (xr) |v| ss += v * v;
            const inv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(width)) + eps);
            for (val, xr, w) |*o, v, wv| o.* = wv * (v * inv);
        }
        for (mag, val) |*m, v| m.* = @abs(v);
        const has_rope = k != .mtplx_dsv41_fp_rmsnorm_tg128_d1280;
        if (has_rope) {
            const cos = in_host[if (rope_only) 1 else 3][r * (rd / 2) ..][0 .. rd / 2];
            const sin = in_host[if (rope_only) 2 else 4][r * (rd / 2) ..][0 .. rd / 2];
            const sign: f64 = if (k == .mtplx_dsv41_fp_rope_h64_hd512_rd64_inv) -1 else 1;
            var h0: usize = 0;
            while (h0 < width) : (h0 += seg) {
                const t0 = h0 + seg - rd;
                for (0..rd / 2) |p| {
                    const v0 = val[t0 + 2 * p];
                    const v1 = val[t0 + 2 * p + 1];
                    const c = cos[p];
                    const sn = sign * sin[p];
                    val[t0 + 2 * p] = v0 * c - v1 * sn;
                    val[t0 + 2 * p + 1] = v0 * sn + v1 * c;
                    const m = @abs(v0 * c) + @abs(v1 * sn) + @abs(v0 * sn) + @abs(v1 * c);
                    mag[t0 + 2 * p] = m;
                    mag[t0 + 2 * p + 1] = m;
                }
            }
        }
        for (val, mag, 0..) |v, m, i| {
            const word = std.mem.readInt(u16, got[(r * width + i) * 2 ..][0..2], .little);
            const ratio = bf16Ratio(word, v, k36_slack * m);
            worst = @max(worst, ratio);
            beyond += @intFromBool(!(ratio <= 1.0));
        }
    }
    try h.record(.{ .kernel = k, .check = .f64, .words = rows * width, .bad = beyond, .metric = worst, .limit = 1.0, .ok = beyond == 0 });
}

// ── DIG-X GEMMs: wave composition and float64 parity ──

const dig_rows = [_]u32{ 70, 37, 20, 17 };

fn isGateUp(k: Kernel) bool {
    return k == .q3_prefill_dig_gemm_5120x2304_gu_xmul1hk3 or k == .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128 or k == .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk3_m128lut or k == .dsv41_prefill_dig_gemm_5120x2304_gu_xmul1hk2_m128;
}

/// One DIG GEMM launch over `wave` with the A rows `xs` (f16 [rows, 1, K], one per operand); `rout` for the fused
/// down GEMM's epilogue (its bank's rout, before the table).
fn digLaunch(h: *H, sc: *Scope, k: Kernel, xs: []const mlx.mlx_array, codes: []const mlx.mlx_array, rout: ?mlx.mlx_array, wave: *const Wave) ![xk.max_outputs]mlx.mlx_array {
    const e = h.reg.get(k);
    const tbl_arg = &e.inputs[e.inputs.len - 1];
    var t: [80]i32 = undefined;
    const tgs = wave.table(tbl_arg.domain.tiles, tbl_arg.domain.bm, &t);
    const tbl = try fromHost(sc, std.mem.sliceAsBytes(&t), &.{80}, .int32);
    var vars: Vars = .initFill(0);
    vars.set(.rows, wave.total());
    vars.set(.tgs, tgs);
    vars.set(.cap, 4);
    vars.set(.experts, wave.n);
    var ins: [inputs_max]mlx.mlx_array = @splat(.{});
    var n: usize = 0;
    for (xs) |x| {
        ins[n] = x;
        n += 1;
    }
    for (codes) |c| {
        ins[n] = c;
        n += 1;
    }
    if (rout) |r| {
        ins[n] = r;
        n += 1;
    }
    ins[n] = tbl;
    n += 1;
    return launch(h, sc, k, ins[0..n], &vars, null);
}

const DigFull = struct {
    xs: [2]mlx.mlx_array,
    codes: [2]mlx.mlx_array,
    rout: ?mlx.mlx_array = null,
    n_ops: usize,
    outs: [xk.max_outputs]mlx.mlx_array,
    wave: Wave,
    k_dim: c_int,
};

fn digFull(h: *H, sc: *Scope, k: Kernel) !DigFull {
    return digFullAt(h, sc, k, &dig_rows);
}

/// `digFull` over experts of `rows_of` rows (slots 0, 1, ...).
fn digFullAt(h: *H, sc: *Scope, k: Kernel, rows_of: []const u32) !DigFull {
    const e = h.reg.get(k);
    var wave: Wave = .{ .n = rows_of.len };
    for (rows_of, 0..) |r, j| {
        wave.slots[j] = @intCast(j);
        wave.rows[j] = r;
    }
    var vars: Vars = .initFill(0);
    vars.set(.rows, wave.total());
    vars.set(.cap, 4);
    vars.set(.experts, wave.n);
    const n_ops: usize = if (isGateUp(k)) 2 else 1;
    var f: DigFull = .{ .xs = @splat(.{}), .codes = @splat(.{}), .n_ops = n_ops, .outs = undefined, .wave = wave, .k_dim = @intCast(e.inputs[0].shape[2].m) };
    for (0..n_ops) |o| {
        f.xs[o] = try genInput(h, sc, &e.inputs[o], &vars, &wave);
        f.codes[o] = try genInput(h, sc, &e.inputs[n_ops + o], &vars, &wave);
    }
    if (fusedOf(k) != null) f.rout = try genInput(h, sc, &e.inputs[2 * n_ops], &vars, &wave);
    f.outs = try digLaunch(h, sc, k, f.xs[0..n_ops], f.codes[0..n_ops], f.rout, &wave);
    return f;
}

fn rowsOf(sc: *Scope, s: mlx.mlx_stream, x: mlx.mlx_array, row0: u32, n: u32, k_dim: c_int) !mlx.mlx_array {
    return op(sc, mlx.mlx_slice, .{ x, &[3]c_int{ @intCast(row0), 0, 0 }, @as(usize, 3), &[3]c_int{ @intCast(row0 + n), 1, k_dim }, @as(usize, 3), &[3]c_int{ 1, 1, 1 }, @as(usize, 3), s });
}

fn row0Of(w: *const Wave, j: usize) u32 {
    var r: u32 = 0;
    for (w.rows[0..j]) |x| r += x;
    return r;
}

/// wave == each expert alone == regroup (1, 3) == rerun, f32 words.
fn checkComposition(h: *H, k: Kernel) !void {
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    const f = try digFull(h, &sc, k);
    var full: [2][]u8 = undefined;
    for (0..f.n_ops) |o| full[o] = try hostCopy(h, f.outs[o]);
    defer for (full[0..f.n_ops]) |b| h.a.free(b);
    const n_cols: usize = h.reg.get(k).outputs[0].shape[1].m;
    const row_bytes = n_cols * 4;
    var words: u64 = 0;
    var bad: u64 = 0;
    const groups = [_][]const usize{ &.{0}, &.{1}, &.{2}, &.{3}, &.{ 1, 3 }, &.{ 0, 1, 2, 3 } };
    for (groups) |grp| {
        var sg: Scope = .{ .a = h.a };
        defer sg.deinit();
        var w: Wave = .{ .n = grp.len };
        var xs: [2]mlx.mlx_array = @splat(.{});
        for (0..f.n_ops) |o| {
            var parts: [4]mlx.mlx_array = undefined;
            for (grp, 0..) |j, i| parts[i] = try rowsOf(&sg, h.s, f.xs[o], row0Of(&f.wave, j), f.wave.rows[j], f.k_dim);
            const v = mlx.mlx_vector_array_new_data(&parts, grp.len);
            defer _ = mlx.mlx_vector_array_free(v);
            xs[o] = try op(&sg, mlx.mlx_concatenate_axis, .{ v, @as(c_int, 0), h.s });
        }
        for (grp, 0..) |j, i| {
            w.slots[i] = f.wave.slots[j];
            w.rows[i] = f.wave.rows[j];
        }
        const outs = try digLaunch(h, &sg, k, xs[0..f.n_ops], f.codes[0..f.n_ops], f.rout, &w);
        for (0..f.n_ops) |o| {
            const got = try hostCopy(h, outs[o]);
            defer h.a.free(got);
            var at: usize = 0;
            for (grp) |j| {
                const len = f.wave.rows[j] * row_bytes;
                const want = full[o][row0Of(&f.wave, j) * row_bytes ..][0..len];
                words += len / 4;
                bad += countDiff(want, got[at..][0..len], 4);
                at += len;
            }
        }
    }
    try h.record(.{ .kernel = k, .check = .composition, .words = words, .bad = bad, .ok = bad == 0 and words > 0 });
}

/// The 128-row texts' twin experts: full and partial 128-row tiles, a 64-row expert, a single row.
const twin_rows = [_]u32{ 270, 129, 64, 1 };

/// A DIG-X GEMM against its twin (`twinOf`) on the same x, codes and wave: every z word.
fn checkTwin(h: *H, k: Kernel) !void {
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    const f = try digFullAt(h, &sc, k, &twin_rows);
    const ref = try digLaunch(h, &sc, twinOf(k).?, f.xs[0..f.n_ops], f.codes[0..f.n_ops], null, &f.wave);
    var words: u64 = 0;
    var bad: u64 = 0;
    for (0..f.n_ops) |o| {
        const got = try hostCopy(h, f.outs[o]);
        defer h.a.free(got);
        const want = try hostCopy(h, ref[o]);
        defer h.a.free(want);
        words += got.len / 4;
        bad += if (got.len == want.len) countDiff(want, got, 4) else got.len / 4;
    }
    try h.record(.{ .kernel = k, .check = .twin, .words = words, .bad = bad, .ok = bad == 0 and words > 0 });
}

/// The fused down GEMM against its two-kernel chain on the same x, codes, rout and wave (experts of 270 / 129 / 64 / 1
/// rows): the 128-row down text's z, then rot_widen1 over it (each row's expert into the wave table's slots). Every
/// output word.
fn checkFused(h: *H, k: Kernel) !void {
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    const f = try digFullAt(h, &sc, k, &twin_rows);
    const z = try digLaunch(h, &sc, fusedOf(k).?, f.xs[0..1], f.codes[0..1], null, &f.wave);
    const total: usize = @intCast(f.wave.total());
    const rhs = try h.a.alloc(u32, total);
    defer h.a.free(rhs);
    var r: usize = 0;
    for (f.wave.rows[0..f.wave.n], 0..) |rows, j| for (0..rows) |_| {
        rhs[r] = @intCast(j);
        r += 1;
    };
    var t: [80]i32 = undefined;
    _ = f.wave.table(h.reg.get(k).inputs[3].domain.tiles, 128, &t);
    var vars: Vars = .initFill(0);
    vars.set(.rows, total);
    vars.set(.a_rows, total);
    vars.set(.cap, 4);
    vars.set(.experts, f.wave.n);
    const slots = try fromHost(&sc, std.mem.sliceAsBytes(&t), &.{80}, .int32);
    const rhs_a = try fromHost(&sc, std.mem.sliceAsBytes(rhs), &.{@intCast(total)}, .uint32);
    const ref = try launch(h, &sc, .q3_prefill_dig_rot_widen1_5120, &.{ z[0], rhs_a, slots, f.rout.? }, &vars, null);
    const got = try hostCopy(h, f.outs[0]);
    defer h.a.free(got);
    const want = try hostCopy(h, ref[0]);
    defer h.a.free(want);
    const words = got.len / 4;
    const bad = if (got.len == want.len) countDiff(want, got, 4) else words;
    try h.record(.{ .kernel = k, .check = .fused, .words = words, .bad = bad, .ok = bad == 0 and words > 0 });
}

/// f16(z) of the first and last rows of experts 0 and 3 vs x (f16) @ W_hat (exl3_ref decode)
/// in float64, rounded to f16: max error <= 2^-8 x row rms.
fn digF64(h: *H, k: Kernel) !void {
    var sc: Scope = .{ .a = h.a };
    defer sc.deinit();
    const e = h.reg.get(k);
    const f = try digFull(h, &sc, k);
    const code_arg = &e.inputs[f.n_ops];
    const n_i: usize = code_arg.shape[1].m;
    const n_j: usize = code_arg.shape[2].m;
    const kd = n_i * 16;
    const nd = n_j * 16;
    const k16: usize = code_arg.shape[3].m;
    const plane_words = n_i * n_j * k16;
    const plane = try h.a.alloc(i16, plane_words);
    defer h.a.free(plane);
    const w = try h.a.alloc(u16, kd * nd);
    defer h.a.free(w);
    const ref = try h.a.alloc(f64, nd);
    defer h.a.free(ref);
    const got16 = try h.a.alloc(f64, nd);
    defer h.a.free(got16);
    var worst: f64 = 0;
    var words: u64 = 0;
    for (0..f.n_ops) |o| {
        const x = try hostF64(h, f.xs[o]);
        defer h.a.free(x);
        const code = try hostCopy(h, f.codes[o]);
        defer h.a.free(code);
        const z = try hostF64(h, f.outs[o]);
        defer h.a.free(z);
        for ([_]usize{ 0, 3 }) |j| {
            const slot: usize = f.wave.slots[j];
            @memcpy(std.mem.sliceAsBytes(plane), code[slot * plane_words * 2 ..][0 .. plane_words * 2]);
            xk.reconstruct(plane, n_i, n_j, k16 / 16, h.table, w, null);
            const r0 = row0Of(&f.wave, j);
            for ([_]u32{ r0, r0 + f.wave.rows[j] - 1 }) |row| {
                @memset(ref, 0);
                for (0..kd) |kk| {
                    const xv = x[row * kd + kk];
                    for (ref, w[kk * nd ..][0..nd]) |*acc, wb| acc.* += xv * @as(f64, @as(f16, @bitCast(wb)));
                }
                for (ref, got16, z[row * nd ..][0..nd]) |*rv, *gv, zv| {
                    rv.* = roundF16(rv.*);
                    gv.* = roundF16(zv);
                }
                worst = @max(worst, maxErrOverRowRms(got16, ref, nd));
                words += nd;
            }
        }
    }
    try recordTol(h, k, words, worst, 1.0 / 256.0);
}

// ── Tests ──

const testing = std.testing;

test "dsv41 kernels: every self-check the manifest plans has an executor" {
    var diag: xk.Diag = .{};
    var reg = xk.Registry.init(testing.allocator, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
        std.debug.print("exl3 kernels refused: {s}\n", .{diag.message()});
        return e;
    };
    defer reg.deinit();
    var planned: usize = 0;
    for (&reg.entries) |*e| {
        var it = e.checks.iterator();
        while (it.next()) |c| {
            if (!implemented(e.kernel, c)) std.debug.print("no executor for {t} {t}\n", .{ e.kernel, c });
            try testing.expect(implemented(e.kernel, c));
            planned += 1;
        }
    }
    try testing.expect(planned >= 2 * xk.n_kernels);
}

test "dsv41 kernels: a raised self-check's receipt line carries the MLX message, as valid JSON" {
    const a = testing.allocator;
    var report: Report = .{};
    defer report.deinit(a);
    try report.results.append(a, .{ .kernel = .q3pf_hc_pre_norm, .check = .compile, .ok = true });
    const msg = "[metal::Device] Unable to build metal library from source\nutils.h:520:30: error: no matching function for call to 'q3pf_ld'\n\t\"x\" \\ \x01";
    var buf: [256]u8 = undefined;
    @memcpy(buf[0..msg.len], msg);
    try report.appendRaised(a, .q3pf_hc_pre_norm, .row_invariance, error.MlxError, buf[0..msg.len]);
    @memset(&buf, 0); // the latch buffer is reused: the report keeps its own copy
    const lines = try report.writeJsonLines(a);
    defer a.free(lines);
    var it = std.mem.splitScalar(u8, lines, '\n');
    // a passing line is the receipt format of record, byte for byte
    try testing.expectEqualStrings("{\"kernel\":\"q3pf_hc_pre_norm\",\"check\":\"compile\",\"site\":\"\",\"words\":0,\"bad\":0,\"metric\":0e0,\"limit\":0e0,\"ok\":true,\"err\":\"\"}", it.next().?);
    const Line = struct { kernel: []const u8, check: []const u8, site: []const u8, words: u64, bad: u64, metric: f64, limit: f64, ok: bool, err: []const u8, msg: []const u8 };
    const parsed = try std.json.parseFromSlice(Line, a, it.next().?, .{});
    defer parsed.deinit();
    try testing.expectEqualStrings("row_invariance", parsed.value.check);
    try testing.expectEqualStrings("MlxError", parsed.value.err);
    try testing.expectEqualStrings(msg, parsed.value.msg);
    try testing.expect(!parsed.value.ok);
    try testing.expectEqualStrings("", it.next().?);
    try testing.expect(it.next() == null);
}

test "dsv41 kernels: an HCTAPE combine word is judged against its f32 chain, not one f64-rounded word" {
    // Bar: a cancelled output the chain moves six words passes, a wrong word or binding does not.
    const t: HcTerms = .{ .x = 1.3867188e-1, .r = .{ -4.9023438e-1, -1.8261719e-1, -1.734375, 1.671875 }, .post = 8.703464e-1, .comb = .{ 8.401252e-3, 5.460165e-1, 8.935257e-1, 9.1684204e-1 } };
    const e = hcExact(t);
    try testing.expectEqual(@as(u16, 0xb5c1), hcChainWord(t));
    try testing.expectEqual(@as(u16, 0xb5bb), bf16Bits(@floatCast(e.value)));
    try testing.expect(hcWordWithinChain(0xb5c1, e));
    const e1 = hcExact(.{ .x = 1.0, .r = .{ 0.5, 0.25, 0.125, 0.0625 }, .post = 1.0, .comb = .{ 1, 1, 1, 1 } });
    try testing.expect(hcWordWithinChain(0x3FF8, e1));
    try testing.expect(!hcWordWithinChain(0x3FF9, e1));
    var prng: std.Random.DefaultPrng = .init(20260928);
    const r = prng.random();
    const n = 5120;
    var refused: usize = 0;
    for (0..n) |_| {
        var rs: [4]f32 = undefined;
        for (&rs) |*v| v.* = @floatCast(bf16Value(bf16Bits(@floatCast(r.floatNorm(f64) * 2.0))));
        const x: f32 = @floatCast(bf16Value(bf16Bits(@floatCast(r.floatNorm(f64) * 0.5))));
        var comb: [16]f32 = undefined;
        for (&comb) |*v| v.* = @floatCast(r.float(f64));
        for (0..4) |kk| {
            var ok: HcTerms = .{ .x = x, .r = rs, .post = @floatCast(0.2 + 1.6 * r.float(f64)), .comb = undefined };
            var swapped = ok;
            for (0..4) |j| {
                ok.comb[j] = comb[j * 4 + kk];
                swapped.comb[j] = comb[kk * 4 + j];
            }
            try testing.expect(hcWordWithinChain(hcChainWord(ok), hcExact(ok)));
            refused += @intFromBool(!hcWordWithinChain(hcChainWord(swapped), hcExact(ok)));
        }
    }
    try testing.expect(refused * 10 > 9 * 4 * n);
}

// GPU only: DSV41_KERNELS_GPU=1.
// DSV41_KERNELS_RECEIPT=<path> keeps the per-check JSON lines.
test "dsv41 kernels gpu: every kernel of record passes its self-check" {
    _ = env.get(.DSV41_KERNELS_GPU) orelse return error.SkipZigTest;
    const a = testing.allocator;
    mlx.installErrorHandler();
    var diag: xk.Diag = .{};
    var reg = xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag) catch |e| {
        std.debug.print("exl3 kernels refused: {s}\n", .{diag.message()});
        return e;
    };
    defer reg.deinit();
    const stream = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    var bound = reg.bind(stream, &diag) catch |e| {
        std.debug.print("exl3 kernels bind refused: {s}\n", .{diag.message()});
        return e;
    };
    defer bound.deinit();
    var report: Report = .{};
    defer report.deinit(a);
    // The whole plan, then the acceptance verdict over it.
    const verdict: anyerror!void = if (runOver(a, &reg, &bound, .full, .full, &report)) |_| judge(&report, &diag) else |e| e;
    try mlx.check(mlx.mlx_synchronize(stream));
    const lines = try report.writeJsonLines(a);
    defer a.free(lines);
    std.debug.print("{s}", .{lines});
    if (env.get(.DSV41_KERNELS_RECEIPT)) |path| try std.Io.Dir.cwd().writeFile(testing.io, .{ .sub_path = std.mem.span(path), .data = lines });
    std.debug.print("[exl3 selfcheck] {d} checks, {d} failed {s}\n", .{ report.results.items.len, report.failures(), diag.message() });
    try verdict;
}

test "dsv41 selfcheck: the construction plan compiles every kernel and probes nothing; the full plan is the manifest's" {
    const a = testing.allocator;
    var diag: xk.Diag = .{};
    var reg = try xk.Registry.init(a, &xk.embedded, xk.manifest_sha256, &diag);
    defer reg.deinit();
    const every: std.EnumSet(Kernel) = .full;
    const full = plan(&reg, every, .full);
    const compile = plan(&reg, every, .compile);
    var n_full: usize = 0;
    for (&reg.entries) |*e| {
        var manifest = e.checks;
        manifest.remove(.layout_guard);
        try testing.expect(full.get(e.kernel).eql(manifest));
        // every manifest kernel has its compile check; the construction runs it and nothing else
        try testing.expect(manifest.contains(.compile));
        try testing.expect(compile.get(e.kernel).eql(std.EnumSet(Check).initOne(.compile)));
        n_full += manifest.count();
    }
    std.debug.print("\nself-check plan: full {d} checks (the device test), construction {d} compiles\n", .{ n_full, reg.entries.len });
}
