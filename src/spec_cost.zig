//! Measured spec-decode cost model.
//!
//! Both speculative width knobs — the MTP draft depth and the DFlash/DSpark
//! block — were fenced by hand-typed per-silicon tables
//! (`mtp.adaptiveDepthCapForMachine`, `dflash.blockCapForMachine`) and by an
//! EV round-cost surface (`generate.MTP_EV_DEFAULT_COSTS`) that has been
//! refit four times. Every one of those constants was fitted on ONE machine,
//! at ONE quant width, at ONE context length, and is then applied at all of
//! them.
//!
//! This module is the pure decision layer over a MEASURED cost curve: a
//! boot-time ladder of verify-forward costs on THIS machine with THIS
//! checkpoint. It is deliberately free of MLX — everything here is unit
//! tested against the numbers the hand fits were derived from, so a probe
//! that measures the wrong thing shows up as a fit that does not reproduce
//! the shipped constants.
//!
//! Unit discipline (the two knobs do NOT share one):
//!   * a WIDTH curve is indexed by verify rows (what the probe forwards, and
//!     what a DFlash block is),
//!   * a DEPTH curve is indexed by drafts (what `--mtp-depth` is, and what
//!     the EV controller's `T(k)` doc numbers are). Depth k verifies k+1
//!     rows — `depthCurveFromWidthCurve` is the ONE conversion.

const std = @import("std");

/// Curve capacity. The widest verify any lane serves is 16 (the NAX m16
/// tile); a curve never needs more entries than that.
pub const MAX_ENTRIES: usize = 16;

/// Bumped whenever the curve's MEANING changes (what the probe times, the
/// unit of `ms`, the fit's contract). A cached curve carrying a different
/// version is a quiet MISS — never a wrong answer served from disk.
pub const CURVE_VERSION: u32 = 1;

/// A measured cost ladder. `ms[i]` is the cost of one forward/round at
/// `widths[i]`; `kv_ms_per_token` is the learned per-KV-token term B (0 =
/// unlearned, which makes every consumer fall back to the kv-blind fit).
pub const SpecCostCurve = struct {
    n: usize = 0,
    widths: [MAX_ENTRIES]u32 = std.mem.zeroes([MAX_ENTRIES]u32),
    ms: [MAX_ENTRIES]f32 = std.mem.zeroes([MAX_ENTRIES]f32),
    kv_ms_per_token: f32 = 0,
    /// Cost of ONE sequential draft step (the MTP head), measured separately
    /// (`mtp.probeStepMs`) because a verify-forward ladder cannot see it.
    /// An m-deep round is one trunk forward PLUS m head steps, and on a 27B
    /// the head steps DOMINATE the per-position marginal: measured
    /// 2026-08-21 the forward's own marginal at low depth is ~0.8 ms/position
    /// against a hand-fitted composite of ~7.6. Fitting the EV surface from
    /// the forward alone under-prices depth ~9x. Zero = unmeasured.
    draft_ms: f32 = 0,
    /// Per-width kv anchors for the online B fit: the LOWEST and HIGHEST kv
    /// length each width has been observed at, holding the MINIMUM round cost
    /// seen there.
    ///
    /// They live on the MODEL's curve, not on a Generator, because a
    /// Generator is PER REQUEST and a request's kv only spans its own
    /// `max_tokens`. Anchors that reset every request can only identify B on
    /// a single generation longer than `MTP_KV_FIT_MIN_SPAN` tokens, which
    /// almost nothing is — measured live 2026-08-21, a 21k-token prompt
    /// generating 256 tokens never engaged the term at all. The kv VARIATION
    /// that identifies B is ACROSS requests (a 500-token prompt and a 21k
    /// one), so the anchors have to outlive one. Written only from the
    /// inference thread, which is the sole MLX caller.
    kv_lo_len: [MAX_ENTRIES + 1]u32 = std.mem.zeroes([MAX_ENTRIES + 1]u32),
    kv_lo_ms: [MAX_ENTRIES + 1]f32 = std.mem.zeroes([MAX_ENTRIES + 1]f32),
    kv_hi_len: [MAX_ENTRIES + 1]u32 = std.mem.zeroes([MAX_ENTRIES + 1]u32),
    kv_hi_ms: [MAX_ENTRIES + 1]f32 = std.mem.zeroes([MAX_ENTRIES + 1]f32),

    pub fn add(self: *SpecCostCurve, width: u32, ms_val: f32) void {
        if (self.n >= MAX_ENTRIES) return;
        self.widths[self.n] = width;
        self.ms[self.n] = ms_val;
        self.n += 1;
    }

    pub fn at(self: SpecCostCurve, width: u32) ?f32 {
        for (self.widths[0..self.n], self.ms[0..self.n]) |w, m| {
            if (w == width) return m;
        }
        return null;
    }

    pub fn maxWidth(self: SpecCostCurve) u32 {
        var m: u32 = 0;
        for (self.widths[0..self.n]) |w| m = @max(m, w);
        return m;
    }

    /// A curve is usable only when it is monotone in width and has at least
    /// three points — two points cannot tell a flat region from a ramp.
    pub fn usable(self: SpecCostCurve) bool {
        if (self.n < 3) return false;
        var prev_w: u32 = 0;
        var prev_ms: f32 = 0;
        for (self.widths[0..self.n], self.ms[0..self.n]) |w, m| {
            if (w <= prev_w) return false;
            if (!std.math.isFinite(m) or m <= 0) return false;
            if (m < prev_ms) return false; // a wider forward is never cheaper
            prev_w = w;
            prev_ms = m;
        }
        return true;
    }
};

/// Mirror of `generate.Generator.MtpEvCosts` in FLOOR UNITS. Kept as its own
/// type so this module stays free of the generator (and hermetically
/// testable); `generate.mtpEvCostsFromFit` is the one converter.
pub const EvCosts = struct {
    draft: f32,
    per_pos_lo: f32,
    per_pos_hi: f32,
    flat_max: u32,
    sync: f32,
    nax_from: u32 = 0,
    per_pos_nax: f32 = 0,
};

pub const FitOptions = struct {
    /// A per-position delta stays in the current region while it is under
    /// `region_ratio` x the region's own smallest delta. 1.6 separates the
    /// refit-#4 regions (6.4/8.2/9.0 | 13.6 | 23.45 ms) and is loose enough
    /// that ordinary run-to-run noise does not split a flat region in two.
    region_ratio: f32 = 1.6,
    /// The chunk-A confidence read-back is not visible in a forward ladder
    /// (it is a host sync, not a forward), so it is carried in, not fitted.
    sync: f32 = 0.01,
};

/// Convert a WIDTH curve (verify rows, what the probe measures) into a DEPTH
/// curve (drafts, what the EV controller prices). Depth k verifies k+1 rows,
/// so width 1 — a plain serial step — has no depth and is dropped.
pub fn depthCurveFromWidthCurve(curve: SpecCostCurve) SpecCostCurve {
    var out = SpecCostCurve{ .kv_ms_per_token = curve.kv_ms_per_token, .draft_ms = curve.draft_ms };
    for (curve.widths[0..curve.n], curve.ms[0..curve.n]) |w, m| {
        if (w < 2) continue;
        const depth = w - 1;
        // A depth-k ROUND is the width-(k+1) forward plus k sequential head
        // steps. Omitting the second term is what makes a forward-only fit
        // contradict the hand-measured surface it should reproduce.
        out.add(depth, m + curve.draft_ms * @as(f32, @floatFromInt(depth)));
    }
    return out;
}

/// Per-position marginals in ms, one per depth 1..maxWidth. A gap in the
/// ladder (measured at 4 and 6, say) spreads its delta evenly across the
/// positions it spans — that is exactly how the hand fits read their own
/// sparse sweeps. `out[0]` (depth 1) has no predecessor to difference
/// against and is seeded from the flat region once it is known.
fn marginalsMs(curve: SpecCostCurve, out: []f32) usize {
    var n: usize = 0;
    var prev_w: u32 = 0;
    var prev_ms: f32 = 0;
    for (curve.widths[0..curve.n], curve.ms[0..curve.n]) |w, m| {
        if (prev_w != 0) {
            const span = w - prev_w;
            const per = (m - prev_ms) / @as(f32, @floatFromInt(span));
            var k = prev_w + 1;
            while (k <= w) : (k += 1) {
                if (n >= out.len) break;
                out[n] = per;
                n += 1;
            }
        } else {
            if (n < out.len) {
                out[n] = 0; // depth 1: seeded below
                n += 1;
            }
        }
        prev_w = w;
        prev_ms = m;
    }
    return n;
}

/// Fit the EV controller's piecewise surface from a measured DEPTH curve.
///
/// The controller's struct is already the model we want — a flat region, a
/// ramp, and an optional third (NAX-tile) region, all in units of the serial
/// floor — so fitting it leaves `mtpEvMarginalCost`, `mtpEvRoundCost` and
/// `mtpEvPlanFor` completely untouched.
///
/// The `draft`/`per_pos_*` split is not separately identifiable from a round
/// ladder (only the sums enter the controller), so the flat region's
/// composite is split evenly — which is exactly the split the shipped
/// constants carry.
///
/// Returns null when the curve cannot support a fit; the caller keeps its
/// table constant.
pub fn fitEvCosts(curve: SpecCostCurve, opts: FitOptions) ?EvCosts {
    if (!curve.usable()) return null;
    var marg: [MAX_ENTRIES + 1]f32 = undefined;
    const n = marginalsMs(curve, &marg);
    if (n < 3) return null;

    // Flat region: from depth 2 outward while the delta stays within
    // region_ratio of the region's smallest.
    var lo_min: f32 = marg[1];
    var flat_end: usize = 1; // index of the last flat depth
    {
        var i: usize = 1;
        while (i < n) : (i += 1) {
            if (marg[i] > lo_min * opts.region_ratio) break;
            lo_min = @min(lo_min, marg[i]);
            flat_end = i;
        }
    }
    var lo_sum: f32 = 0;
    for (marg[1 .. flat_end + 1]) |m| lo_sum += m;
    const lo_ms = lo_sum / @as(f32, @floatFromInt(flat_end));
    if (!(lo_ms > 0)) return null;

    // Depth 1's own marginal is unobservable (no T(0)), so it takes the flat
    // region's rate — which is what makes the floor solvable.
    marg[0] = lo_ms;
    const floor_ms = curve.ms[0] - marg[0] * @as(f32, @floatFromInt(curve.widths[0]));
    if (!(floor_ms > 0)) return null;

    // Second region: the ramp, until a further jump (the NAX tile).
    var hi_sum: f32 = 0;
    var hi_count: usize = 0;
    var hi_min: f32 = 0;
    var nax_from: u32 = 0;
    var nax_sum: f32 = 0;
    var nax_count: usize = 0;
    {
        var i: usize = flat_end + 1;
        while (i < n) : (i += 1) {
            if (hi_count == 0) {
                hi_min = marg[i];
            } else if (marg[i] > hi_min * opts.region_ratio) {
                nax_from = @intCast(i + 1);
                break;
            } else {
                hi_min = @min(hi_min, marg[i]);
            }
            hi_sum += marg[i];
            hi_count += 1;
        }
        if (nax_from != 0) {
            var j: usize = nax_from - 1;
            while (j < n) : (j += 1) {
                nax_sum += marg[j];
                nax_count += 1;
            }
        }
    }

    const lo_composite = lo_ms / floor_ms;
    const draft = lo_composite * 0.5;
    const hi_composite = if (hi_count > 0)
        (hi_sum / @as(f32, @floatFromInt(hi_count))) / floor_ms
    else
        lo_composite;
    const nax_composite = if (nax_count > 0)
        (nax_sum / @as(f32, @floatFromInt(nax_count))) / floor_ms
    else
        0;

    return .{
        .draft = draft,
        .per_pos_lo = lo_composite - draft,
        .per_pos_hi = @max(hi_composite - draft, 0),
        .flat_max = @intCast(flat_end + 1),
        .sync = opts.sync,
        .nax_from = nax_from,
        .per_pos_nax = if (nax_count > 0) @max(nax_composite - draft, 0) else 0,
    };
}

/// Widest width worth speculating at, from the measured curve alone.
///
/// The criterion is cost PER VERIFIED POSITION — `T(w)/w`, the width's own
/// best case. It falls while the forward is read-bound (weights and KV are
/// read once whatever `w` is) and turns back up at the GEMM tile cliff,
/// which is precisely what the per-silicon cap tables encode by hand.
/// `tolerance` is the fraction by which a width may exceed the running best
/// before it counts as past the cliff.
pub fn cliffCapFromCurve(curve: SpecCostCurve, tolerance: f32) u32 {
    if (!curve.usable()) return 0;
    var best: f32 = std.math.floatMax(f32);
    var cap: u32 = curve.widths[0];
    for (curve.widths[0..curve.n], curve.ms[0..curve.n]) |w, m| {
        const per = m / @as(f32, @floatFromInt(w));
        if (per > best * (1.0 + tolerance)) break;
        if (per < best) best = per;
        cap = w;
    }
    return cap;
}

/// The cliff, re-evaluated at a KV length.
///
/// `T(w, L) ~= T(w, 0) + B*L`: the KV read is shared across all `w` rows, so
/// the whole ladder shifts up by the SAME amount and the per-position
/// criterion moves in favour of wider widths. The optimal width therefore
/// RISES with context — which is why a single boot-measured cap, applied at
/// every context length, leaves width on the table exactly where decode is
/// slowest. `kv_ms_per_token` of 0 (unlearned) reduces this to
/// `cliffCapFromCurve`.
pub fn cliffCapAtKv(curve: SpecCostCurve, tolerance: f32, kv_ms_per_token: f32, kv_len: u32) u32 {
    if (!(kv_ms_per_token > 0) or kv_len == 0) return cliffCapFromCurve(curve, tolerance);
    var shifted = curve;
    const add = kv_ms_per_token * @as(f32, @floatFromInt(kv_len));
    for (shifted.ms[0..shifted.n]) |*m| m.* += add;
    return cliffCapFromCurve(shifted, tolerance);
}

// ── Cache key + persistence ─────────────────────────────────────────────

/// Identity of a measurement: the same probe on a different chip, a
/// different checkpoint, a different quant geometry or a different OS build
/// is a different curve. A key collision would serve one machine's cliff to
/// another, so every field the cost surface actually depends on is in it.
pub fn cacheKey(buf: []u8, chip: []const u8, model_dir: []const u8, quant: []const u8, os_build: []const u8) []const u8 {
    var h = std.hash.Fnv1a_64.init();
    for ([_][]const u8{ chip, model_dir, quant, os_build }) |part| {
        h.update(part);
        h.update("\x00");
    }
    return std.fmt.bufPrint(buf, "v{d}-{x:0>16}", .{ CURVE_VERSION, h.final() }) catch buf[0..0];
}

pub fn serialize(buf: []u8, curve: SpecCostCurve) ![]const u8 {
    var w = std.Io.Writer.fixed(buf);
    try w.print("{{\"version\":{d},\"kv_ms_per_token\":{d:.6},\"draft_ms\":{d:.4},\"widths\":[", .{ CURVE_VERSION, curve.kv_ms_per_token, curve.draft_ms });
    for (curve.widths[0..curve.n], 0..) |x, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{d}", .{x});
    }
    try w.writeAll("],\"ms\":[");
    for (curve.ms[0..curve.n], 0..) |x, i| {
        if (i != 0) try w.writeAll(",");
        try w.print("{d:.6}", .{x});
    }
    try w.writeAll("]}");
    return w.buffered();
}

/// Parse a persisted curve. A version mismatch, a shape mismatch or any
/// unusable content is a QUIET MISS (null) — the caller re-probes or keeps
/// its table. Never a partially-applied curve.
pub fn parse(allocator: std.mem.Allocator, text: []const u8) ?SpecCostCurve {
    var parsed = std.json.parseFromSlice(std.json.Value, allocator, text, .{}) catch return null;
    defer parsed.deinit();
    const obj = switch (parsed.value) {
        .object => |o| o,
        else => return null,
    };
    const ver = obj.get("version") orelse return null;
    switch (ver) {
        .integer => |i| if (i != CURVE_VERSION) return null,
        else => return null,
    }
    const widths = switch (obj.get("widths") orelse return null) {
        .array => |a| a,
        else => return null,
    };
    const ms = switch (obj.get("ms") orelse return null) {
        .array => |a| a,
        else => return null,
    };
    if (widths.items.len != ms.items.len or widths.items.len == 0) return null;
    var out = SpecCostCurve{};
    if (obj.get("draft_ms")) |d| out.draft_ms = switch (d) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => 0,
    };
    if (obj.get("kv_ms_per_token")) |k| out.kv_ms_per_token = switch (k) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => 0,
    };
    for (widths.items, ms.items) |wv, mv| {
        const w: u32 = switch (wv) {
            .integer => |i| if (i > 0 and i <= MAX_ENTRIES) @intCast(i) else return null,
            else => return null,
        };
        const m: f32 = switch (mv) {
            .float => |f| @floatCast(f),
            .integer => |i| @floatFromInt(i),
            else => return null,
        };
        out.add(w, m);
    }
    if (!out.usable()) return null;
    return out;
}

// ── Boot resolution: env gate, on-disk cache, probe ─────────────────────

const log = @import("log.zig");

/// The ladder the probe times. 1 is the serial reference (every curve needs
/// it — the floor is what the EV surface is expressed in); 8 is MAX_DEPTH+1
/// and the widest block any lane on an M1-M4 serves.
pub const DEFAULT_WIDTHS = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
/// Timed reps per width, on top of one DISCARDED warm-up rep.
pub const DEFAULT_REPS: u32 = 3;

/// `MLX_SERVE_SPEC_COST_PROBE=0` restores the hand-typed tables verbatim —
/// the A/B arm, and the escape hatch if the probe misjudges a machine.
pub fn probeEnabled() bool {
    const raw = std.c.getenv("MLX_SERVE_SPEC_COST_PROBE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(raw), "0");
}

fn homeDir() []const u8 {
    return std.mem.span(std.c.getenv("HOME") orelse return "/tmp");
}

fn cachePath(buf: []u8, key: []const u8) ?[]const u8 {
    return std.fmt.bufPrint(buf, "{s}/.mlx-serve/spec-cost/{s}.json", .{ homeDir(), key }) catch null;
}

/// Read a persisted curve. Any failure — missing file, unreadable, stale
/// version, unusable shape — is a QUIET miss.
pub fn loadCached(allocator: std.mem.Allocator, io: std.Io, key: []const u8) ?SpecCostCurve {
    var path_buf: [512]u8 = undefined;
    const path = cachePath(&path_buf, key) orelse return null;
    const f = std.Io.Dir.openFileAbsolute(io, path, .{}) catch return null;
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const text = rs.interface.allocRemaining(allocator, .limited(8192)) catch return null;
    defer allocator.free(text);
    return parse(allocator, text);
}

/// Persist a curve. Best-effort: a machine that cannot write the cache
/// simply re-probes next boot (~1-2 s), so no failure here is worth an
/// error path.
pub fn storeCached(io: std.Io, key: []const u8, curve: SpecCostCurve) void {
    var dir_buf: [512]u8 = undefined;
    const dir = std.fmt.bufPrint(&dir_buf, "{s}/.mlx-serve/spec-cost", .{homeDir()}) catch return;
    std.Io.Dir.cwd().createDirPath(io, dir) catch return;
    var path_buf: [512]u8 = undefined;
    const path = cachePath(&path_buf, key) orelse return;
    var text: [4096]u8 = undefined;
    const body = serialize(&text, curve) catch return;
    const f = std.Io.Dir.createFileAbsolute(io, path, .{}) catch return;
    defer f.close(io);
    var wb: [4096]u8 = undefined;
    var fw = f.writer(io, &wb);
    fw.interface.writeAll(body) catch return;
    fw.interface.flush() catch {};
}

/// Resolve this model's cost curve: cache hit, else probe + persist.
/// `xfm` is anything exposing `probeSpecCostCurve` (the Transformer) — kept
/// generic so this module stays free of MLX and of the forward pass.
///
/// MUST be called on the inference thread: the probe forwards.
pub fn resolve(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: anytype,
    key: []const u8,
) ?SpecCostCurve {
    if (!probeEnabled()) {
        log.info("[spec-cost] probe disabled (MLX_SERVE_SPEC_COST_PROBE=0) — per-silicon tables apply\n", .{});
        return null;
    }
    if (loadCached(allocator, io, key)) |cached| {
        logCurve("cached", cached);
        return cached;
    }
    const curve = xfm.probeSpecCostCurve(io, &DEFAULT_WIDTHS, DEFAULT_REPS) catch |err| {
        log.warn("[spec-cost] probe failed ({s}) — per-silicon tables apply\n", .{@errorName(err)});
        return null;
    };
    if (!curve.usable()) {
        log.warn("[spec-cost] probe produced an unusable curve — per-silicon tables apply\n", .{});
        return null;
    }
    storeCached(io, key, curve);
    logCurve("measured", curve);
    return curve;
}

/// One boot line naming the widths and their measured cost. A fence nobody
/// can see is a fence nobody can debug — and that applies to a measured
/// fence exactly as it does to a hand-typed one.
pub fn logCurve(source: []const u8, curve: SpecCostCurve) void {
    var buf: [256]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    for (curve.widths[0..curve.n], curve.ms[0..curve.n], 0..) |width, ms_val, i| {
        if (i != 0) w.writeAll(" ") catch break;
        w.print("{d}:{d:.1}", .{ width, ms_val }) catch break;
    }
    log.info("[spec-cost] {s} verify ladder (ms/forward) {s}\n", .{ source, w.buffered() });
}

// ── Tests ───────────────────────────────────────────────────────────────

const testing = std.testing;

/// The numbers the shipped `MTP_EV_DEFAULT_COSTS` was hand-fitted from
/// (refit #4, 2026-08-15, M4 Max, Jundot oQ4e 27B @8K). These are ROUND
/// costs at DEPTH k, so they feed `fitEvCosts` directly.
fn refit4DepthCurve() SpecCostCurve {
    var c = SpecCostCurve{};
    c.add(1, 44.6);
    c.add(2, 51.0);
    c.add(3, 59.2);
    c.add(4, 68.2);
    c.add(6, 95.4);
    c.add(8, 142.3);
    return c;
}

test "fitEvCosts reproduces the hand-fitted refit-#4 surface" {
    const fit = fitEvCosts(refit4DepthCurve(), .{}).?;
    // Shipped: .draft=0.10 .per_pos_lo=0.10 .per_pos_hi=0.26 .flat_max=4
    //          .nax_from=7 .per_pos_nax=0.52
    try testing.expectEqual(@as(u32, 4), fit.flat_max);
    try testing.expectEqual(@as(u32, 7), fit.nax_from);
    try testing.expectApproxEqAbs(@as(f32, 0.10), fit.draft, 0.02);
    try testing.expectApproxEqAbs(@as(f32, 0.10), fit.per_pos_lo, 0.02);
    try testing.expectApproxEqAbs(@as(f32, 0.26), fit.per_pos_hi, 0.02);
    try testing.expectApproxEqAbs(@as(f32, 0.52), fit.per_pos_nax, 0.03);
    try testing.expectApproxEqAbs(@as(f32, 0.01), fit.sync, 1e-6);
}

test "fitEvCosts: the composite marginals are what the controller consumes" {
    const fit = fitEvCosts(refit4DepthCurve(), .{}).?;
    // draft + per_pos_* is the only combination the controller reads, so the
    // bar is on the sums, not on a split that is not identifiable.
    try testing.expectApproxEqAbs(@as(f32, 0.20), fit.draft + fit.per_pos_lo, 0.02);
    try testing.expectApproxEqAbs(@as(f32, 0.36), fit.draft + fit.per_pos_hi, 0.02);
    try testing.expectApproxEqAbs(@as(f32, 0.62), fit.draft + fit.per_pos_nax, 0.03);
}

test "fitEvCosts: a two-region curve leaves the NAX region off" {
    var c = SpecCostCurve{};
    c.add(1, 44.6);
    c.add(2, 51.0);
    c.add(3, 59.2);
    c.add(4, 68.2);
    c.add(6, 95.4);
    const fit = fitEvCosts(c, .{}).?;
    try testing.expectEqual(@as(u32, 0), fit.nax_from);
    try testing.expectEqual(@as(f32, 0), fit.per_pos_nax);
    try testing.expectEqual(@as(u32, 4), fit.flat_max);
}

test "fitEvCosts declines a curve too short or non-monotone to fit" {
    var short = SpecCostCurve{};
    short.add(1, 44.6);
    short.add(2, 51.0);
    try testing.expect(fitEvCosts(short, .{}) == null);
    var bad = SpecCostCurve{};
    bad.add(1, 44.6);
    bad.add(2, 51.0);
    bad.add(3, 40.0); // a wider forward is never cheaper
    try testing.expect(fitEvCosts(bad, .{}) == null);
}

test "cliffCapFromCurve finds the M4 generic depth cap of 6" {
    // Cost per position falls to depth 6 (15.9 ms/pos) and turns back up at
    // 8 (17.8) — the split-K verify lane's M=7 ceiling, which is exactly
    // what MTP_ADAPTIVE_DEFAULT_CAP encodes by hand.
    try testing.expectEqual(@as(u32, 6), cliffCapFromCurve(refit4DepthCurve(), 0.05));
}

test "cliffCapFromCurve finds an early cliff (the M1 Pro width-6 shape)" {
    // M1 Pro measured 13.01 tok/s at depth 4 against 10.78/9.63 at 5/6 —
    // a curve whose per-position cost turns up right after 4.
    var c = SpecCostCurve{};
    c.add(1, 44.0);
    c.add(2, 50.0);
    c.add(3, 57.0);
    c.add(4, 64.0);
    c.add(5, 90.0);
    c.add(6, 120.0);
    try testing.expectEqual(@as(u32, 4), cliffCapFromCurve(c, 0.05));
}

test "cliffCapFromCurve: a curve that never turns up caps at its widest" {
    var c = SpecCostCurve{};
    c.add(1, 40.0);
    c.add(2, 42.0);
    c.add(4, 46.0);
    c.add(8, 54.0);
    try testing.expectEqual(@as(u32, 8), cliffCapFromCurve(c, 0.05));
}

test "depthCurveFromWidthCurve: depth k verifies k+1 rows AND pays k draft steps" {
    var w = SpecCostCurve{};
    w.add(1, 38.0);
    w.add(2, 44.6);
    w.add(3, 51.0);
    const blind = depthCurveFromWidthCurve(w);
    try testing.expectEqual(@as(usize, 2), blind.n);
    try testing.expectEqual(@as(u32, 1), blind.widths[0]);
    try testing.expectApproxEqAbs(@as(f32, 44.6), blind.ms[0], 1e-5);
    // With a measured head step the round costs k of them on top. Live
    // 2026-08-21, Qwen3.8-27B oQ4e on M4 Max: the forward marginal at low
    // depth is 0.8 ms/position, so a forward-only fit prices depth at ~1/9
    // of the hand-measured composite and drafts far too deep.
    w.draft_ms = 6.8;
    const paid = depthCurveFromWidthCurve(w);
    try testing.expectApproxEqAbs(@as(f32, 44.6 + 6.8), paid.ms[0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 51.0 + 13.6), paid.ms[1], 1e-4);
}

test "the draft step is what makes the measured fit reproduce the hand surface" {
    // The live M4 Max ladder for the very checkpoint MTP_EV_DEFAULT_COSTS was
    // fitted on (Jundot Qwen3.8-27B oQ4e), plus its measured head step.
    var w = SpecCostCurve{};
    w.add(1, 37.5);
    w.add(2, 40.3);
    w.add(3, 41.1);
    w.add(4, 45.4);
    w.add(5, 51.0);
    w.add(6, 58.8);
    w.add(7, 67.8);
    w.add(8, 91.4);
    // Forward-only: the marginals collapse to ~1/10 of the hand fit.
    const blind = fitEvCosts(depthCurveFromWidthCurve(w), .{}).?;
    try testing.expect(blind.draft + blind.per_pos_lo < 0.05);
    // Fold in a head step of the magnitude the hand fit implies (~6.8 ms:
    // its 0.20 composite over a ~38 ms floor is 7.6 ms/position, against the
    // 0.8 the forward contributes) and the flat composite lands back on the
    // hand-fitted surface. The bar is the ORDER OF MAGNITUDE — a fit that
    // reads a tenth of the hand number is measuring the wrong thing, which
    // is exactly what this caught live.
    w.draft_ms = 6.8;
    const paid = fitEvCosts(depthCurveFromWidthCurve(w), .{}).?;
    try testing.expectApproxEqAbs(@as(f32, 0.20), paid.draft + paid.per_pos_lo, 0.06);
    try testing.expect(paid.draft + paid.per_pos_lo > (blind.draft + blind.per_pos_lo) * 4);
}

test "cacheKey: every identity field moves the key, and the version prefixes it" {
    var a: [64]u8 = undefined;
    var b: [64]u8 = undefined;
    const base = cacheKey(&a, "Apple M4 Max", "/m/qwen", "q4g64", "26.2");
    try testing.expect(std.mem.startsWith(u8, base, "v1-"));
    inline for (.{
        .{ "Apple M3 Ultra", "/m/qwen", "q4g64", "26.2" },
        .{ "Apple M4 Max", "/m/other", "q4g64", "26.2" },
        .{ "Apple M4 Max", "/m/qwen", "q6g64", "26.2" },
        .{ "Apple M4 Max", "/m/qwen", "q4g64", "26.3" },
    }) |v| {
        try testing.expect(!std.mem.eql(u8, base, cacheKey(&b, v[0], v[1], v[2], v[3])));
    }
}

test "curve round-trips through serialize/parse" {
    var c = refit4DepthCurve();
    c.kv_ms_per_token = 0.00125;
    c.draft_ms = 6.8;
    var buf: [512]u8 = undefined;
    const text = try serialize(&buf, c);
    const back = parse(testing.allocator, text).?;
    try testing.expectEqual(c.n, back.n);
    for (0..c.n) |i| {
        try testing.expectEqual(c.widths[i], back.widths[i]);
        try testing.expectApproxEqAbs(c.ms[i], back.ms[i], 1e-3);
    }
    try testing.expectApproxEqAbs(c.kv_ms_per_token, back.kv_ms_per_token, 1e-6);
    try testing.expectApproxEqAbs(c.draft_ms, back.draft_ms, 1e-4);
}

test "a stale cache version is a MISS, never a wrong answer" {
    const stale = "{\"version\":0,\"kv_ms_per_token\":0,\"widths\":[1,2,3],\"ms\":[40,44,50]}";
    try testing.expect(parse(testing.allocator, stale) == null);
    const current = "{\"version\":1,\"kv_ms_per_token\":0,\"widths\":[1,2,3],\"ms\":[40,44,50]}";
    try testing.expect(parse(testing.allocator, current) != null);
    // Truncated / mismatched shapes are misses too.
    try testing.expect(parse(testing.allocator, "{\"version\":1,\"widths\":[1,2],\"ms\":[40]}") == null);
    try testing.expect(parse(testing.allocator, "not json") == null);
}

test "cliffCapAtKv: the optimal width RISES with context" {
    var c = SpecCostCurve{};
    c.add(1, 38.0);
    c.add(2, 44.6);
    c.add(3, 51.0);
    c.add(4, 59.2);
    c.add(5, 68.2);
    c.add(6, 82.0);
    c.add(7, 110.0);
    c.add(8, 150.0);
    const short = cliffCapFromCurve(c, 0.05);
    const long = cliffCapAtKv(c, 0.05, 0.002, 32000);
    try testing.expect(long > short);
    // Unlearned kv term = the boot-measured cliff, unchanged.
    try testing.expectEqual(short, cliffCapAtKv(c, 0.05, 0, 32000));
    try testing.expectEqual(short, cliffCapAtKv(c, 0.05, 0.002, 0));
}
