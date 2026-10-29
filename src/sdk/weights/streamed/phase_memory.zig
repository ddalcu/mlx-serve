//! The memory bills' shared machinery (DeepSeek-V4.1's `arch/deepseek_v41/bill.zig` and GLM-5.3's
//! `arch/glm_moe_dsa/bill.zig`, and their modules): the standard request and the fill's floor, the page tables of a phase's wired bytes and their
//! row-following term, one phase boundary's record of the billed terms against what the kernel and MLX measured (its
//! residuals), the construction check's arithmetic, a phase change's settle, the page-cache check the bills' assumption
//! rests on, and the receipts tests' checks of a record and of the fill's edge. A bill keeps its own terms (a struct of
//! named fields, `Terms`) and each module its own conditions and refusals; the machinery is the same for both.

const std = @import("std");
const sdk = @import("sdk");
const mlx = sdk.mlx;

/// The page granule of the kernel and of the GPU (ARM64 16 KiB).
pub const wire_page_bytes: u64 = 16_384;

/// The memory the wiring of `wired` bytes costs outside the process footprint (DeepSeek-V4.1's measurements, its
/// ledger sec. 73: +0.091-0.126 GB in every served run 19 cell): per 16 KiB page a CPU and a GPU leaf entry and the
/// kernel's wiring record (8 B each), one CPU + GPU L2 entry per 32 MiB and L1 entry per 64 GiB. No per-buffer term:
/// ~4,000 live buffers at construction left no room for one under the measured figure. Monotone and subadditive, so a
/// fill's per-row step can carry `wireTables(per_row)`.
pub fn wireTables(wired: u64) u64 {
    return (std.math.divCeil(u64, wired, wire_page_bytes) catch unreachable) * 24 +
        (std.math.divCeil(u64, wired, 1 << 25) catch unreachable) * 16 +
        (std.math.divCeil(u64, wired, 1 << 36) catch unreachable) * 16;
}

/// The wiring overhead per live buffer if all of it were per buffer (measured on DeepSeek-V4.1): the largest
/// construction reading, 125,852,608 B (run 3bk control1), over the 4,493 buffers live there, rounded up to a KiB.
pub const wire_buffer_bytes: u64 = 28_672;

/// The sum of every field of `t` (a bill's phase terms: a struct of u64 fields).
pub fn sumFields(t: anytype) u64 {
    var n: u64 = 0;
    inline for (@typeInfo(@TypeOf(t)).@"struct".field_names) |name| n += @field(t, name);
    return n;
}

/// One phase boundary's memory record over a bill's phase terms `Terms`: the phase's billed terms, the process
/// ledgers (the footprint and its split), MLX's allocator (active, cache, the peak since the previous boundary), the
/// box's pages, and billed minus measured. The fields and their order are the receipt's (DeepSeek-V4.1's
/// `NATIVE DSV41_PHASE_MEMORY`, GLM-5.3's `glm_moe_dsa: phase memory`).
pub fn Record(comptime Terms: type) type {
    return struct {
        phase: []const u8,
        billed: Terms,
        billed_process_bytes: u64,
        process: sdk.memory.ProcessMemory,
        mlx_active_bytes: u64,
        mlx_cache_bytes: u64,
        mlx_peak_bytes: u64,
        box_physical_used_bytes: u64,
        box_file_backed_bytes: u64,
        /// Speculative (read-ahead) pages: not in the guard's used count until the kernel ages them into
        /// inactive, which it can do at any later allocation (DeepSeek-V4.1's v6c2 / SERVED kills: 7.7 GB at the grow).
        box_speculative_bytes: u64 = 0,
        /// The page cache the step created: file-backed pages now less at the step's vm start (the bills assume 0).
        file_cache_created_bytes: i64 = 0,
        /// The phase's billed process bytes less its measured footprint high-water mark (negative: over the bill).
        residual_bytes: i64,
        /// The bill's MLX-active terms (its `device` sum) less MLX's peak over the phase.
        mlx_residual_bytes: i64,
        /// A phase change's boundary only: how long the driver took to reclaim the frees before the grow.
        settle_ms: ?u32 = null,
        /// The box's wired pages (vm_stat), for wire_tables' proof: box wired less this process's graphics footprint
        /// less the idle census's wired.
        box_wired_bytes: u64 = 0,
    };
}

/// A boundary's reading from what the kernel and MLX already track (no new counter): MLX's allocator (active, cache,
/// the peak since the last `restart`), the process's ledgers, the box's pages.
pub const Reading = struct {
    active: u64,
    cache: u64,
    peak: u64,
    process: sdk.memory.ProcessMemory,
    vm: sdk.memory.VmBytes,

    pub fn now() Reading {
        var active: usize = 0;
        var cache: usize = 0;
        var peak: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        _ = mlx.mlx_get_peak_memory(&peak);
        return .{ .active = active, .cache = cache, .peak = peak, .process = sdk.memory.processMemory(), .vm = sdk.memory.vmBytes() };
    }
};

/// Restarts both high-water marks for the next phase: MLX's peak and the footprint's interval.
pub fn restart() void {
    _ = mlx.mlx_reset_peak_memory();
    sdk.memory.startFootprintInterval();
}

/// A record's arithmetic (host-testable): `process` the billed process bytes (the terms' sum), `device` the billed
/// terms MLX's active bytes hold; the residuals against the footprint's high-water mark and MLX's peak.
pub fn recordOf(comptime Terms: type, phase: []const u8, billed: Terms, process: u64, device: u64, pm: sdk.memory.ProcessMemory, active: u64, cache: u64, peak: u64, physical: u64, file_backed: u64, speculative: u64, file_backed_start: u64) Record(Terms) {
    const measured = @max(pm.footprint_interval_peak, pm.footprint);
    return .{
        .phase = phase,
        .billed = billed,
        .billed_process_bytes = process,
        .process = pm,
        .mlx_active_bytes = active,
        .mlx_cache_bytes = cache,
        .mlx_peak_bytes = @max(peak, active),
        .box_physical_used_bytes = physical,
        .box_file_backed_bytes = file_backed,
        .box_speculative_bytes = speculative,
        .file_cache_created_bytes = @as(i64, @intCast(file_backed)) - @as(i64, @intCast(file_backed_start)),
        .residual_bytes = @as(i64, @intCast(process)) - @as(i64, @intCast(measured)),
        .mlx_residual_bytes = @as(i64, @intCast(device)) - @as(i64, @intCast(@max(peak, active))),
    };
}

/// `recordOf` at reading `r`, the box's wired pages beside it.
pub fn recordAt(comptime Terms: type, phase: []const u8, billed: Terms, process: u64, device: u64, r: Reading, file_backed_start: u64) Record(Terms) {
    var rec = recordOf(Terms, phase, billed, process, device, r.process, r.active, r.cache, r.peak, sdk.memory.physicalUsedBytes(r.vm), r.vm.external, r.vm.speculative, file_backed_start);
    rec.box_wired_bytes = r.vm.wired;
    return rec;
}

/// A record as one JSON line's payload (the receipt), in `a`.
pub fn jsonOf(a: std.mem.Allocator, r: anytype) ![]u8 {
    return std.json.Stringify.valueAlloc(a, r, .{});
}

/// The page cache a step may leave: the bills credit only the file cache present at the step's start, so page cache
/// the step creates is unbilled memory that can land at any later allocation.
pub const page_cache_tolerance_bytes: u64 = 500_000_000;

/// The step left no page cache to be aged into the box's count later (DeepSeek-V4.1's v6c2: 15 GB of speculative pages
/// from unaligned F_NOCACHE reads, 7.7 GB aged in at the grow).
pub fn checkPageCache(created: i64) error{ConstructionLeftPageCache}!void {
    if (created > @as(i64, @intCast(page_cache_tolerance_bytes))) return error.ConstructionLeftPageCache;
}

/// The request a construction bills when the model sets no context (`ctx_size`): the standard 16K cell's prompt and
/// token cap (both bills'; a longer request is admitted, or refused by name, by the arch's own rule at its time).
pub const fill_prompt_tokens: u64 = 16384;
pub const fill_max_tokens: u64 = 1024;
/// The fill's floor: fewer persistent rows per routed layer refuse the load by name.
pub const min_fill_rows: u32 = 16;

/// The longest prompt a module serves: its `max_context_tokens`, else the standard request's.
pub fn servedContext(max_context_tokens: ?u32) u64 {
    return max_context_tokens orelse fill_prompt_tokens;
}

/// The page tables at `slots` slot-row bytes (`sdk.MemoryBill.RowTerms`, both bills'): `data` holds each phase's wired
/// bytes less its slot rows.
pub fn wiringAt(data: *const [4]u64, phase: sdk.MemoryBill.Phase, rows: u32, slots: u64) u64 {
    _ = rows;
    return wireTables(data[@intFromEnum(phase)] + slots);
}

/// How far the constructed footprint may sit above the bill's construction terms (the ledger's page rounding and the
/// host side's own movement; DeepSeek-V4.1's cell4 measured 0.59 GB under them).
pub const construction_tolerance_bytes: u64 = 250_000_000;

/// The construction check's arithmetic (both modules'): the bill's construction terms at the built prompt rows against
/// the footprint (every command retired, MLX's cache cleared), and the host side (the footprint less MLX's active and
/// cache) against each measured term's bound. Each module logs and refuses in its own words.
pub const Construction = struct {
    billed: u64,
    footprint: u64,
    active: u64,
    cache: u64,
    /// The host side's part a bill names by construction (GLM-5.3's read pool staging): the measured terms bound the rest.
    host_known: u64 = 0,
    /// The bytes the host's other models hold in this process (`foreignBytes`, read once at the check): the footprint
    /// is the whole process's, the bill this module's, so both bounds rise by them (bytes held now, never a peak).
    foreign: u64 = 0,

    pub fn of(mb: sdk.MemoryBill, prompt_rows: u32, footprint: u64, active: u64, cache: u64) Construction {
        return .{ .billed = mb.constructionBytes(prompt_rows), .footprint = footprint, .active = active, .cache = cache };
    }

    pub fn hostSide(c: Construction) u64 {
        return c.footprint -| c.active -| c.cache;
    }

    pub fn residual(c: Construction) i64 {
        return @as(i64, @intCast(c.billed + c.foreign)) - @as(i64, @intCast(c.footprint));
    }

    /// The footprint within `tolerance` of the construction terms and the other models' bytes (`sdk.checkConstruction`).
    pub fn checkFootprint(c: Construction, tolerance: u64) error{ConstructionOverBill}!void {
        return sdk.checkConstruction(c.billed + c.foreign, c.footprint, tolerance);
    }

    /// The host side the measured terms bound: past the part the bill names by construction and the other models' bytes
    /// (a model outside MLX holds them on the host side; one on MLX leaves this bound the looser by them).
    pub fn measuredHost(c: Construction) u64 {
        return c.hostSide() -| c.host_known -| c.foreign;
    }

    /// The first measured term the measured host side exceeds (`sdk.checkMeasured`), null when none does.
    pub fn overMeasured(c: Construction, mb: sdk.MemoryBill) ?sdk.MemoryBill.Term {
        for (mb.terms) |t| if (t.measured) sdk.checkMeasured(t, c.measuredHost()) catch return t;
        return null;
    }
};

/// The bytes the host's other models hold in this process: #8's `sdk.memory.foreignBytes` (the host keeps it current),
/// 0 where the SDK has no such reading yet. The construction checks raise their bounds by it (`Construction.foreign`).
pub fn foreignBytes() u64 {
    return if (comptime @hasDecl(sdk.memory, "foreignBytes")) sdk.memory.foreignBytes() else 0;
}

comptime {
    // #8 publishes the setter and the reading together: a setter without the reading this file calls would silently
    // read 0.
    if (@hasDecl(sdk.memory, "setForeignBytes") and !@hasDecl(sdk.memory, "foreignBytes")) @compileError("phase_memory: sdk.memory.setForeignBytes without foreignBytes");
}

/// MLX's allocator and this process's footprint at a boundary, without the box's pages (host_statistics64 is
/// rate-limited box-wide for non-platform binaries, so a settle's polls read only this process's own ledgers): MLX's
/// active, cache and peak since the last restart, the footprint and its interval high-water mark.
pub const Ledgers = struct {
    active: u64 = 0,
    cache: u64 = 0,
    peak: u64 = 0,
    footprint: u64 = 0,
    footprint_peak: u64 = 0,

    pub fn now() Ledgers {
        var active: usize = 0;
        var cache: usize = 0;
        var peak: usize = 0;
        _ = mlx.mlx_get_active_memory(&active);
        _ = mlx.mlx_get_cache_memory(&cache);
        _ = mlx.mlx_get_peak_memory(&peak);
        const pm = sdk.memory.processMemory();
        return .{ .active = active, .cache = cache, .peak = @max(peak, active), .footprint = pm.footprint, .footprint_peak = @max(pm.footprint_interval_peak, pm.footprint) };
    }

    /// The footprint less MLX's active and cache: the host side.
    pub fn hostSide(m: Ledgers) u64 {
        return m.footprint -| m.active -| m.cache;
    }
};

/// A phase change's settle (both modules'): how often it reads, how long at most, how far over its expected footprint a
/// reading may sit (the ledger's page rounding and the host side's own movement).
pub const settle_poll_ms: u32 = 5;
pub const settle_max_ms: u32 = 10_000;
pub const settle_tolerance_bytes: u64 = 250_000_000;

/// How far over its reading before a phase change's frees the host side may read once they settled
/// (`HostSideBack`): the ledger's page rounding and the host's own movement between the two readings.
pub const settle_host_tolerance_bytes: u64 = 50_000_000;

/// A settle condition (`settle`) for a phase change whose frees are all MLX's: MLX's cache empty and the host side (the
/// footprint less MLX's active and cache) back at most `settle_host_tolerance_bytes` over `host_before`, its reading
/// before the frees. Until the driver retires a freed buffer the footprint still carries it, and it reads as host side:
/// a footprint test with a tolerance takes such a reading, and whatever reads the footprint next counts the freed bytes
/// twice. DeepSeek-V4.1's module waits on its footprint instead: its `until_freed` holds the footprint under an absolute
/// bound (the decode bill's), which a reading with the freed pages still in it fails as well.
pub fn HostSideBack(comptime R: type) type {
    return struct {
        host_before: u64,

        pub fn ok(c: @This(), m: R) bool {
            return m.cache == 0 and m.hostSide() <= c.host_before + settle_host_tolerance_bytes;
        }
    };
}

/// The live boundary reader of a reading type `R` (`R.now()`): waits on the host's io; a buffer that reached MLX's cache
/// after the boundary's clear (a command's temporaries dropped at its completion) goes back to the system before the
/// next reading.
pub fn LiveReader(comptime R: type) type {
    return struct {
        io: std.Io,

        pub fn now(_: @This()) R {
            return R.now();
        }

        pub fn sleep(self: @This(), ms: u32) void {
            std.Io.sleep(self.io, .fromMilliseconds(ms), .awake) catch {};
        }

        pub fn clearCache(_: @This()) void {
            _ = mlx.mlx_clear_cache();
        }
    };
}

/// A settle's last reading, the time it waited, and whether the condition held on it.
pub fn Settled(comptime R: type) type {
    return struct { after: R, waited_ms: u32, settled: bool };
}

/// After a phase change's frees: `reader` (`now()`, `sleep(ms)`, `clearCache()`) read every `poll_ms` until `cond.ok`
/// holds on its reading (each module's own condition), at most `max_ms`: the footprint's ledger can trail a release while
/// the driver retires it. A reading with MLX's cache not empty clears it before the next. The last reading either way.
pub fn settle(reader: anytype, poll_ms: u32, max_ms: u32, cond: anytype) Settled(@TypeOf(reader.now())) {
    var m = reader.now();
    var waited: u32 = 0;
    while (!cond.ok(m) and waited < max_ms) {
        if (m.cache != 0) reader.clearCache();
        reader.sleep(poll_ms);
        waited += poll_ms;
        m = reader.now();
    }
    return .{ .after = m, .waited_ms = waited, .settled = cond.ok(m) };
}

// ── The receipts tests' checks (a served run's record rebuilt from its bill) ──

/// A record's measured side as the receipt printed it.
pub const Measured = struct { footprint: u64, interval_peak: u64, mlx_active: u64, mlx_cache: u64, mlx_peak: u64, physical: u64, file_backed: u64, speculative: u64, residual: i64, mlx_residual: i64 };

/// The record `r` (rebuilt from the bill's terms and the receipt's measured side) gives the receipt's residuals and
/// page cache back, to the byte.
pub fn expectMeasured(r: anytype, m: Measured, file_backed_start: u64) !void {
    try std.testing.expectEqual(m.residual, r.residual_bytes);
    try std.testing.expectEqual(m.mlx_residual, r.mlx_residual_bytes);
    try std.testing.expectEqual(@as(i64, @intCast(m.file_backed)) - @as(i64, @intCast(file_backed_start)), r.file_cache_created_bytes);
}

/// The fill's edge on the SDK's view of a bill: `sdk.fill` gives `want`; `want` is admitted; one decode row more is over
/// the target, and so is one prompt row more; a target at the larger total keeps the rows and a byte under loses one.
pub fn expectFillEdge(mb: sdk.MemoryBill, baseline: u64, target: u64, n_experts: u32, min_rows: u32, want: sdk.Rows) !void {
    try std.testing.expectEqual(want, try sdk.fill(mb, baseline, target, n_experts, min_rows));
    try sdk.admit(mb, baseline, want, target);
    if (want.decode < n_experts) try std.testing.expectError(error.DecodeOverTarget, sdk.admit(mb, baseline, .{ .prompt = want.prompt, .decode = want.decode + 1 }, target));
    if (want.prompt < want.decode) try std.testing.expectError(error.PromptOverTarget, sdk.admit(mb, baseline, .{ .prompt = want.prompt + 1, .decode = want.decode }, target));
    const edge = @max(mb.total(.prompt, baseline, want.prompt), mb.total(.decode, baseline, want.decode));
    try std.testing.expectEqual(want, try sdk.fill(mb, baseline, edge, n_experts, min_rows));
    if (edge > baseline) if (sdk.fill(mb, baseline, edge - 1, n_experts, min_rows)) |under| {
        try std.testing.expect(under.prompt + under.decode == want.prompt + want.decode - 1);
    } else |_| {};
}

/// A scripted boundary reader for the settle's tests: `readings[i]` at the i-th read (the last one repeats), no real
/// sleep; the sleeps and the cache clears counted.
pub fn FakeReader(comptime R: type) type {
    return struct {
        readings: []const R,
        i: *usize,
        slept_ms: *u32,
        clears: ?*u32 = null,

        pub fn now(self: @This()) R {
            const r = self.readings[@min(self.i.*, self.readings.len - 1)];
            self.i.* += 1;
            return r;
        }

        pub fn sleep(self: @This(), ms: u32) void {
            self.slept_ms.* += ms;
        }

        pub fn clearCache(self: @This()) void {
            if (self.clears) |c| c.* += 1;
        }
    };
}

const testing = std.testing;

test "phase memory: wire tables bill 24 B a page, 16 B per 32 MiB and per 64 GiB, subadditive" {
    try testing.expectEqual(@as(u64, 24 + 16 + 16), wireTables(1));
    try testing.expectEqual(@as(u64, 2048 * 24 + 16 + 16), wireTables(1 << 25));
    try testing.expectEqual(@as(u64, 133_927_424), wireTables(91_398_000_000));
    try testing.expect(wireTables(91_398_000_000 + 532_623_360) <= wireTables(91_398_000_000) + wireTables(532_623_360));
}

test "phase memory: the settle reads until its condition holds, clears a late cache, and stops at its wait" {
    const Cond = struct {
        bound: u64,
        pub fn ok(c: @This(), m: Ledgers) bool {
            return m.cache == 0 and m.footprint <= c.bound;
        }
    };
    var i: usize = 0;
    var slept: u32 = 0;
    var clears: u32 = 0;
    const readings = [_]Ledgers{ .{ .cache = 4096, .footprint = 100 }, .{ .footprint = 100 }, .{ .footprint = 90 } };
    const st = settle(FakeReader(Ledgers){ .readings = &readings, .i = &i, .slept_ms = &slept, .clears = &clears }, settle_poll_ms, settle_max_ms, Cond{ .bound = 90 });
    try testing.expect(st.settled);
    try testing.expectEqual(@as(u32, 2 * settle_poll_ms), st.waited_ms);
    try testing.expectEqual(@as(u32, 1), clears);
    i = 0;
    slept = 0;
    const never = settle(FakeReader(Ledgers){ .readings = readings[0..1], .i = &i, .slept_ms = &slept }, settle_poll_ms, settle_max_ms, Cond{ .bound = 90 });
    try testing.expect(!never.settled);
    try testing.expectEqual(settle_max_ms, never.waited_ms);
}

test "phase memory: a settle on the host side waits while the footprint still carries the freed pages" {
    var i: usize = 0;
    var slept: u32 = 0;
    // The host side 0.92 GB before the frees; after them 1.13 GB (0.21 GB of freed pages), then 0.93 GB.
    const readings = [_]Ledgers{ .{ .active = 80_000_000_000, .footprint = 81_130_000_000 }, .{ .active = 80_000_000_000, .footprint = 80_930_000_000 } };
    const st = settle(FakeReader(Ledgers){ .readings = &readings, .i = &i, .slept_ms = &slept }, settle_poll_ms, settle_max_ms, HostSideBack(Ledgers){ .host_before = 920_000_000 });
    try testing.expect(st.settled);
    try testing.expectEqual(@as(u64, 80_930_000_000), st.after.footprint);
    try testing.expect(!(HostSideBack(Ledgers){ .host_before = 920_000_000 }).ok(.{ .active = 80_000_000_000, .footprint = 80_970_000_001, .cache = 0 }));
    try testing.expect(!(HostSideBack(Ledgers){ .host_before = 920_000_000 }).ok(.{ .active = 80_000_000_000, .footprint = 80_930_000_000, .cache = 1 }));
}

test "phase memory: the construction check bills the construction terms at the prompt rows and bounds the measured host side" {
    const terms = [_]sdk.MemoryBill.Term{
        .{ .name = "residents", .bytes = .{ 100, 100 }, .at_construction = true },
        .{ .name = "host side", .bytes = .{ 30, 30 }, .at_construction = true, .measured = true, .construction = 20 },
    };
    const mb: sdk.MemoryBill = .{ .terms = &terms, .per_row = 10 };
    var c = Construction.of(mb, 3, 140, 100, 0);
    try testing.expectEqual(@as(u64, 150), c.billed);
    try testing.expectEqual(@as(i64, 10), c.residual());
    try c.checkFootprint(0);
    // A host side of 40 B, 15 B of it a term billed by construction: 25 B over the measured bound of 20.
    c.host_known = 15;
    try testing.expectEqual(@as(u64, 25), c.measuredHost());
    try testing.expectEqualStrings("host side", c.overMeasured(mb).?.name);
    c.host_known = 20;
    try testing.expect(c.overMeasured(mb) == null);
    try testing.expectError(error.ConstructionOverBill, Construction.of(mb, 3, 151, 100, 0).checkFootprint(0));
    // Another model in the process holds 60 B: the footprint passes the bill by them, and both bounds rise by them.
    var f = Construction.of(mb, 3, 200, 100, 0);
    f.host_known = 20;
    try testing.expectError(error.ConstructionOverBill, f.checkFootprint(0));
    f.foreign = 60;
    try f.checkFootprint(0);
    try testing.expectEqual(@as(i64, 10), f.residual());
    try testing.expect(f.overMeasured(mb) == null);
    // Fewer than the 50 B the footprint passes the bill by: refused, and the measured host side is over its bound.
    f.foreign = 49;
    try testing.expectError(error.ConstructionOverBill, f.checkFootprint(0));
    try testing.expectEqualStrings("host side", f.overMeasured(mb).?.name);
    // The reading the check takes: #8's, where the SDK publishes it.
    if (comptime @hasDecl(sdk.memory, "setForeignBytes")) {
        sdk.memory.setForeignBytes(60);
        defer sdk.memory.setForeignBytes(0);
        try testing.expectEqual(@as(u64, 60), foreignBytes());
    } else try testing.expectEqual(@as(u64, 0), foreignBytes());
}

test "phase memory: a record's residuals are the billed process less the footprint's high-water mark, the device terms less MLX's peak" {
    const T = struct { a: u64 = 0, b: u64 = 0 };
    const t: T = .{ .a = 7, .b = 5 };
    try testing.expectEqual(@as(u64, 12), sumFields(t));
    const pm: sdk.memory.ProcessMemory = .{ .footprint = 9, .footprint_interval_peak = 11 };
    const r = recordOf(T, "p", t, sumFields(t), 7, pm, 6, 2, 0, 100, 40, 3, 25);
    try testing.expectEqual(@as(i64, 1), r.residual_bytes);
    try testing.expectEqual(@as(i64, 1), r.mlx_residual_bytes);
    try testing.expectEqual(@as(u64, 6), r.mlx_peak_bytes);
    try testing.expectEqual(@as(i64, 15), r.file_cache_created_bytes);
    try expectMeasured(r, .{ .footprint = 9, .interval_peak = 11, .mlx_active = 6, .mlx_cache = 2, .mlx_peak = 0, .physical = 100, .file_backed = 40, .speculative = 3, .residual = 1, .mlx_residual = 1 }, 25);
    const json = try jsonOf(testing.allocator, r);
    defer testing.allocator.free(json);
    try testing.expect(std.mem.startsWith(u8, json, "{\"phase\":\"p\",\"billed\":{\"a\":7,\"b\":5},\"billed_process_bytes\":12,"));
    try testing.expectError(error.ConstructionLeftPageCache, checkPageCache(500_000_001));
    try checkPageCache(500_000_000);
    try checkPageCache(-300_000_000);
}
