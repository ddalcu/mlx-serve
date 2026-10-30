//! Gate 5: every registered launch against each chip's limits, from declarations and recorded compiles alone, so it
//! runs with no GPU in `zig build test`. Metal: threads and threadgroup bytes on M1..M5 (M1/M2 cap a pipeline's
//! threads by its register use). CUDA: threads, registers, shared bytes, launch bounds and clusters on sm_80..sm_121.

const std = @import("std");

/// An Apple GPU family we serve; on M1 and M2 a pipeline's thread limit falls as its registers rise.
pub const Apple = struct { name: []const u8, gen: u32, threads: u32 = 1024, tg_memory: u32 = 32768, by_registers: bool };

pub const apple = [_]Apple{
    .{ .name = "M1", .gen = 13, .by_registers = true },
    .{ .name = "M2", .gen = 14, .by_registers = true },
    .{ .name = "M3", .gen = 15, .by_registers = false },
    .{ .name = "M4", .gen = 16, .by_registers = false },
    .{ .name = "M5", .gen = 17, .by_registers = false },
};

/// A Metal pipeline as launched: most threads a group, threadgroup bytes, and the M1/M2 limits its compiles recorded.
pub const MetalLaunch = struct {
    name: []const u8,
    threads: u32,
    tg_memory: u32,
    /// M1 / M2 compiled thread limits, the least over its served instantiations (`offline.zig`); 0: none recorded.
    m1: u32 = 0,
    m2: u32 = 0,
    /// The launcher reads the pipeline's own thread limit (or probes it) and sizes or declines to it.
    adapts: bool = false,
    /// False for tensor-op kernels: Xcode's translator cannot build them for M1/M2 ahead of time.
    offline: bool = true,
    /// The least Apple family it is dispatched on (13 = M1; NAX kernels 17).
    min_gen: u32 = 13,
};

/// An NVIDIA target we build for: its per-block and per-SM limits (CUDA's compute capability tables).
pub const Sm = struct {
    sm: u32,
    threads: u32 = 1024,
    regs_thread: u32 = 255,
    regs_block: u32 = 65536,
    threads_sm: u32,
    smem_default: u32 = 48 * 1024,
    smem_optin: u32,
    clusters: bool,
};

pub const sms = [_]Sm{
    .{ .sm = 80, .threads_sm = 2048, .smem_optin = 163 * 1024, .clusters = false },
    .{ .sm = 86, .threads_sm = 1536, .smem_optin = 99 * 1024, .clusters = false },
    .{ .sm = 89, .threads_sm = 1536, .smem_optin = 99 * 1024, .clusters = false },
    .{ .sm = 120, .threads_sm = 1536, .smem_optin = 99 * 1024, .clusters = true },
    .{ .sm = 121, .threads_sm = 1536, .smem_optin = 99 * 1024, .clusters = true },
};

/// A CUDA kernel as launched: most threads a block, dynamic shared bytes, cluster width, raised limit, least SM.
pub const CudaLaunch = struct {
    name: []const u8,
    threads: u32,
    dynamic: u32 = 0,
    cluster: u32 = 1,
    opt_in: bool = false,
    min_sm: u32 = 0,
};

/// What the compiler made of one kernel for one SM (registers, static shared bytes, __launch_bounds__ threads).
pub const Usage = struct { regs: u32, static: u32, bound: u32 = 0 };

/// Every problem one Metal launch has on one family, written to `out`; returns how many.
pub fn metal(l: MetalLaunch, f: Apple, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    if (f.gen < l.min_gen) return 0;
    var n: usize = 0;
    if (l.threads > f.threads) n += try note(out, a, "{s} on {s}: {d} threads a group, the family's most is {d}", .{ l.name, f.name, l.threads, f.threads });
    if (l.tg_memory > f.tg_memory) n += try note(out, a, "{s} on {s}: {d} threadgroup bytes, the family's most is {d}", .{ l.name, f.name, l.tg_memory, f.tg_memory });
    if (f.by_registers and !l.adapts) {
        const lim = if (f.gen == 13) l.m1 else l.m2;
        if (!l.offline and l.threads > 256) n += try note(out, a, "{s} on {s}: {d} threads past 256 and no M1/M2 compile to show they fit", .{ l.name, f.name, l.threads });
        if (l.offline and lim == 0) n += try note(out, a, "{s} on {s}: no compiled thread limit recorded", .{ l.name, f.name });
        if (lim != 0 and lim < l.threads) n += try note(out, a, "{s} on {s}: {d} threads, the pipeline compiles for {d}", .{ l.name, f.name, l.threads, lim });
    }
    return n;
}

/// Registers a block takes on one SM: each warp's registers rounded up to 256, times its warps.
pub fn blockRegs(regs: u32, threads: u32) u32 {
    const warp = (regs * 32 + 255) / 256 * 256;
    return warp * ((threads + 31) / 32);
}

/// Every problem one CUDA launch has on one SM given what the compiler made of it; returns how many.
pub fn cuda(l: CudaLaunch, u: Usage, t: Sm, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    const name = l.name;
    if (t.sm < l.min_sm) return 0;
    var n: usize = 0;
    if (l.threads > t.threads) n += try note(out, a, "{s} on sm_{d}: {d} threads a block, the most is {d}", .{ name, t.sm, l.threads, t.threads });
    if (u.bound != 0 and l.threads > u.bound) n += try note(out, a, "{s} on sm_{d}: launched at {d} threads past its __launch_bounds__ {d}", .{ name, t.sm, l.threads, u.bound });
    if (u.regs > t.regs_thread) n += try note(out, a, "{s} on sm_{d}: {d} registers a thread", .{ name, t.sm, u.regs });
    if (blockRegs(u.regs, l.threads) > t.regs_block) n += try note(out, a, "{s} on sm_{d}: {d} registers x {d} threads need {d}, a block has {d}", .{ name, t.sm, u.regs, l.threads, blockRegs(u.regs, l.threads), t.regs_block });
    const smem = u.static + l.dynamic;
    if (smem > t.smem_optin) n += try note(out, a, "{s} on sm_{d}: {d} shared bytes, the most a block takes is {d}", .{ name, t.sm, smem, t.smem_optin });
    if (smem > t.smem_default and !l.opt_in) n += try note(out, a, "{s} on sm_{d}: {d} shared bytes past 48 KiB without raising the limit", .{ name, t.sm, smem });
    if (l.cluster > 1 and !t.clusters) n += try note(out, a, "{s} on sm_{d}: a cluster of {d} where the SM has no clusters", .{ name, t.sm, l.cluster });
    return n;
}

/// The table's row for `sm` (a target the records cover).
pub fn target(sm: u32) ?Sm {
    for (sms) |t| if (t.sm == sm) return t;
    return null;
}

/// Every problem of every Metal launch on every family, written to `out`; returns how many.
pub fn metalAll(list: []const MetalLaunch, out: *std.ArrayList(u8), a: std.mem.Allocator) !usize {
    var n: usize = 0;
    for (list) |l| for (apple) |f| {
        n += try metal(l, f, out, a);
    };
    return n;
}

fn note(out: *std.ArrayList(u8), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype) !usize {
    try out.print(a, fmt ++ "\n", args);
    return 1;
}

// ── tests ──

test "kernel contract: a launch inside every family passes; each planted excess is named" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    const ok = MetalLaunch{ .name = "ok", .threads = 1024, .tg_memory = 256, .m1 = 1024, .m2 = 1024 };
    for (apple) |f| try std.testing.expectEqual(@as(usize, 0), try metal(ok, f, &out, a));
    const big = MetalLaunch{ .name = "big", .threads = 256, .tg_memory = 36 * 1024, .m1 = 1024, .m2 = 1024 };
    for (apple) |f| try std.testing.expectEqual(@as(usize, 1), try metal(big, f, &out, a));
    const heavy = MetalLaunch{ .name = "heavy", .threads = 1024, .tg_memory = 0, .m1 = 448, .m2 = 640 };
    try std.testing.expectEqual(@as(usize, 1), try metal(heavy, apple[0], &out, a));
    try std.testing.expectEqual(@as(usize, 0), try metal(heavy, apple[2], &out, a));
    var probing = heavy;
    probing.adapts = true;
    try std.testing.expectEqual(@as(usize, 0), try metal(probing, apple[0], &out, a));
    const unknown = MetalLaunch{ .name = "unknown", .threads = 64, .tg_memory = 0 };
    try std.testing.expectEqual(@as(usize, 1), try metal(unknown, apple[1], &out, a));
    const tensor_op = MetalLaunch{ .name = "tensor op", .threads = 128, .tg_memory = 0, .offline = false };
    try std.testing.expectEqual(@as(usize, 0), try metal(tensor_op, apple[0], &out, a));
    var wide_op = tensor_op;
    wide_op.threads = 512;
    try std.testing.expectEqual(@as(usize, 1), try metal(wide_op, apple[0], &out, a));
    const nax = MetalLaunch{ .name = "nax", .threads = 1024, .tg_memory = 0, .offline = false, .min_gen = 17 };
    for (apple) |f| try std.testing.expectEqual(@as(usize, 0), try metal(nax, f, &out, a));
}

test "kernel contract: CUDA: registers by warp, launch bounds, shared bytes and clusters per SM" {
    const a = std.testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(a);
    try std.testing.expectEqual(@as(u32, 65536), blockRegs(64, 1024));
    try std.testing.expectEqual(@as(u32, 512 * 4), blockRegs(13, 128)); // 13 registers round up to 16 a thread
    const l = CudaLaunch{ .name = "k", .threads = 256, .dynamic = 40 * 1024 };
    const sm80 = target(80).?;
    const sm86 = target(86).?;
    try std.testing.expectEqual(@as(usize, 0), try cuda(l, .{ .regs = 64, .static = 0, .bound = 256 }, sm86, &out, a));
    try std.testing.expectEqual(@as(usize, 1), try cuda(l, .{ .regs = 64, .static = 16 * 1024, .bound = 256 }, sm86, &out, a));
    var big = l;
    big.dynamic = 120 * 1024;
    big.opt_in = true;
    try std.testing.expectEqual(@as(usize, 0), try cuda(big, .{ .regs = 64, .static = 0, .bound = 256 }, sm80, &out, a));
    try std.testing.expectEqual(@as(usize, 1), try cuda(big, .{ .regs = 64, .static = 0, .bound = 256 }, sm86, &out, a));
    var wide = l;
    wide.threads = 1024;
    try std.testing.expectEqual(@as(usize, 2), try cuda(wide, .{ .regs = 72, .static = 0, .bound = 256 }, target(121).?, &out, a));
    var cl = l;
    cl.cluster = 4;
    try std.testing.expectEqual(@as(usize, 1), try cuda(cl, .{ .regs = 32, .static = 0 }, target(89).?, &out, a));
    try std.testing.expectEqual(@as(usize, 0), try cuda(cl, .{ .regs = 32, .static = 0 }, target(120).?, &out, a));
    cl.min_sm = 90;
    try std.testing.expectEqual(@as(usize, 0), try cuda(cl, .{ .regs = 32, .static = 0 }, sm80, &out, a));
    try std.testing.expectEqual(@as(?Sm, null), target(75));
}
