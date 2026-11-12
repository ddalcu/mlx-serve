//! A built-in MoE model over a pack an outside quant plugin claims (`outside.zig`): the table open for the load
//! (`Load`), the projections and a resident routed layer's experts it stores, run on it, and the pack's streamed bank
//! opened at a budget (`openBank`) for each routed layer to bind (`moe.bindRouted`). The model keeps its forward.

const std = @import("std");
const plugin_abi = @import("../../abi/plugin.zig");
const quant_abi = @import("../../abi/quant.zig");
const mlx = @import("../../mlx/ffi.zig");
const system = @import("../util/system.zig");
const outside = @import("outside.zig");
const Linear = @import("../bases/blocks/linear.zig").Linear;
const moe = @import("../bases/blocks/moe.zig");

/// One model load's view of the table claiming its pack.
pub const Load = struct {
    a: std.mem.Allocator,
    /// The claiming table's load, open until `close`; null: no loaded table claims the pack.
    format: ?*outside.Format = null,
    /// Every weight the table loaded for the model, released by `deinit`.
    kept: std.ArrayList(outside.Weight) = .empty,

    /// The load through the table claiming the pack at `ctx` over `source`; `format` null when none claims it.
    pub fn open(a: std.mem.Allocator, ctx: *const plugin_abi.LoadCtx, source: quant_abi.Source, s: mlx.mlx_stream) !Load {
        return .{ .a = a, .format = try outside.Format.open(a, ctx.dir, ctx.model_dir.slice(), source, s, ctx.host) };
    }

    /// The projection `name` over `in_dim` inputs (0: any) as a `Linear` that runs on the table; null when no table
    /// is open or the pack stores it in no form the table reads.
    pub fn linear(self: *Load, name: []const u8, in_dim: u32) !?Linear {
        const x = (try self.weight(.tensor, name, in_dim)) orelse return null;
        return .{ .w = .{ .ctx = x.w.w.ctx }, .ext = x };
    }

    /// The routed experts `<base>.{gate,up,down}_proj` (`hidden` to `inter` and back) as the shared block's format
    /// banks; null as `linear`. A pack storing some of the three but not all is refused by name.
    pub fn experts(self: *Load, base: []const u8, hidden: u32, inter: u32) !?moe.FormatBanks {
        var nb: [256]u8 = undefined;
        const gate = (try self.weight(.bank, try std.fmt.bufPrint(&nb, "{s}.gate_proj", .{base}), hidden)) orelse return null;
        const up = (try self.weight(.bank, try std.fmt.bufPrint(&nb, "{s}.up_proj", .{base}), hidden)) orelse return error.MissingWeight;
        const down = (try self.weight(.bank, try std.fmt.bufPrint(&nb, "{s}.down_proj", .{base}), inter)) orelse return error.MissingWeight;
        return .{ .table = gate.t, .gate = gate.w, .up = up.w, .down = down.w };
    }

    fn weight(self: *Load, comptime how: enum { tensor, bank }, name: []const u8, in_dim: u32) !?outside.Weight {
        const f = self.format orelse return null;
        var x = (try (if (how == .tensor) f.tensor(name, in_dim) else f.bank(name, in_dim))) orelse return null;
        self.kept.append(self.a, x) catch |e| {
            x.deinit();
            return e;
        };
        return x;
    }

    /// Every weight loaded so far made resident.
    pub fn eval(self: *const Load) !void {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (self.kept.items) |*x| for (x.arrays()) |arr| if (arr.ctx != null) {
            _ = mlx.mlx_vector_array_append_value(vec, arr);
        };
        try mlx.check(mlx.mlx_eval(vec));
    }

    /// The load's end: the table closes; the weights stay the model's until `deinit`.
    pub fn close(self: *Load) void {
        if (self.format) |f| f.close();
        self.format = null;
    }

    pub fn deinit(self: *Load) void {
        self.close();
        for (self.kept.items) |*x| x.deinit();
        self.kept.deinit(self.a);
    }
};

/// The bytes a bank's slot rows may take: one budget, or the model's at a context it picks.
pub const Budget = union(enum) {
    fixed: quant_abi.BankBudget,
    rungs: Rungs,
};

/// The model's budget at `context` positions (`at`): at `ctx_size` when set, else at the first of `rungs` (largest
/// first) whose prompt rows reach `min_rows`, else at `fallback`.
pub const Rungs = struct {
    ctx: *const anyopaque,
    at: *const fn (ctx: *const anyopaque, context: u64) quant_abi.BankBudget,
    ctx_size: ?u64,
    rungs: []const u64,
    min_rows: u32,
    fallback: u64,
};

/// The host's room for a bank (`LoadCtx.stream_room`) past the memory in use now, in either phase.
pub fn roomBudget(room: u64) quant_abi.BankBudget {
    const free = room -| (system.getTotalMemBytes() -| system.getAvailableMemBytes());
    return .{ .prompt = free, .decode = free };
}

pub const Opened = struct {
    bank: *outside.Bank,
    rows: quant_abi.BankRows,
    /// The context the rows were budgeted for (`Rungs`); 0 for a fixed budget.
    context: u64 = 0,
};

/// The pack's streamed bank over `geo` at the rows `budget` buys, after the load's freed buffers go back to the system
/// (a budget reads the memory in use). `StreamedBankUnclaimed` when the table streams none.
pub fn openBank(f: *const outside.Format, geo: quant_abi.BankGeometry, budget: Budget, s: mlx.mlx_stream) !Opened {
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var context: u64 = 0;
    const at: quant_abi.BankBudget = switch (budget) {
        .fixed => |b| b,
        .rungs => |r| blk: {
            context = r.ctx_size orelse for (r.rungs) |rung| {
                const fit = (f.bankRows(geo, r.at(r.ctx, rung)) catch |e| if (e == error.StreamedRowsDoNotFit) continue else return e) orelse return error.StreamedBankUnclaimed;
                if (fit.prompt >= r.min_rows) break rung;
            } else r.fallback;
            break :blk r.at(r.ctx, context);
        },
    };
    const rows = (try f.bankRows(geo, at)) orelse return error.StreamedBankUnclaimed;
    const b = (try f.openBank(geo, at)) orelse return error.StreamedBankUnclaimed;
    return .{ .bank = b, .rows = rows, .context = context };
}

// ── tests ──

const testing = std.testing;
const common = @import("../../abi/common.zig");

const Toy = struct {
    var opened: u32 = 0;
    var closed: u32 = 0;
    var released: u32 = 0;
    var last_budget: quant_abi.BankBudget = .{ .prompt = 0, .decode = 0 };
    var arr: u8 = 0;

    fn claims(block: quant_abi.Str, _: quant_abi.Str) callconv(.c) common.Claim {
        return if (std.mem.indexOf(u8, block.slice(), "\"toy\"") != null) .by_marker else .no;
    }
    fn validate(_: quant_abi.Str, _: quant_abi.Str, _: *const common.PackDir) callconv(.c) common.Verdict {
        return .ok;
    }
    fn bill(_: quant_abi.Str, _: quant_abi.Str, stored: u64) callconv(.c) u64 {
        return stored;
    }
    fn open(_: *quant_abi.LoadCtx) callconv(.c) common.Verdict {
        return .ok;
    }
    fn close(_: *quant_abi.LoadCtx) callconv(.c) void {}
    /// Stores `q`, and the experts `e.gate_proj` / `e.up_proj` / `e.down_proj`, of `half.gate_proj` only the gate.
    fn load(_: *const quant_abi.LoadCtx, name: quant_abi.Str, in_dim: u32, out: *quant_abi.Weight) callconv(.c) common.Verdict {
        const n = name.slice();
        const stored = std.mem.eql(u8, n, "q") or std.mem.startsWith(u8, n, "e.") or std.mem.eql(u8, n, "half.gate_proj");
        if (!stored) return common.Verdict.refuse(.failed, "MissingWeight");
        out.* = .{ .w = .{ .ctx = &arr }, .block = in_dim };
        return .ok;
    }
    fn release(w: *quant_abi.Weight) callconv(.c) void {
        released += 1;
        w.* = .{};
    }
    fn matmul(_: *const quant_abi.Weight, _: quant_abi.Array, _: quant_abi.Stream, _: *quant_abi.Array) callconv(.c) common.Verdict {
        return common.Verdict.refuse(.unsupported, "toy");
    }
    fn moeRun(_: *const quant_abi.Weight, _: quant_abi.Array, _: quant_abi.Array, _: quant_abi.Array, _: bool, _: quant_abi.Stream, _: *quant_abi.Array) callconv(.c) common.Verdict {
        return common.Verdict.refuse(.unsupported, "toy");
    }
    /// One prompt row per 100 bytes; none below 100.
    fn bankRowsOf(_: *const quant_abi.LoadCtx, _: *const quant_abi.BankGeometry, budget: quant_abi.BankBudget, out: *quant_abi.BankRows) callconv(.c) common.Verdict {
        if (budget.prompt < 100) return common.Verdict.refuse(.too_large, "StreamedRowsDoNotFit");
        out.* = .{ .prompt = @intCast(budget.prompt / 100), .decode = @intCast(budget.decode / 100), .row_bytes = 100 };
        return .ok;
    }
    fn openBankOf(_: *const quant_abi.LoadCtx, _: *const quant_abi.BankGeometry, budget: quant_abi.BankBudget, out: *quant_abi.Bank) callconv(.c) common.Verdict {
        opened += 1;
        last_budget = budget;
        out.* = .{ .ctx = &arr, .close = &bankClose, .about = .of("toy"), .decode_ids = 48, .rows = &bankRows, .settle = &ok0, .phase = &phase, .read_ahead = &readAhead };
        return .ok;
    }
    fn bankClose(_: ?*anyopaque) callconv(.c) void {
        closed += 1;
    }
    fn bankRows(_: ?*anyopaque, _: u32, _: quant_abi.Array, _: quant_abi.Array, _: quant_abi.Array, _: *quant_abi.Array) callconv(.c) common.Verdict {
        return .ok;
    }
    fn ok0(_: ?*anyopaque) callconv(.c) common.Verdict {
        return .ok;
    }
    fn phase(_: ?*anyopaque, _: bool) callconv(.c) common.Verdict {
        return .ok;
    }
    fn readAhead(_: ?*anyopaque, _: u32, _: [*]const u16, _: usize) callconv(.c) common.Verdict {
        return .ok;
    }
    const table: quant_abi.QuantTable = .{ .name = .of("toy"), .claims = &claims, .validate = &validate, .bill = &bill, .open = &open, .close = &close, .load_tensor = &load, .load_bank = &load, .release = &release, .matmul = &matmul, .moe = &moeRun, .bank_rows = &bankRowsOf, .open_bank = &openBankOf };
};

fn toyLoad(t: *const quant_abi.QuantTable) !Load {
    const f = try testing.allocator.create(outside.Format);
    f.* = .{ .a = testing.allocator, .t = t, .config = try testing.allocator.alloc(u8, 0), .block = try testing.allocator.alloc(u8, 0), .source = undefined, .ctx = .{ .block_json = .{}, .source = undefined } };
    return .{ .a = testing.allocator, .format = f };
}

test "outside model: a stored projection is a Linear on the table, kept until deinit; one the pack lacks is null" {
    Toy.released = 0;
    var none: Load = .{ .a = testing.allocator };
    defer none.deinit();
    try testing.expect(try none.linear("q", 8) == null);
    var l = try toyLoad(&Toy.table);
    const q = (try l.linear("q", 8)).?;
    try testing.expect(q.w.ctx != null and q.s.ctx == null);
    try testing.expectEqual(@as(u32, 8), q.ext.?.w.block);
    try testing.expect(try l.linear("k", 8) == null);
    l.close();
    try testing.expect(l.format == null);
    // The weights outlive the table's close.
    try testing.expectEqual(@as(u32, 0), Toy.released);
    l.deinit();
    try testing.expectEqual(@as(u32, 1), Toy.released);
}

test "outside model: routed experts are the block's format banks at their input widths; a partial set is refused" {
    Toy.released = 0;
    var l = try toyLoad(&Toy.table);
    defer l.deinit();
    const fb = (try l.experts("e", 16, 4)).?;
    try testing.expect(fb.table == &Toy.table);
    try testing.expectEqual(@as(u32, 16), fb.up.block);
    try testing.expectEqual(@as(u32, 4), fb.down.block);
    try testing.expect(try l.experts("none", 16, 4) == null);
    try testing.expectError(error.MissingWeight, l.experts("half", 16, 4));
    try testing.expectEqual(@as(usize, 4), l.kept.items.len);
}

const ToyRungs = struct {
    /// The budget at `context` positions: 1000 bytes less one per 64 positions.
    fn at(_: *const anyopaque, context: u64) quant_abi.BankBudget {
        const b = 1000 -| context / 64;
        return .{ .prompt = b, .decode = b };
    }
};

test "outside model: a bank opens at a fixed budget, or at the largest rung keeping its fewest prompt rows" {
    var l = try toyLoad(&Toy.table);
    defer l.deinit();
    const geo: quant_abi.BankGeometry = .{ .hidden = 8, .inter = 4, .top_k = 2, .n_experts = 16, .n_layers = 2 };
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const fixed = try openBank(l.format.?, geo, .{ .fixed = .{ .prompt = 700, .decode = 900 } }, s);
    try testing.expectEqual(@as(u32, 7), fixed.rows.prompt);
    try testing.expectEqual(@as(u32, 9), fixed.rows.decode);
    try testing.expectEqual(@as(u64, 0), fixed.context);
    fixed.bank.close();
    var r: Rungs = .{ .ctx = undefined, .at = &ToyRungs.at, .ctx_size = null, .rungs = &.{ 57600, 32000, 6400 }, .min_rows = 4, .fallback = 640 };
    // 57600 leaves 100 bytes (1 row), 32000 leaves 500 (5 rows): the first to keep 4.
    const auto = try openBank(l.format.?, geo, .{ .rungs = r }, s);
    try testing.expectEqual(@as(u64, 32000), auto.context);
    try testing.expectEqual(@as(u32, 5), auto.rows.prompt);
    auto.bank.close();
    // No rung keeps 10 rows (6400 leaves 9): the fallback.
    r.min_rows = 10;
    const low = try openBank(l.format.?, geo, .{ .rungs = r }, s);
    try testing.expectEqual(@as(u64, 640), low.context);
    try testing.expectEqual(@as(u64, 990), Toy.last_budget.prompt);
    low.bank.close();
    // A model's own context wins.
    r.ctx_size = 64;
    const own = try openBank(l.format.?, geo, .{ .rungs = r }, s);
    try testing.expectEqual(@as(u64, 64), own.context);
    own.bank.close();
    try testing.expectError(error.StreamedRowsDoNotFit, openBank(l.format.?, geo, .{ .fixed = .{ .prompt = 10, .decode = 10 } }, s));
    var plain = Toy.table;
    plain.bank_rows = null;
    plain.open_bank = null;
    var p = try toyLoad(&plain);
    defer p.deinit();
    try testing.expectError(error.StreamedBankUnclaimed, openBank(p.format.?, geo, .{ .fixed = .{ .prompt = 700, .decode = 700 } }, s));
}

test "outside model: a routed layer binds to a source bank whatever its banks' layout; a model's own kernel refuses" {
    const ctx: u8 = 0;
    const Src = struct {
        fn sum(_: *anyopaque, _: u32, _: mlx.mlx_array, _: mlx.mlx_array, _: mlx.mlx_array) @import("../weights/routed.zig").Ret(mlx.mlx_array) {
            return .{};
        }
        fn settle(_: *anyopaque) @import("../weights/routed.zig").Ret(void) {
            return .{};
        }
    };
    const r: @import("../weights/routed.zig").Routed = .{ .ctx = @constCast(@ptrCast(&ctx)), .math = .{ .source = Src.sum }, .settle_fn = Src.settle };
    const none: mlx.mlx_array = .{ .ctx = null };
    const q: @import("params.zig").QuantParams = .{ .bits = 0, .group_size = 0, .mode = .mxfp4 };
    var w: moe.Moe.Weights = .{ .router = .{}, .route = .softmax, .gate = .{ .w = none, .qp = q }, .up = .{ .w = none, .qp = q }, .down = .{ .w = none, .qp = q }, .top_k = 2, .experts = 16 };
    try moe.bindRouted(&w, r, 5);
    try testing.expectEqual(@as(u32, 5), w.streamed.?.layer);
    var fused = w;
    fused.streamed = null;
    fused.fused = .{ .rows = 1, .run = undefined };
    try testing.expectError(error.StreamedExpertsUnsupported, moe.bindRouted(&fused, r, 5));
}
