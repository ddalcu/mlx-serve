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

/// The `[out, in]` rows a projection the table loaded computes (`in_dim` its inputs): its matmul over the identity,
/// `chunk` input columns a call (identity row i reads column i of W), in `dtype`. For a model that needs a weight's
/// rows (a requantized copy) where the pack stores none MLX reads; a load-time transient of out x in.
pub fn denseRows(w: *const outside.Weight, in_dim: u32, dtype: mlx.mlx_dtype, chunk: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const n: c_int = @intCast(in_dim);
    const step: c_int = @intCast(@max(chunk, 1));
    var cols = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cols);
    try mlx.check(mlx.mlx_arange(&cols, 0, @floatFromInt(n), 1, .int32, s));
    const parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    var c0: c_int = 0;
    while (c0 < n) : (c0 += step) {
        const c = @min(step, n - c0);
        var idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idx);
        try mlx.check(mlx.mlx_arange(&idx, @floatFromInt(c0), @floatFromInt(c0 + c), 1, .int32, s));
        var col = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(col);
        try mlx.check(mlx.mlx_reshape(&col, idx, &[_]c_int{ c, 1 }, 2, s));
        var hit = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hit);
        try mlx.check(mlx.mlx_equal(&hit, col, cols, s));
        var eye = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eye);
        try mlx.check(mlx.mlx_astype(&eye, hit, dtype, s));
        // One chunk of rows, as a prompt reaches the projection: [1, c, in].
        var eye3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eye3);
        try mlx.check(mlx.mlx_reshape(&eye3, eye, &[_]c_int{ 1, c, n }, 3, s));
        const y = try w.matmul(eye3, s);
        defer _ = mlx.mlx_array_free(y);
        const out_dim = mlx.getShape(y)[mlx.getShape(y).len - 1];
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_reshape(&part, y, &[_]c_int{ c, out_dim }, 2, s));
        try mlx.check(mlx.mlx_array_eval(part));
        _ = mlx.mlx_vector_array_append_value(parts, part);
    }
    var cols_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cols_t);
    try mlx.check(mlx.mlx_concatenate_axis(&cols_t, parts, 0, s));
    var rows = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_transpose(&rows, cols_t, s));
    return rows;
}

/// The bytes a bank's slot rows may take: one budget, the model's at a context it picks, or each phase's own (`Fit`).
pub const Budget = union(enum) {
    fixed: quant_abi.BankBudget,
    rungs: Rungs,
    fit: Fit,
};

/// Each phase's budget from its own room (`LoadCtx.stream_room`, `LoadCtx.decodeRoom()`), read past the memory in use
/// once the load's freed buffers are back (`fitBudget`); with a calibration (`bank-fit.json`) from measured peaks.
pub const Fit = struct {
    prompt_room: u64,
    decode_room: u64,
    /// Growth of the box's memory outside the server between the load and the peak (a calibration's allowance).
    inuse_margin: u64 = 1 << 30,
    /// The bills' growth from the calibration's evidence (`Calibration.evidence`) to the server's context limit
    /// (`LoadCtx.streamGrowth`).
    grow_prompt: u64 = 0,
    grow_decode: u64 = 0,
    calibration: ?Calibration = null,
    /// The memory read at the load's start, before the trunk (`readMemory`): what was outside the server then settles
    /// the open's reading (`settled`).
    load_start: ?Reading = null,
};

/// A pack's measured fit on one ceiling (`bank-fit.json`): each phase's base is the evidence run's peak of the box's
/// non-file memory less the bank's bytes there, at the rooms' cell (a 16K + 1K request); `rest` the box's non-file memory
/// outside the server at that run's bank open (`Reading`). `evidence` is the tokens that run's request held (prompt +
/// reply): its bases already hold their KV, so the host's growth starts there (the cell when the file names none).
pub const Calibration = struct {
    ceiling: u64,
    cell: u64,
    evidence: u64,
    base_prompt: u64,
    base_decode: u64,
    rest: u64,
};

/// The memory a budget reads: the box's non-file memory in use (`system.getNonFileMemBytes`: a page cache of another
/// size reads the same) and the server's own footprint within it.
pub const Reading = struct { in_use: u64, footprint: u64 };

/// `fit`'s budget at `r`. Bills only: each phase's room less the memory in use (equal rooms: `roomBudget`). Calibrated:
/// each phase's bytes are the ceiling less its base and its bills' growth to the context limit, the evidence's rest
/// given back and what is outside the server now (plus the margin) taken. Every term saturates at 0.
pub fn fitBudget(fit: Fit, r: Reading) quant_abi.BankBudget {
    const c = fit.calibration orelse return .{ .prompt = fit.prompt_room -| r.in_use, .decode = fit.decode_room -| r.in_use };
    const allow = (r.in_use -| r.footprint) +| fit.inuse_margin;
    return .{
        .prompt = (c.ceiling +| c.rest) -| c.base_prompt -| fit.grow_prompt -| allow,
        .decode = (c.ceiling +| c.rest) -| c.base_decode -| fit.grow_decode -| allow,
    };
}

/// The open's reading `now`, settled: macOS's page counts show the server's new buffers a second or two late while its
/// footprint shows them at once, so the memory in use is at least what was outside the server at `start` plus the
/// server now.
pub fn settled(now: Reading, start: Reading) Reading {
    return .{ .in_use = @max(now.in_use, (start.in_use -| start.footprint) +| now.footprint), .footprint = now.footprint };
}

pub const OpenBudget = struct {
    budget: quant_abi.BankBudget,
    /// The reading the budget came from (settled when the fit has a `load_start`).
    reading: Reading,
    refused: ?error{BankReadingBelowFootprint} = null,
};

/// `fit`'s budget at the open's reading `now`. A reading still below the server's own footprint is refused by name: the
/// bills stand on `fallback_in_use` (`total - available`), never on less than the footprint.
pub fn openBudget(fit: Fit, now: Reading, fallback_in_use: u64) OpenBudget {
    const r = if (fit.load_start) |start| settled(now, start) else now;
    if (r.in_use >= r.footprint) return .{ .budget = fitBudget(fit, r), .reading = r };
    const in_use = @max(fallback_in_use, r.footprint);
    return .{ .budget = .{ .prompt = fit.prompt_room -| in_use, .decode = fit.decode_room -| in_use }, .reading = r, .refused = error.BankReadingBelowFootprint };
}

/// The calibration file's name in a pack and its format.
pub const bank_fit_file = "bank-fit.json";
pub const bank_fit_format = "mlx-serve-bank-fit-v2";
/// The first format: its `rest_bytes` read `total - available`, which moves with the page cache; refused by name.
pub const bank_fit_format_v1 = "mlx-serve-bank-fit-v1";

/// `bank-fit.json`'s calibration when its ceiling is the host's (`ceiling`, 0 = unknown: none applies) and its evidence
/// ran the rooms' cell (`cell`, the bills' growth past it is the host's); a named reason otherwise (logged by the caller;
/// the bills stand).
pub fn parseBankFit(json: []const u8, ceiling: u64, cell: u64) error{ BankFitInvalid, BankFitFormat, BankFitV1Rest, BankFitOtherCeiling, BankFitOtherCell }!Calibration {
    const F = struct { format: []const u8 = "", ceiling_bytes: u64 = 0, cell_tokens: u64 = 0, evidence_tokens: u64 = 0, base_prompt_bytes: u64 = 0, base_decode_bytes: u64 = 0, rest_bytes: u64 = 0 };
    var buf: [4096]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);
    const f = std.json.parseFromSliceLeaky(F, fba.allocator(), json, .{ .ignore_unknown_fields = true }) catch return error.BankFitInvalid;
    if (std.mem.eql(u8, f.format, bank_fit_format_v1)) return error.BankFitV1Rest;
    if (!std.mem.eql(u8, f.format, bank_fit_format)) return error.BankFitFormat;
    if (f.ceiling_bytes == 0 or f.cell_tokens == 0 or f.base_prompt_bytes == 0 or f.base_decode_bytes == 0) return error.BankFitInvalid;
    if (ceiling == 0 or f.ceiling_bytes != ceiling) return error.BankFitOtherCeiling;
    if (f.cell_tokens != cell) return error.BankFitOtherCell;
    return .{ .ceiling = f.ceiling_bytes, .cell = f.cell_tokens, .evidence = if (f.evidence_tokens > 0) f.evidence_tokens else f.cell_tokens, .base_prompt = f.base_prompt_bytes, .base_decode = f.base_decode_bytes, .rest = f.rest_bytes };
}

/// The memory in use now and the server's footprint (`fitBudget`'s reading).
pub fn readMemory() Reading {
    return .{ .in_use = system.getNonFileMemBytes(), .footprint = @as(u64, system.getAppMemFootprintMb()) << 20 };
}

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
    /// The budget the rows were planned from, and for a `fit` the memory read at the open (a calibration's evidence:
    /// `rest = in_use - footprint`).
    budget: quant_abi.BankBudget = .{ .prompt = 0, .decode = 0 },
    reading: ?Reading = null,
    /// A `fit`'s reading as read, before `settled`.
    read: ?Reading = null,
    refused: ?error{BankReadingBelowFootprint} = null,
};

/// The pack's streamed bank over `geo` at the rows `budget` buys, after the load's freed buffers go back to the system
/// (a budget reads the memory in use). `StreamedBankUnclaimed` when the table streams none.
pub fn openBank(f: *const outside.Format, geo: quant_abi.BankGeometry, budget: Budget, s: mlx.mlx_stream) !Opened {
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var context: u64 = 0;
    var reading: ?Reading = null;
    var read: ?Reading = null;
    var refused: ?error{BankReadingBelowFootprint} = null;
    const at: quant_abi.BankBudget = switch (budget) {
        .fixed => |b| b,
        .rungs => |r| blk: {
            context = r.ctx_size orelse for (r.rungs) |rung| {
                const fit = (f.bankRows(geo, r.at(r.ctx, rung)) catch |e| if (e == error.StreamedRowsDoNotFit) continue else return e) orelse return error.StreamedBankUnclaimed;
                if (fit.prompt >= r.min_rows) break rung;
            } else r.fallback;
            break :blk r.at(r.ctx, context);
        },
        .fit => |fit| blk: {
            read = readMemory();
            const o = openBudget(fit, read.?, system.getTotalMemBytes() -| system.getAvailableMemBytes());
            reading = o.reading;
            refused = o.refused;
            break :blk o.budget;
        },
    };
    const rows = (try f.bankRows(geo, at)) orelse return error.StreamedBankUnclaimed;
    const b = (try f.openBank(geo, at)) orelse return error.StreamedBankUnclaimed;
    return .{ .bank = b, .rows = rows, .context = context, .budget = at, .reading = reading, .read = read, .refused = refused };
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

/// `x @ W^T` over the weight's own array W.
const DenseToy = struct {
    fn matmul(w: *const quant_abi.Weight, x: quant_abi.Array, s: quant_abi.Stream, out: *quant_abi.Array) callconv(.c) common.Verdict {
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        var y = mlx.mlx_array_new();
        if (mlx.mlx_transpose(&wt, .{ .ctx = w.w.ctx }, .{ .ctx = s.ctx }) != 0 or mlx.mlx_matmul(&y, .{ .ctx = x.ctx }, wt, .{ .ctx = s.ctx }) != 0) return common.Verdict.refuse(.failed, "DenseToy");
        out.* = .{ .ctx = y.ctx };
        return .ok;
    }
    const table: quant_abi.QuantTable = blk: {
        var t = Toy.table;
        t.matmul = &matmul;
        break :blk t;
    };
};

test "outside model: a stored projection's rows are its matmul over the identity, a chunk of columns at a time" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_arange(&flat, 0, 15, 1, .float32, s));
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    try mlx.check(mlx.mlx_reshape(&w, flat, &[_]c_int{ 3, 5 }, 2, s));
    const weight: outside.Weight = .{ .t = &DenseToy.table, .w = .{ .w = .{ .ctx = w.ctx } } };
    const rows = try denseRows(&weight, 5, .float32, 2, s);
    defer _ = mlx.mlx_array_free(rows);
    var got = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_contiguous(&got, rows, false, s));
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqualSlices(c_int, &.{ 3, 5 }, mlx.getShape(got));
    for (mlx.mlx_array_data_float32(got).?[0..15], 0..) |v, i| try testing.expectEqual(@as(f32, @floatFromInt(i)), v);
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
    // The budget the rows came from; no memory reading but a fit's.
    try testing.expectEqual(quant_abi.BankBudget{ .prompt = 700, .decode = 900 }, fixed.budget);
    try testing.expect(fixed.reading == null);
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

test "outside model: a fit's bills-only budget is each phase's room past the memory in use; equal rooms are roomBudget's" {
    const r: Reading = .{ .in_use = 30 << 30, .footprint = 20 << 30 };
    try testing.expectEqual(quant_abi.BankBudget{ .prompt = 70 << 30, .decode = 70 << 30 }, fitBudget(.{ .prompt_room = 100 << 30, .decode_room = 100 << 30 }, r));
    // DECODE_ROOM: decode's own room (no prompt scratch) buys it more.
    try testing.expectEqual(quant_abi.BankBudget{ .prompt = 70 << 30, .decode = 82 << 30 }, fitBudget(.{ .prompt_room = 100 << 30, .decode_room = 112 << 30 }, r));
    try testing.expectEqual(quant_abi.BankBudget{ .prompt = 0, .decode = 0 }, fitBudget(.{ .prompt_room = 1 << 30, .decode_room = 1 << 30 }, r));
}

test "outside model: a calibrated fit gives each phase the ceiling less its base and growth, the evidence's rest back and what is outside the server now" {
    const cal: Calibration = .{ .ceiling = 110 << 30, .cell = 17_408, .evidence = 17_408, .base_prompt = 40 << 30, .base_decode = 28 << 30, .rest = 14 << 30 };
    // 9 GiB outside the server now, a 1 GiB margin: 110 + 14 - 40 - 10 = 74 GiB, and 86 GiB for decode.
    const r: Reading = .{ .in_use = 60 << 30, .footprint = 51 << 30 };
    try testing.expectEqual(quant_abi.BankBudget{ .prompt = 74 << 30, .decode = 86 << 30 }, fitBudget(.{ .prompt_room = 1, .decode_room = 1, .calibration = cal }, r));
    // A context past the cell: each phase gives its bills' growth up.
    try testing.expectEqual(quant_abi.BankBudget{ .prompt = 71 << 30, .decode = 85 << 30 }, fitBudget(.{ .prompt_room = 1, .decode_room = 1, .grow_prompt = 3 << 30, .grow_decode = 1 << 30, .calibration = cal }, r));
    // More outside the server than the evidence had: fewer bytes, never below 0.
    try testing.expectEqual(quant_abi.BankBudget{ .prompt = 0, .decode = 0 }, fitBudget(.{ .prompt_room = 1, .decode_room = 1, .calibration = cal }, .{ .in_use = 200 << 30, .footprint = 0 }));
}

test "outside model: an open reading below the server's footprint settles from the load's start, else is refused, never the whole room" {
    const MiB: u64 = 1 << 20;
    // The page counts have not yet shown the server's new buffers: in use reads below its footprint.
    const lag: Reading = .{ .in_use = 10_703 * MiB, .footprint = 23_582 * MiB };
    const start: Reading = .{ .in_use = 13_000 * MiB, .footprint = 1_500 * MiB };
    const bills: Fit = .{ .prompt_room = 100 << 30, .decode_room = 110 << 30, .load_start = start };
    const o = openBudget(bills, lag, 0);
    try testing.expect(o.refused == null);
    try testing.expectEqual(Reading{ .in_use = 11_500 * MiB + 23_582 * MiB, .footprint = 23_582 * MiB }, o.reading);
    try testing.expectEqual(quant_abi.BankBudget{ .prompt = (100 << 30) - 35_082 * MiB, .decode = (110 << 30) - 35_082 * MiB }, o.budget);
    // A reading past the start's figure stands as read.
    const live: Reading = .{ .in_use = 40_000 * MiB, .footprint = 23_582 * MiB };
    try testing.expectEqual(live, openBudget(bills, live, 0).reading);
    // Calibrated: what is outside the server is the start's 11,500 MiB, not 0.
    const cal: Calibration = .{ .ceiling = 110 << 30, .cell = 17_408, .evidence = 17_408, .base_prompt = 40 << 30, .base_decode = 28 << 30, .rest = 11_500 * MiB };
    var calibrated = bills;
    calibrated.calibration = cal;
    try testing.expectEqual(quant_abi.BankBudget{ .prompt = (70 << 30) - calibrated.inuse_margin, .decode = (82 << 30) - calibrated.inuse_margin }, openBudget(calibrated, lag, 0).budget);
    // No start reading: refused by name; the bills on total - available, or the footprint when that reads lower.
    for ([_]Fit{ bills, calibrated }) |f0| {
        var f = f0;
        f.load_start = null;
        const r = openBudget(f, lag, 36_000 * MiB);
        try testing.expectEqual(error.BankReadingBelowFootprint, r.refused.?);
        try testing.expectEqual(quant_abi.BankBudget{ .prompt = (100 << 30) - 36_000 * MiB, .decode = (110 << 30) - 36_000 * MiB }, r.budget);
        try testing.expectEqual(quant_abi.BankBudget{ .prompt = (100 << 30) - 23_582 * MiB, .decode = (110 << 30) - 23_582 * MiB }, openBudget(f, lag, 9_000 * MiB).budget);
    }
}

test "outside model: a calibrated budget reads the same with more file pages on the box (the non-file reading)" {
    if (!@import("builtin").os.tag.isDarwin()) return error.SkipZigTest;
    const GB: u64 = 1 << 30;
    const cal: Calibration = .{ .ceiling = 110 * GB, .cell = 17_408, .evidence = 17_408, .base_prompt = 40 * GB, .base_decode = 28 * GB, .rest = 12 * GB };
    const fit: Fit = .{ .prompt_room = 1, .decode_room = 1, .calibration = cal };
    const foot: u64 = 19 * GB;
    defer system.reading_override = null;
    var budgets: [2]quant_abi.BankBudget = undefined;
    // the same wired, anonymous, compressor and purgeable pages; the second box holds 7 GB more file pages
    for (&budgets, [_]u64{ 4, 11 }) |*b, file_gb| {
        system.reading_override = system.plantVm(128 * GB, 16384, .{ .wired = 25 * GB, .anonymous = 6 * GB, .compressor = GB, .purgeable = GB / 2, .file = file_gb * GB });
        b.* = fitBudget(fit, .{ .in_use = system.getNonFileMemBytes(), .footprint = foot });
    }
    try testing.expectEqual(budgets[0], budgets[1]);
    try testing.expectEqual(quant_abi.BankBudget{ .prompt = 110 * GB + 12 * GB - 40 * GB - (31 * GB + GB / 2 - foot) - fit.inuse_margin, .decode = 110 * GB + 12 * GB - 28 * GB - (31 * GB + GB / 2 - foot) - fit.inuse_margin }, budgets[0]);
}

test "outside model: bank-fit.json applies on its own ceiling and cell; another format, ceiling, cell or a bad file is refused by name" {
    const good = "{\"format\":\"mlx-serve-bank-fit-v2\",\"ceiling_bytes\":118111600640,\"cell_tokens\":17408,\"base_prompt_bytes\":40000000000,\"base_decode_bytes\":28000000000,\"rest_bytes\":15000000000,\"evidence\":\"g53cpab4\"}";
    const c = try parseBankFit(good, 118111600640, 17_408);
    try testing.expectEqual(@as(u64, 40000000000), c.base_prompt);
    try testing.expectEqual(@as(u64, 15000000000), c.rest);
    // No evidence_tokens: the growth starts at the cell; named: at the evidence request's prompt + reply.
    try testing.expectEqual(@as(u64, 17_408), c.evidence);
    const named = "{\"format\":\"mlx-serve-bank-fit-v2\",\"ceiling_bytes\":118111600640,\"cell_tokens\":17408,\"evidence_tokens\":18380,\"base_prompt_bytes\":40000000000,\"base_decode_bytes\":28000000000,\"rest_bytes\":15000000000}";
    try testing.expectEqual(@as(u64, 18_380), (try parseBankFit(named, 118111600640, 17_408)).evidence);
    try testing.expectError(error.BankFitOtherCeiling, parseBankFit(good, 112 << 30, 17_408));
    try testing.expectError(error.BankFitOtherCeiling, parseBankFit(good, 0, 17_408));
    try testing.expectError(error.BankFitOtherCell, parseBankFit(good, 118111600640, 33_792));
    try testing.expectError(error.BankFitFormat, parseBankFit("{\"format\":\"other\"}", 118111600640, 17_408));
    // v1's rest read total - available: refused by name, the same file otherwise
    const v1 = "{\"format\":\"mlx-serve-bank-fit-v1\",\"ceiling_bytes\":118111600640,\"cell_tokens\":17408,\"base_prompt_bytes\":40000000000,\"base_decode_bytes\":28000000000,\"rest_bytes\":15000000000}";
    try testing.expectError(error.BankFitV1Rest, parseBankFit(v1, 118111600640, 17_408));
    try testing.expectError(error.BankFitInvalid, parseBankFit("{\"format\":\"mlx-serve-bank-fit-v2\",\"ceiling_bytes\":1}", 1, 1));
    try testing.expectError(error.BankFitInvalid, parseBankFit("{", 1, 1));
}
