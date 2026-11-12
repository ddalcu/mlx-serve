//! Outside quant plugins as an in-tree model reaches them: the tables the host loaded (`register`), the one that claims
//! a pack's quantization block (`Format.open`), the weights it loads, and their matmul and SwiGLU experts, which the
//! table runs. The model keeps its own forward; only the math over these weights is the plugin's.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const common = @import("../../abi/common.zig");
const Host = @import("../../abi/host.zig").Host;
const mlx = @import("../../mlx/ffi.zig");
const log = @import("../util/log.zig");
const routed_mod = @import("../weights/routed.zig");

const max_tables = 16;
const Tables = struct { items: [max_tables]*const abi.QuantTable = undefined, n: usize = 0 };
/// Set once at startup by the host's plugin loader, before any load; read-only after.
pub const tables = @import("../util/shared.zig").Shared(Tables, "outside_quant_tables", .{});

pub fn register(t: *const abi.QuantTable) void {
    const s = tables.get();
    if (s.n == max_tables) return log.warn("[plugins] {s}: more than {d} quant plugins, not served\n", .{ t.name.slice(), max_tables });
    s.items[s.n] = t;
    s.n += 1;
}

/// The loaded table that claims `block_json` of `config_json`: the highest claim; null when none claims or two tie.
pub fn claimant(block_json: []const u8, config_json: []const u8) ?*const abi.QuantTable {
    const s = tables.get();
    var claims: [max_tables]common.Claim = undefined;
    for (s.items[0..s.n], claims[0..s.n]) |t, *c| c.* = t.claims(.of(block_json), .of(config_json));
    const i = (common.pick(claims[0..s.n]) catch return null) orelse return null;
    return s.items[i];
}

/// The loaded table claiming `config_json`'s `quantization_config` or `quantization` block, with that block's JSON (in
/// `a`); null when none claims either.
fn claimed(a: std.mem.Allocator, config_json: []const u8) !?struct { t: *const abi.QuantTable, block: []u8 } {
    if (tables.get().n == 0) return null;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), config_json, .{}) catch return null;
    if (root != .object) return null;
    for ([_][]const u8{ "quantization_config", "quantization" }) |key| {
        const block = try std.json.Stringify.valueAlloc(a, root.object.get(key) orelse continue, .{});
        if (claimant(block, config_json)) |t| return .{ .t = t, .block = block };
        a.free(block);
    }
    return null;
}

/// Whether a loaded table claims the pack at `model_dir`: its tensors then load as stored, in the dtypes the plugin
/// reads.
pub fn claimsPack(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) bool {
    if (tables.get().n == 0) return false;
    var dir = std.Io.Dir.cwd().openDir(io, model_dir, .{}) catch return false;
    defer dir.close(io);
    const config = dir.readFileAlloc(io, "config.json", a, .limited(4 << 20)) catch return false;
    defer a.free(config);
    const c = (claimed(a, config) catch return false) orelse return false;
    a.free(c.block);
    return true;
}

/// A weight an outside table loaded; it owns its arrays until `deinit`.
pub const Weight = struct {
    t: *const abi.QuantTable,
    w: abi.Weight,

    /// `x @ W^T`, a new array.
    pub fn matmul(self: *const Weight, x: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
        var out: abi.Array = .{};
        try check(self.t, self.t.matmul(&self.w, .{ .ctx = x.ctx }, .{ .ctx = s.ctx }, &out));
        return .{ .ctx = out.ctx };
    }

    /// Rows one `matmul` takes with each row given its one-row call's bytes (`QuantTable.exact_rows`, at least 1).
    pub fn exactRows(self: *const Weight) u32 {
        return @max(1, self.t.exactRows());
    }

    /// Its arrays, for a model that evaluates every weight it reads at load.
    pub fn arrays(self: *const Weight) [4]mlx.mlx_array {
        return .{ .{ .ctx = self.w.w.ctx }, .{ .ctx = self.w.scales.ctx }, .{ .ctx = self.w.biases.ctx }, .{ .ctx = self.w.side.ctx } };
    }

    pub fn deinit(self: *Weight) void {
        self.t.release(&self.w);
    }
};

/// `x` [B, S, D] through the SwiGLU experts `inds` [B, S, K] of three banks, weighted by `scores`; `limit` > 0 clamps
/// the activation. A new array.
pub fn swiglu(gate: *const Weight, up: *const Weight, down: *const Weight, x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array, limit: i32, s: mlx.mlx_stream) !mlx.mlx_array {
    const f = gate.t.swiglu orelse {
        log.err("{s}: the quant plugin runs no SwiGLU experts\n", .{gate.t.name.slice()});
        return error.OutsideQuantRefused;
    };
    var out: abi.Array = .{};
    try check(gate.t, f(&gate.w, &up.w, &down.w, .{ .ctx = x.ctx }, .{ .ctx = inds.ctx }, .{ .ctx = scores.ctx }, &.{ .limit = limit }, .{ .ctx = s.ctx }, &out));
    return .{ .ctx = out.ctx };
}

/// A dense SwiGLU MLP (`gate`, `up`, `down`) on the plugin where it takes `x` [L, D]; null where it declines.
pub fn denseSwiglu(gate: *const Weight, up: *const Weight, down: *const Weight, x: mlx.mlx_array, limit: i32, s: mlx.mlx_stream) !?mlx.mlx_array {
    const f = gate.t.swiglu orelse return null;
    var out: abi.Array = .{};
    const v = f(&gate.w, &up.w, &down.w, .{ .ctx = x.ctx }, .{}, .{}, &.{ .limit = limit }, .{ .ctx = s.ctx }, &out);
    if (v.code == .unsupported) return null;
    try check(gate.t, v);
    return .{ .ctx = out.ctx };
}

/// One load through the outside table that claims the pack: `tensor` and `bank` load its weights (they outlive
/// `close`).
pub const Format = struct {
    a: std.mem.Allocator,
    t: *const abi.QuantTable,
    config: []u8,
    block: []u8,
    source: abi.Source,
    ctx: abi.LoadCtx,

    /// The load through the table claiming the pack's `quantization_config` or `quantization` block over `source`; null
    /// when no loaded table claims either. A refusal is logged and named.
    pub fn open(a: std.mem.Allocator, dir: *const common.PackDir, model_dir: []const u8, source: abi.Source, s: mlx.mlx_stream, host: ?*const Host) !?*Format {
        if (tables.get().n == 0) return null;
        const n = dir.fileBytes("config.json");
        if (n == 0 or n > 4 << 20) return null;
        const config = try a.alloc(u8, @intCast(n));
        errdefer a.free(config);
        var why: common.Verdict = .{};
        _ = dir.readFile("config.json", config, &why) orelse return error.OutsideQuantRefused;
        const c = (try claimed(a, config)) orelse {
            a.free(config);
            return null;
        };
        errdefer a.free(c.block);
        try check(c.t, c.t.validate(.of(c.block), .of(config), dir));
        const f = try a.create(Format);
        errdefer a.destroy(f);
        f.* = .{ .a = a, .t = c.t, .config = config, .block = c.block, .source = source, .ctx = undefined };
        f.ctx = .{ .model_dir = .of(model_dir), .dir = dir, .block_json = .of(c.block), .config_json = .of(config), .source = &f.source, .stream = .{ .ctx = s.ctx }, .host = host };
        try check(c.t, c.t.open(&f.ctx));
        return f;
    }

    pub fn close(f: *Format) void {
        f.t.close(&f.ctx);
        f.a.free(f.block);
        f.a.free(f.config);
        f.a.destroy(f);
    }

    pub fn name(f: *const Format) []const u8 {
        return f.t.name.slice();
    }

    /// The projection `name` (`in_dim` its inputs); null when the pack stores it in no form the table reads.
    pub fn tensor(f: *const Format, name_: []const u8, in_dim: u32) !?Weight {
        return f.load(f.t.load_tensor, name_, in_dim);
    }

    /// The routed experts `name` as one bank; null as `tensor`.
    pub fn bank(f: *const Format, name_: []const u8, in_dim: u32) !?Weight {
        return f.load(f.t.load_bank, name_, in_dim);
    }

    /// The slot rows per routed layer `budget` buys over the pack's streamed bank (`StreamedRowsDoNotFit` when none
    /// do, unlogged: a caller tries several); null when the table streams none.
    pub fn bankRows(f: *const Format, geo: abi.BankGeometry, budget: abi.BankBudget) !?abi.BankRows {
        const rows_fn = f.t.bank_rows orelse return null;
        var out: abi.BankRows = .{};
        const v = rows_fn(&f.ctx, &geo, budget, &out);
        if (v.code == .too_large) return error.StreamedRowsDoNotFit;
        try check(f.t, v);
        return out;
    }

    /// The pack's streamed bank at the rows `budget` buys; null when the table streams none. It outlives `close`.
    pub fn openBank(f: *const Format, geo: abi.BankGeometry, budget: abi.BankBudget) !?*Bank {
        const open_fn = f.t.open_bank orelse return null;
        const b = try f.a.create(Bank);
        errdefer f.a.destroy(b);
        b.* = .{ .a = f.a, .b = undefined };
        // Zeroed: an older plugin's shorter `Bank` leaves the optional members null.
        @memset(std.mem.asBytes(&b.b), 0);
        const v = open_fn(&f.ctx, &geo, budget, &b.b);
        if (v.code == .too_large) return error.StreamedRowsDoNotFit;
        try check(f.t, v);
        if (b.b.size < @offsetOf(abi.Bank, "rows_shared")) {
            log.err("{s}: its streamed bank lacks the members every bank has: rebuild it with this release's kit\n", .{f.t.name.slice()});
            // The plugin built it: closed when it gave its `close` (read as optional: the zeroed word may be null).
            const Close = ?*const fn (?*anyopaque) callconv(.c) void;
            const given = @as(*const Close, @ptrCast(&b.b.close)).*;
            if (b.b.size >= @offsetOf(abi.Bank, "close") + @sizeOf(Close)) if (given) |c| c(b.b.ctx);
            return error.OutsideQuantRefused;
        }
        return b;
    }

    fn load(f: *const Format, from: *const fn (*const abi.LoadCtx, abi.Str, u32, *abi.Weight) callconv(.c) common.Verdict, name_: []const u8, in_dim: u32) !?Weight {
        var w: abi.Weight = .{};
        const v = from(&f.ctx, .of(name_), in_dim, &w);
        if (!v.isOk() and std.mem.eql(u8, v.why.slice(), "MissingWeight")) return null;
        try check(f.t, v);
        return .{ .t = f.t, .w = w };
    }
};

/// A streamed bank an outside table opened, as the trunk's MoE reads it (`routed`).
pub const Bank = struct {
    a: std.mem.Allocator,
    b: abi.Bank,
    /// The phase the trunk last set (`Routed.phase`).
    prompt: bool = true,

    pub fn close(self: *Bank) void {
        self.b.close(self.b.ctx);
        self.a.destroy(self);
    }

    /// The `source` arm of `Routed` over this bank, with the fused forms the bank offers.
    pub fn routed(self: *Bank) routed_mod.Routed {
        const b = &self.b;
        return .{
            .ctx = self,
            .math = .{ .source = C.rows },
            .settle_fn = C.settle,
            .phase_fn = C.phase,
            .read_ahead_fn = C.readAhead,
            .decode_ids = b.decode_ids,
            .exact_rows = b.size >= @offsetOf(abi.Bank, "exact_rows") + @sizeOf(bool) and b.exact_rows,
            .source_shared = if (offers(b, .rows_shared)) C.rowsShared else null,
            .source_expand = if (offers(b, .rows_expand)) C.expand(.rows_expand) else null,
            .source_expand_into = if (offers(b, .rows_expand_into)) C.expand(.rows_expand_into) else null,
            .source_decode_expand = if (offers(b, .rows_decode_expand)) C.expand(.rows_decode_expand) else null,
            .next_fn = if (offers(b, .next_ahead)) C.nextAhead else null,
        };
    }

    /// The bank's summary lines for the request (`abi.Bank.meters`); empty when it offers none.
    pub fn meters(self: *Bank) []const u8 {
        if (!offers(&self.b, .meters)) return "";
        var out: abi.Str = .{};
        self.b.meters.?(self.b.ctx, &out);
        return out.slice();
    }

    /// Whether the bank sets optional member `field`, read only when its `size` reaches it.
    fn offers(b: *const abi.Bank, comptime field: @EnumLiteral()) bool {
        const name = @tagName(field);
        return b.size >= @offsetOf(abi.Bank, name) + @sizeOf(@FieldType(abi.Bank, name)) and @field(b, name) != null;
    }

    const A = mlx.mlx_array;
    const C = struct {
        fn of(ctx: *anyopaque) *Bank {
            return @ptrCast(@alignCast(ctx));
        }
        fn ret(v: common.Verdict) routed_mod.Ret(void) {
            return if (v.isOk()) .{} else .{ .err = why(v) };
        }
        fn arr(v: common.Verdict, out: abi.Array) routed_mod.Ret(A) {
            return if (v.isOk()) .{ .val = .{ .ctx = out.ctx } } else .{ .err = why(v) };
        }
        /// A refusal's name; one that names nothing is still a refusal.
        fn why(v: common.Verdict) []const u8 {
            return if (v.why.len > 0) v.why.slice() else "OutsideQuantRefused";
        }
        fn rows(ctx: *anyopaque, layer: u32, x: A, idx: A, wts: A) routed_mod.Ret(A) {
            const b = &of(ctx).b;
            var out: abi.Array = .{};
            return arr(b.rows(b.ctx, layer, .{ .ctx = x.ctx }, .{ .ctx = idx.ctx }, .{ .ctx = wts.ctx }, &out), out);
        }
        fn rowsShared(ctx: *anyopaque, layer: u32, x: A, idx: A, wts: A, shared: A) routed_mod.Ret(A) {
            const b = &of(ctx).b;
            var out: abi.Array = .{};
            return arr(b.rows_shared.?(b.ctx, layer, .{ .ctx = x.ctx }, .{ .ctx = idx.ctx }, .{ .ctx = wts.ctx }, .{ .ctx = shared.ctx }, &out), out);
        }
        fn expand(comptime field: @EnumLiteral()) fn (*anyopaque, u32, A, A, A, A, A, A) routed_mod.Ret(A) {
            return struct {
                fn f(ctx: *anyopaque, layer: u32, x: A, idx: A, wts: A, shared: A, res: A, pc: A) routed_mod.Ret(A) {
                    const b = &of(ctx).b;
                    var out: abi.Array = .{};
                    return arr(@field(b, @tagName(field)).?(b.ctx, layer, .{ .ctx = x.ctx }, .{ .ctx = idx.ctx }, .{ .ctx = wts.ctx }, .{ .ctx = shared.ctx }, .{ .ctx = res.ctx }, .{ .ctx = pc.ctx }, &out), out);
                }
            }.f;
        }
        fn settle(ctx: *anyopaque) routed_mod.Ret(void) {
            const b = &of(ctx).b;
            return ret(b.settle(b.ctx));
        }
        fn phase(ctx: *anyopaque, prompt: bool) routed_mod.Ret(void) {
            const self = of(ctx);
            const r = ret(self.b.phase(self.b.ctx, prompt));
            if (r.err.len == 0) self.prompt = prompt;
            return r;
        }
        fn readAhead(ctx: *anyopaque, layer: u32, ids: []const u16) routed_mod.Ret(void) {
            const b = &of(ctx).b;
            return ret(b.read_ahead(b.ctx, layer, ids.ptr, ids.len));
        }
        fn nextAhead(ctx: *anyopaque, layer: u32, ids: A) routed_mod.Ret(void) {
            const b = &of(ctx).b;
            return ret(b.next_ahead.?(b.ctx, layer, .{ .ctx = ids.ctx }));
        }
    };
};

fn check(t: *const abi.QuantTable, v: common.Verdict) !void {
    if (v.isOk()) return;
    log.err("{s}: {s}\n", .{ t.name.slice(), v.why.slice() });
    return error.OutsideQuantRefused;
}

// ── tests ──

const testing = std.testing;

const Toy = struct {
    fn claims(block: abi.Str, _: abi.Str) callconv(.c) common.Claim {
        return if (std.mem.indexOf(u8, block.slice(), "\"toy\"") != null) .by_marker else .no;
    }
    fn validate(_: abi.Str, _: abi.Str, _: *const common.PackDir) callconv(.c) common.Verdict {
        return .ok;
    }
    fn bill(_: abi.Str, _: abi.Str, stored: u64) callconv(.c) u64 {
        return stored;
    }
    fn open(_: *abi.LoadCtx) callconv(.c) common.Verdict {
        return .ok;
    }
    fn close(_: *abi.LoadCtx) callconv(.c) void {}
    fn load(ctx: *const abi.LoadCtx, name: abi.Str, _: u32, out: *abi.Weight) callconv(.c) common.Verdict {
        if (!std.mem.eql(u8, name.slice(), "a")) return common.Verdict.refuse(.failed, "MissingWeight");
        out.* = .{ .w = ctx.source.tensor("a.trellis"), .block = 7 };
        return .ok;
    }
    fn release(w: *abi.Weight) callconv(.c) void {
        w.* = .{};
    }
    fn matmul(_: *const abi.Weight, _: abi.Array, _: abi.Stream, _: *abi.Array) callconv(.c) common.Verdict {
        return common.Verdict.refuse(.unsupported, "toy");
    }
    fn moe(_: *const abi.Weight, _: abi.Array, _: abi.Array, _: abi.Array, _: bool, _: abi.Stream, _: *abi.Array) callconv(.c) common.Verdict {
        return common.Verdict.refuse(.unsupported, "toy");
    }
    fn swigluDecline(_: *const abi.Weight, _: *const abi.Weight, _: *const abi.Weight, _: abi.Array, inds: abi.Array, _: abi.Array, _: *const abi.Swiglu, _: abi.Stream, _: *abi.Array) callconv(.c) common.Verdict {
        return if (inds.ctx == null) common.Verdict.refuse(.unsupported, "dense") else common.Verdict.refuse(.failed, "toy");
    }
    const table: abi.QuantTable = .{ .name = .of("toy"), .claims = &claims, .validate = &validate, .bill = &bill, .open = &open, .close = &close, .load_tensor = &load, .load_bank = &load, .release = &release, .matmul = &matmul, .moe = &moe, .swiglu = &swigluDecline };
};

test "outside quant: the table claiming a pack's quantization_config loads its weights; an absent one is null, not an error" {
    const fixture = @import("../testing/fixture.zig");
    const saved = tables.get().*;
    defer tables.get().* = saved;
    tables.get().* = .{};
    const mem: fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = "{\"model_type\":\"x\",\"quantization_config\":{\"quant_method\":\"toy\"}}" }} };
    const dir = mem.packDir();
    const none: abi.Source = .{ .ctx = null, .get = struct {
        fn get(_: ?*anyopaque, _: abi.Str) callconv(.c) abi.Array {
            return .{};
        }
    }.get };
    try testing.expect(try Format.open(testing.allocator, &dir, "/m", none, .{ .ctx = null }, null) == null);
    register(&Toy.table);
    const f = (try Format.open(testing.allocator, &dir, "/m", none, .{ .ctx = null }, null)).?;
    defer f.close();
    try testing.expectEqualStrings("toy", f.name());
    try testing.expectEqualStrings("{\"quant_method\":\"toy\"}", f.ctx.block_json.slice());
    try testing.expectEqual(@as(u32, 7), (try f.tensor("a", 0)).?.w.block);
    try testing.expect(try f.bank("b", 0) == null);
    try testing.expectError(error.OutsideQuantRefused, (try f.tensor("a", 0)).?.matmul(.{ .ctx = null }, .{ .ctx = null }));
}

test "outside quant: a weight's row-exact width is its table's declaration, one row where it declares none" {
    var table = Toy.table;
    const w: Weight = .{ .t = &table, .w = .{} };
    try testing.expectEqual(@as(u32, 1), w.exactRows());
    table.exact_rows = 16;
    try testing.expectEqual(@as(u32, 16), w.exactRows());
}

test "outside quant: a dense SwiGLU the table declines is null, the routed one's refusal an error" {
    const w: Weight = .{ .t = &Toy.table, .w = .{} };
    const none: mlx.mlx_array = .{ .ctx = null };
    try testing.expect(try denseSwiglu(&w, &w, &w, none, 10, .{ .ctx = null }) == null);
    var one: u8 = 0;
    try testing.expectError(error.OutsideQuantRefused, swiglu(&w, &w, &w, none, .{ .ctx = &one }, none, 10, .{ .ctx = null }));
}

const ToyBank = struct {
    var closed: u32 = 0;
    var sum: u8 = 0;
    fn rows(_: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.BankRows) callconv(.c) common.Verdict {
        if (budget.prompt < 100) return common.Verdict.refuse(.too_large, "StreamedRowsDoNotFit");
        out.* = .{ .prompt = @intCast(budget.prompt / 100), .decode = geo.n_experts, .row_bytes = 100 };
        return .ok;
    }
    fn meters(_: ?*anyopaque, out: *abi.Str) callconv(.c) void {
        out.* = .of("[stream-stats] toy: routes=1");
    }
    fn open(_: *const abi.LoadCtx, geo: *const abi.BankGeometry, _: abi.BankBudget, out: *abi.Bank) callconv(.c) common.Verdict {
        out.* = .{ .ctx = @constCast(@ptrCast(geo.gates.?)), .close = &close, .about = .of("toy"), .decode_ids = 48, .rows = &routedRows, .settle = &ok0, .phase = &phase, .read_ahead = &readAhead, .rows_shared = &shared, .meters = &meters, .exact_rows = true };
        return .ok;
    }
    /// A bank built before its optional members: whatever lies past its `size` is not read.
    fn openOlder(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.Bank) callconv(.c) common.Verdict {
        const v = open(ctx, geo, budget, out);
        out.size = @offsetOf(abi.Bank, "rows_shared");
        return v;
    }
    fn openShort(ctx: *const abi.LoadCtx, geo: *const abi.BankGeometry, budget: abi.BankBudget, out: *abi.Bank) callconv(.c) common.Verdict {
        const v = open(ctx, geo, budget, out);
        out.size = @offsetOf(abi.Bank, "rows");
        return v;
    }
    fn close(_: ?*anyopaque) callconv(.c) void {
        closed += 1;
    }
    fn routedRows(_: ?*anyopaque, layer: u32, _: abi.Array, _: abi.Array, _: abi.Array, out: *abi.Array) callconv(.c) common.Verdict {
        if (layer == 1) return common.Verdict.named(.failed, "ExpertReadShort");
        if (layer == 2) return .{ .code = .failed };
        out.* = .{ .ctx = &sum };
        return .ok;
    }
    fn shared(_: ?*anyopaque, _: u32, _: abi.Array, _: abi.Array, _: abi.Array, _: abi.Array, _: *abi.Array) callconv(.c) common.Verdict {
        return .ok;
    }
    fn ok0(_: ?*anyopaque) callconv(.c) common.Verdict {
        return .ok;
    }
    fn phase(_: ?*anyopaque, prompt: bool) callconv(.c) common.Verdict {
        return if (prompt) .ok else common.Verdict.named(.failed, "NotPrefill");
    }
    fn readAhead(_: ?*anyopaque, _: u32, ids: [*]const u16, n: usize) callconv(.c) common.Verdict {
        sum = @intCast(ids[0] + n);
        return .ok;
    }
};

test "outside quant: a table's streamed bank reads as a Routed source; its refusals cross by name" {
    const err_name = @import("../util/err_name.zig");
    var table = Toy.table;
    const plain: Format = .{ .a = testing.allocator, .t = &Toy.table, .config = &.{}, .block = &.{}, .source = undefined, .ctx = undefined };
    try testing.expect(try plain.bankRows(.{ .hidden = 8, .inter = 4, .top_k = 2, .n_experts = 16, .n_layers = 2 }, .{ .prompt = 1000, .decode = 1000 }) == null);
    table.bank_rows = &ToyBank.rows;
    table.open_bank = &ToyBank.open;
    const f: Format = .{ .a = testing.allocator, .t = &table, .config = &.{}, .block = &.{}, .source = undefined, .ctx = undefined };
    const gates = [2]abi.BankGate{ .{}, .{} };
    const geo: abi.BankGeometry = .{ .hidden = 8, .inter = 4, .top_k = 2, .n_experts = 16, .n_layers = 2, .gates = &gates };
    try testing.expectEqual(@as(u32, 10), (try f.bankRows(geo, .{ .prompt = 1000, .decode = 1000 })).?.prompt);
    try testing.expectError(error.StreamedRowsDoNotFit, f.bankRows(geo, .{ .prompt = 10, .decode = 10 }));
    const b = (try f.openBank(geo, .{ .prompt = 1000, .decode = 1000 })).?;
    const r = b.routed();
    try testing.expectEqual(@as(usize, 48), r.decode_ids);
    try testing.expect(r.exact_rows);
    try testing.expect(r.source_shared != null and r.source_expand == null and r.next_fn == null);
    try testing.expectEqualStrings("[stream-stats] toy: routes=1", b.meters());
    const none: mlx.mlx_array = .{ .ctx = null };
    try testing.expectEqual(@as(?*anyopaque, &ToyBank.sum), (try r.math.source(r.ctx, 0, none, none, none).get()).ctx);
    try testing.expectError(error.EngineFailed, r.math.source(r.ctx, 1, none, none, none).get());
    try testing.expectEqualStrings("ExpertReadShort", err_name.last());
    try r.read_ahead_fn.?(r.ctx, 0, &.{ 5, 6, 7 }).get();
    try testing.expectEqual(@as(u8, 8), ToyBank.sum);
    try r.phase(true);
    try testing.expectError(error.EngineFailed, r.phase(false));
    try testing.expect(b.prompt);
    // A refusal that names nothing is still a refusal.
    try testing.expectError(error.EngineFailed, r.math.source(r.ctx, 2, none, none, none).get());
    b.close();
    try testing.expectEqual(@as(u32, 1), ToyBank.closed);
    table.open_bank = &ToyBank.openOlder;
    const older = (try f.openBank(geo, .{ .prompt = 1000, .decode = 1000 })).?;
    try testing.expect(older.routed().source_shared == null and !older.routed().exact_rows);
    try testing.expectEqualStrings("", older.meters());
    older.close();
    table.open_bank = &ToyBank.openShort;
    try testing.expectError(error.OutsideQuantRefused, f.openBank(geo, .{ .prompt = 1000, .decode = 1000 }));
    try testing.expectEqual(@as(u32, 3), ToyBank.closed); // the refused bank is closed too
}
