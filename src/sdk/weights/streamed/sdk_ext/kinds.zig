//! The `quant` and `expert_source` contracts' tables (docs/plugins.md): what mlx-stream's EXL3 quant and EXL3 expert
//! source declare, checked at comptime by `of`. They are internals of the `deepseek_v41` arch (the host registers the
//! arch only); the arch binds both at comptime, and nothing per layer crosses these tables.

const std = @import("std");
const peek = @import("sdk");
const bill = @import("sdk");
const kernels = @import("kernels.zig");
const quant = @import("quant.zig");
const check = @import("sdk").check;

const Allocator = std.mem.Allocator;
const BillFn = *const fn (gpa: Allocator, io: std.Io, req: *const bill.BillRequest) anyerror!bill.MemoryBill;

fn billOf(comptime T: type, comptime w: []const u8) ?BillFn {
    if (!check.has(T, "bill")) return null;
    check.fnDecl(w, T, "bill", &.{ Allocator, std.Io, *const bill.BillRequest }, bill.MemoryBill);
    return struct {
        fn f(gpa: Allocator, io: std.Io, req: *const bill.BillRequest) anyerror!bill.MemoryBill {
            return T.bill(gpa, io, req);
        }
    }.f;
}

/// Weight load and the MoE matmul for one weight format. The load path asks every quant once per weight group;
/// the arch binds the one that claimed its group at comptime (`sdk.quant`'s contract, C2: Arrays, claims,
/// Accepted, accept).
pub const Quant = struct {
    name: []const u8,
    claims: *const fn (p: *const peek.GroupPeek, why: ?*peek.Diag) ?peek.Priority,
    /// G5: the quant's kernel set pin (`kernel_pin`); null = it brings no kernels.
    kernels: ?kernels.Pin,
    /// G4: resident bytes, including what its codec drops and builds; null = none.
    bill: ?BillFn,

    pub fn of(comptime T: type) Quant {
        comptime {
            const w = "quant " ++ @typeName(T);
            check.nameDecl(w, T);
            check.fnDecl(w, T, "claims", &.{ *const peek.GroupPeek, ?*peek.Diag }, ?peek.Priority);
            quant.check(T);
            if (check.has(T, "kernel_pin")) check.valueDecl(w, T, "kernel_pin", kernels.Pin);
        }
        return .{
            .name = T.name,
            .claims = T.claims,
            .kernels = if (check.has(T, "kernel_pin")) T.kernel_pin else null,
            .bill = comptime billOf(T, "quant " ++ @typeName(T)),
        };
    }
};

/// What an expert source declares it does (`sdk.expert.Caps`).
pub const ExpertCaps = @import("expert.zig").Caps;

/// Where routed experts live and how they reach the GPU (G6). The arch drives its source per layer through the
/// source's comptime contract; this table is the registry's: discovery, `/props` and the source's bill terms.
pub const ExpertSource = struct {
    name: []const u8,
    claims: *const fn (p: *const peek.ConfigPeek) ?peek.Priority,
    caps: ExpertCaps,
    /// G4: per_row, the transient rows per phase at the installed release route, staging; null = none.
    bill: ?BillFn,

    pub fn of(comptime T: type) ExpertSource {
        comptime {
            const w = "expert_source " ++ @typeName(T);
            check.nameDecl(w, T);
            check.fnDecl(w, T, "claims", &.{*const peek.ConfigPeek}, ?peek.Priority);
            if (check.has(T, "caps")) check.valueDecl(w, T, "caps", ExpertCaps);
        }
        return .{
            .name = T.name,
            .claims = T.claims,
            .caps = if (check.has(T, "caps")) T.caps else .{},
            .bill = comptime billOf(T, "expert_source " ++ @typeName(T)),
        };
    }
};

const testing = std.testing;

const fixture = struct {
    fn claimsModel(p: *const peek.ConfigPeek) ?peek.Priority {
        return if (p.modelType() != null) .generic else null;
    }
    fn billBytes(gpa: Allocator, io: std.Io, req: *const bill.BillRequest) !bill.MemoryBill {
        _ = gpa;
        _ = io;
        return .{ .per_row = req.max_tokens };
    }
    fn claimsGroup(g: *const peek.GroupPeek, why: ?*peek.Diag) ?peek.Priority {
        if (g.hidden == 0) return quant.decline(why, "fixture quant: no hidden", .{});
        return .native;
    }
    /// The C2 quant half `quant.check` reads; the call half is never instantiated here.
    fn Quant(comptime with_pin: bool, comptime with_bill: bool) type {
        return struct {
            pub const name = "fixture-quant";
            pub fn Arrays(comptime T: type) type {
                return struct { w: T };
            }
            pub const claims = claimsGroup;
            pub fn Accepted(comptime G: type) type {
                _ = G;
                return struct {};
            }
            pub fn accept() void {}
            pub const kernel_pin = if (with_pin) kernels.Pin{ .manifest_sha256 = "abc123" } else {};
            pub const bill = if (with_bill) billBytes else {};
        };
    }
};

test "sdk kinds: a quant's kernel pin and bill are optional; `{}` switches either off; the bill runs through the table" {
    const full = comptime Quant.of(fixture.Quant(true, true));
    const bare = comptime Quant.of(fixture.Quant(false, false));
    try testing.expectEqualStrings("fixture-quant", full.name);
    try testing.expectEqualStrings("abc123", full.kernels.?.manifest_sha256);
    try testing.expect(bare.kernels == null and bare.bill == null);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try peek.ConfigPeek.parse(arena.allocator(), "/m", "{}");
    const req: bill.BillRequest = .{ .peek = &p, .cfg = &p, .routes = &p, .prompt_tokens = 1, .max_tokens = 4096, .ceiling = 0, .stop = 0 };
    try testing.expectEqual(@as(u64, 4096), (try full.bill.?(testing.allocator, testing.io, &req)).per_row);
    // the claim and its decline reason cross the table
    var why: peek.Diag = .{};
    const g: peek.GroupPeek = .{ .quantization = .null, .hidden = 0, .inter = 0, .n_experts = 0, .n_layers = 0, .layers = &.{} };
    try testing.expectEqual(@as(?peek.Priority, null), full.claims(&g, &why));
    try testing.expectEqualStrings("fixture quant: no hidden", why.message());
}

test "sdk kinds: an expert source's caps default to none and a declared set is copied; its bill is optional" {
    const Plain = struct {
        pub const name = "plain-stream";
        pub const claims = fixture.claimsModel;
    };
    const Declared = struct {
        pub const name = "declared-stream";
        pub const claims = fixture.claimsModel;
        pub const caps: ExpertCaps = .{ .two_phase = true, .event_gates = true };
        pub const bill = fixture.billBytes;
    };
    const plain = comptime ExpertSource.of(Plain);
    const declared = comptime ExpertSource.of(Declared);
    try testing.expectEqual(ExpertCaps{}, plain.caps);
    try testing.expect(plain.bill == null);
    try testing.expectEqual(ExpertCaps{ .two_phase = true, .event_gates = true }, declared.caps);
    try testing.expect(declared.bill != null);
}
