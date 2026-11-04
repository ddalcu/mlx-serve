//! Conformance (docs/plugins.md): the checks a plugin's kinds run against the SDK's contracts, and the
//! fakes the host's own tests drive. Every check declares its lane:
//! - `cpu`: `zig build conformance` with no device; the lane fails if a Metal device was created.
//! - `gpu_small`: fixture shapes on a Mac with a GPU.
//! - `window`: a plugin's own full-size checks, never in the suite.

const std = @import("std");
const builtin = @import("builtin");
const mlx = @import("mlx_host").mlx;
const peek = @import("peek.zig");
const arch = @import("arch.zig");
const spec = @import("spec.zig");
const bill = @import("memory_bill.zig");
const bill_mod = bill;

// ── The CPU lane's device check ──

extern fn _dyld_image_count() u32;
extern fn _dyld_get_image_name(image_index: u32) ?[*:0]const u8;

/// Whether this process created a Metal device: only that maps a GPU driver bundle (AGXMetal*).
pub fn deviceCreated() bool {
    if (builtin.os.tag != .macos) return false;
    for (0.._dyld_image_count()) |i| {
        const name = _dyld_get_image_name(@intCast(i)) orelse continue;
        if (std.mem.indexOf(u8, std.mem.span(name), "AGXMetal") != null) return true;
    }
    return false;
}

/// The CPU lane's last check: nothing before it created a device (any MLX array in the Metal build does).
pub fn expectNoDevice() error{DeviceCreatedInCpuLane}!void {
    if (deviceCreated()) return error.DeviceCreatedInCpuLane;
}

// ── Claims (cpu) ──

/// A fixture config and the claim it must get: the plugin's own config at its priority, near misses declined.
pub const ClaimCase = struct { config: []const u8, want: ?peek.Priority };

/// A weight group as a quant claims it at load: its `quantization` description (JSON) and dims, and the claim it must
/// get.
pub const GroupClaimCase = struct { quantization: []const u8, hidden: u64, inter: u64, want: ?peek.Priority };

pub fn expectGroupClaims(claims: *const fn (*const peek.GroupPeek, ?*peek.Diag) ?peek.Priority, cases: []const GroupClaimCase) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for (cases) |c| {
        const q = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), c.quantization, .{});
        const g: peek.GroupPeek = .{ .quantization = q, .hidden = c.hidden, .inter = c.inter, .n_experts = 0, .n_layers = 0, .layers = &.{} };
        var why: peek.Diag = .{};
        std.testing.expectEqual(c.want, claims(&g, &why)) catch |e| {
            std.debug.print("claims: {s} ({s})\n", .{ c.quantization, why.message() });
            return e;
        };
    }
}

pub fn expectClaims(claims: *const fn (*const peek.ConfigPeek) ?peek.Priority, cases: []const ClaimCase) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for (cases) |c| {
        const p = try peek.ConfigPeek.parse(arena.allocator(), "/fixture", c.config);
        std.testing.expectEqual(c.want, claims(&p)) catch |e| {
            std.debug.print("claims: {s}\n", .{c.config});
            return e;
        };
    }
}

// ── Bills, fills and admission (cpu) ──

/// G4: an arch's itemized bill (`A.bill`) bounds the process exactly as its load preflight (`A.loadBytes`) bills it, at
/// the fill's floor rows: the preflight and the admission read one bill.
pub fn expectBillBoundsLoad(comptime A: type, gpa: std.mem.Allocator, io: std.Io, cfg: *const A.Config, facts: *const arch.LoadFacts, req: *const bill.BillRequest, floor: bill.Rows) !void {
    const mb = try A.bill(gpa, io, req);
    defer mb.free(gpa);
    try std.testing.expectEqual(try A.loadBytes(gpa, io, cfg, facts, req.ceiling), mb.processBound(floor));
}

// ── Fixture files (cpu) ──

/// A test directory's absolute path, in `buf` (a bank or a loader opens absolute paths only).
pub fn tmpRoot(tmp: *std.testing.TmpDir, buf: []u8) ![]const u8 {
    return buf[0..try tmp.dir.realPath(std.testing.io, buf)];
}

/// One tensor of a fixture safetensors file: its name, the file's name for its dtype, its shape and its `len` bytes
/// (`bytes`, else zeros).
pub const StTensor = struct { name: []const u8, dtype: []const u8, shape: []const u64, len: u64, bytes: ?[]const u8 = null };

/// A safetensors image of `tensors` in order: the 8-byte little-endian header length, the JSON header, the data.
/// `metadata` adds mlx-lm's `"__metadata__":{"format":"mlx"}`; `pad` pads the header with spaces to a multiple of 8.
/// Caller frees.
pub fn safetensors(a: std.mem.Allocator, tensors: []const StTensor, opts: struct { metadata: bool = false, pad: bool = false }) ![]u8 {
    var hdr: std.ArrayList(u8) = .empty;
    defer hdr.deinit(a);
    try hdr.append(a, '{');
    if (opts.metadata) try hdr.appendSlice(a, "\"__metadata__\":{\"format\":\"mlx\"}");
    var off: u64 = 0;
    for (tensors, 0..) |t, i| {
        if (i > 0 or opts.metadata) try hdr.append(a, ',');
        try hdr.print(a, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.name, t.dtype });
        for (t.shape, 0..) |d, k| try hdr.print(a, "{s}{d}", .{ if (k == 0) "" else ",", d });
        try hdr.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ off, off + t.len });
        off += t.len;
    }
    try hdr.append(a, '}');
    if (opts.pad) while (hdr.items.len % 8 != 0) try hdr.append(a, ' ');
    const file = try a.alloc(u8, @intCast(8 + hdr.items.len + off));
    @memset(file, 0);
    std.mem.writeInt(u64, file[0..8], hdr.items.len, .little);
    @memcpy(file[8..][0..hdr.items.len], hdr.items);
    var at = 8 + hdr.items.len;
    for (tensors) |t| {
        if (t.bytes) |b| @memcpy(file[at..][0..b.len], b);
        at += @intCast(t.len);
    }
    return file;
}

// ── Fakes for the host's tests (cpu) ──

pub const FakeOptions = struct {
    caps: arch.Caps = .{ .owns_decode_state = true, .prefill_whole_prompt = true, .prefill_yields_last_logits = true },
    /// The model_type the fake claims.
    model_type: []const u8 = "fake_arch",
    handover: bool = true,
    /// Rows a round verifies; 0 = no draft lane.
    block_size: u32 = 0,
    /// The prompt admission's bytes; null = the host's estimator.
    prompt_bytes: ?u64 = null,
    /// `restorePrefix`: the most positions of a prefix-cache match the fake keeps; null = no hook.
    restore_cap: ?u64 = null,
    /// The draft lane also arms clean sampled requests (else greedy clean only).
    lane_samples: bool = false,
};

/// Every call a fake arch's module received.
pub const FakeCalls = struct {
    init: u32 = 0,
    deinit: u32 = 0,
    prefill: u32 = 0,
    step: u32 = 0,
    handover: u32 = 0,
    rounds: u32 = 0,
    restore: u32 = 0,
};

/// An arch for host tests: no MLX (its logits are empty handles), caps per test, every call counted.
pub fn FakeArch(comptime opts: FakeOptions) type {
    return struct {
        pub const name = "fake-arch";
        pub const caps = opts.caps;
        pub const Config = struct { settings_applied: u32 = 0 };
        pub const Module = struct {
            gpa: std.mem.Allocator,
            calls: *FakeCalls,
            position: u64 = 0,
            last_handover: ?arch.DecodeHandover = null,
            last_request: ?arch.RequestShape = null,
            /// The match the last `restorePrefix` was offered.
            last_prefix: usize = 0,
            /// The sampling the last `round` received.
            last_round_sampling: ?spec.SamplingParams = null,
        };
        /// The request the last `arm` received (arm takes a const module).
        pub var last_arm: ?spec.ArmRequest = null;
        /// The counters every module of this fake writes; reset per test.
        pub var calls: FakeCalls = .{};

        pub fn claims(p: *const peek.ConfigPeek) ?peek.Priority {
            const t = p.modelType() orelse return null;
            return if (std.mem.eql(u8, t, opts.model_type)) .native else null;
        }
        pub fn parse(gpa: std.mem.Allocator, p: *const peek.ConfigPeek, diag: *peek.Diag) !*Config {
            if (p.int("refuse") != null) {
                diag.set("fake arch: refused by the fixture", .{});
                return error.FakeArchRefused;
            }
            const c = try gpa.create(Config);
            c.* = .{};
            return c;
        }
        pub fn freeConfig(gpa: std.mem.Allocator, c: *Config) void {
            gpa.destroy(c);
        }
        pub fn shell(_: *const Config) arch.Shell {
            return .{ .num_experts = 4, .num_layers = 2 };
        }
        pub fn applySettings(c: *Config, _: std.json.Value) void {
            c.settings_applied += 1;
        }
        pub fn loadBytes(_: std.mem.Allocator, _: std.Io, _: *const Config, _: *const arch.LoadFacts, _: u64) !u64 {
            return 1_000_000_000;
        }
        pub const promptBytes = if (opts.prompt_bytes) |n| struct {
            fn f(_: *const Config, _: u64, _: u32) u64 {
                return n;
            }
        }.f else {};
        pub fn init(load: *const arch.LoadCtx, _: *const Config) !*Module {
            const m = try load.gpa.create(Module);
            m.* = .{ .gpa = load.gpa, .calls = &calls };
            calls.init += 1;
            return m;
        }
        pub fn deinit(m: *Module) void {
            m.calls.deinit += 1;
            m.gpa.destroy(m);
        }
        pub fn prefill(m: *Module, ids: []const u32, req: arch.RequestShape) !mlx.mlx_array {
            m.calls.prefill += 1;
            m.position = ids.len;
            m.last_request = req;
            return .{};
        }
        pub fn step(m: *Module, ids: []const u32) !mlx.mlx_array {
            m.calls.step += 1;
            m.position += ids.len;
            return .{};
        }
        pub fn position(m: *const Module) u64 {
            return m.position;
        }
        pub const handover = if (opts.handover) struct {
            fn f(m: *Module, h: arch.DecodeHandover) !void {
                m.calls.handover += 1;
                m.last_handover = h;
            }
        }.f else {};
        pub const restorePrefix = if (opts.restore_cap) |cap| struct {
            fn f(m: *Module, prefix: []const u32) u64 {
                m.calls.restore += 1;
                m.last_prefix = prefix.len;
                m.position = @min(prefix.len, cap);
                return m.position;
            }
        }.f else {};
        pub const draft_lane = if (opts.block_size > 0) struct {
            pub fn blockSize(_: *const Module) u32 {
                return opts.block_size;
            }
            pub fn laneName(_: *const Module) []const u8 {
                return "fake lane";
            }
            pub fn arm(_: *Module, req: spec.ArmRequest) spec.DraftArm {
                last_arm = req;
                return if (req.clean and (req.greedy or opts.lane_samples)) .typical else .off;
            }
            /// Keeps min(cap, block - 1) drafts: [t1, t1 + 1, ...], the next token after them.
            pub fn round(m: *Module, a: std.mem.Allocator, t1: u32, accepted_cap: u32, sampling: spec.SamplingParams) !spec.DraftRound {
                m.last_round_sampling = sampling;
                const k = @min(accepted_cap, opts.block_size - 1);
                const toks = try a.alloc(u32, k + 1);
                for (toks, 0..) |*t, i| t.* = t1 + @as(u32, @intCast(i));
                m.calls.rounds += 1;
                m.position += toks.len;
                return .{ .tokens = toks, .accepted = k, .next_token = t1 + k + 1 };
            }
            pub fn stats(m: *const Module) spec.DraftStats {
                return .{ .rounds = m.calls.rounds };
            }
        } else {};
    };
}

const testing = std.testing;

test "sdk testing: the fake arch's table counts every call, and its optional hooks follow its options" {
    const Fake = FakeArch(.{ .block_size = 5, .prompt_bytes = 7 });
    Fake.calls = .{};
    const vt = comptime arch.Arch.of(Fake);
    try testing.expect(vt.caps.owns_decode_state and vt.handover != null and vt.prompt_bytes != null and vt.bill == null);
    var diag: peek.Diag = .{};
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try peek.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\":\"fake_arch\"}");
    try testing.expectEqual(@as(?peek.Priority, .native), vt.claims(&p));
    const cfg = try vt.parse(testing.allocator, &p, &diag);
    defer vt.free_config(testing.allocator, cfg);
    vt.apply_settings(cfg, .null);
    try testing.expectEqual(@as(u32, 2), vt.shell(cfg).num_layers);
    try testing.expectEqual(@as(u64, 7), vt.prompt_bytes.?(cfg, 16384, 1024));
    const load: arch.LoadCtx = .{ .gpa = testing.allocator, .io = testing.io, .stream = .{}, .weights = undefined, .loader = undefined, .facts = .{ .wired_margin_bytes = 0 }, .ceiling = 0 };
    const m = try vt.init(&load, cfg);
    _ = try vt.prefill(m, &.{ 1, 2, 3 }, .{ .prompt_tokens = 3, .max_tokens = 8, .host_context = 4096 });
    try vt.handover.?(m, .{ .prompt_tokens = 3, .reserved_tokens = 0, .native_draft = true });
    _ = try vt.step(m, &.{9});
    vt.deinit(m);
    try testing.expectEqual(FakeCalls{ .init = 1, .deinit = 1, .prefill = 1, .step = 1, .handover = 1 }, Fake.calls);

    const Bare = FakeArch(.{ .handover = false, .caps = .{} });
    const bare = comptime arch.Arch.of(Bare);
    try testing.expect(bare.handover == null and bare.prompt_bytes == null and bare.spec == .none and !bare.caps.owns_decode_state);
    try testing.expect(vt.restore_prefix == null and bare.restore_prefix == null);
    try testing.expectError(error.FakeArchRefused, vt.parse(testing.allocator, &(try peek.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\":\"fake_arch\",\"refuse\":1}")), &diag));
    try testing.expectEqualStrings("fake arch: refused by the fixture", diag.message());
}

test "sdk arch: restore_prefix keeps at most the host's match, counted; absent unless declared" {
    const Fake = FakeArch(.{ .restore_cap = 5 });
    Fake.calls = .{};
    const vt = comptime arch.Arch.of(Fake);
    var m: Fake.Module = .{ .gpa = testing.allocator, .calls = &Fake.calls };
    const ids = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    // The match past what the module holds: it keeps its own 5; a shorter match: the match.
    try testing.expectEqual(@as(u64, 5), vt.restore_prefix.?(&m, &ids));
    try testing.expectEqual(@as(usize, 8), m.last_prefix);
    try testing.expectEqual(@as(u64, 4), vt.restore_prefix.?(&m, ids[0..4]));
    try testing.expectEqual(@as(u64, 0), vt.restore_prefix.?(&m, ids[0..0]));
    try testing.expectEqual(@as(u32, 3), Fake.calls.restore);
}

test "sdk testing: claims fixtures run on any arch's claim" {
    try expectClaims(FakeArch(.{}).claims, &.{
        .{ .config = "{\"model_type\":\"fake_arch\"}", .want = .native },
        .{ .config = "{\"model_type\":\"deepseek_v4\"}", .want = null },
        .{ .config = "{\"architectures\":[\"X\"]}", .want = null },
    });
}

test "sdk testing: the fake's draft lane names itself, arms only greedy clean requests and counts its rounds" {
    const Fake = FakeArch(.{ .block_size = 3 });
    Fake.calls = .{};
    const vt = comptime arch.Arch.of(Fake);
    const lane = vt.spec.draft_lane;
    var calls: FakeCalls = .{};
    var m: Fake.Module = .{ .gpa = testing.allocator, .calls = &calls };
    try testing.expectEqual(@as(u32, 3), lane.block_size(&m));
    try testing.expectEqualStrings("fake lane", lane.lane_name(&m));
    try testing.expectEqual(spec.DraftArm.typical, lane.arm(&m, .{ .greedy = true, .clean = true }));
    try testing.expectEqual(spec.DraftArm.off, lane.arm(&m, .{ .greedy = true, .clean = false }));
    try testing.expectEqual(spec.DraftArm.off, lane.arm(&m, .{ .greedy = false, .clean = true }));
    // a cap past the block keeps block - 1 drafts
    var r = try lane.round(&m, testing.allocator, 40, 9, .{ .temperature = 0.6, .top_p = 0.9, .seed = 7 });
    defer r.deinit(testing.allocator);
    try testing.expectEqualSlices(u32, &.{ 40, 41, 42 }, r.tokens);
    try testing.expectEqual(spec.SamplingParams{ .temperature = 0.6, .top_p = 0.9, .seed = 7 }, m.last_round_sampling.?);
    try testing.expectEqual(@as(u32, 43), r.next_token);
    try testing.expectEqual(@as(u64, 1), lane.stats(&m).rounds);
}

test "sdk testing: an arch's bill bounds its load preflight at the floor rows, or the check names the gap" {
    const gb: u64 = 1_000_000_000;
    const Bills = struct {
        fn of(per_row: u64) type {
            return struct {
                pub const Config = struct {};
                const terms = [_]bill_mod.MemoryBill.Term{.{ .name = "residents", .bytes = .{ 1 * gb, 1 * gb }, .at_construction = true }};
                pub fn bill(gpa: std.mem.Allocator, _: std.Io, _: *const bill_mod.BillRequest) !bill_mod.MemoryBill {
                    return .{ .terms = try gpa.dupe(bill_mod.MemoryBill.Term, &terms), .per_row = per_row };
                }
                pub fn loadBytes(_: std.mem.Allocator, _: std.Io, _: *const Config, _: *const arch.LoadFacts, _: u64) !u64 {
                    // the fake arch's own preflight: 1 GB of residents and 1 GB of slot rows
                    return 2 * gb;
                }
            };
        }
    };
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const p = try peek.ConfigPeek.parse(arena.allocator(), "/m", "{}");
    const req: bill.BillRequest = .{ .peek = &p, .cfg = &p, .routes = &p, .prompt_tokens = 1, .max_tokens = 1, .ceiling = 100 * gb, .stop = 2 * gb };
    const facts: arch.LoadFacts = .{ .wired_margin_bytes = 0 };
    const floor: bill.Rows = .{ .prompt = 8, .decode = 16 };
    const cfg: Bills.of(0).Config = .{};
    // 1 GB + 16 decode rows x 1/16 GB = the preflight's 2 GB
    try expectBillBoundsLoad(Bills.of(gb / 16), testing.allocator, testing.io, &cfg, &facts, &req, floor);
    // twice the row bytes: the bill bounds 3 GB where the preflight asks 2
    try testing.expectError(error.TestExpectedEqual, expectBillBoundsLoad(Bills.of(gb / 8), testing.allocator, testing.io, &cfg, &facts, &req, floor));
}

test "sdk testing: the CPU lane's device probe reads the loaded images and has created no device here" {
    try testing.expect(!deviceCreated());
    try expectNoDevice();
}

test "sdk arch: the table's load preflight and bill call the arch's own; a hook switched off is absent" {
    const Base = FakeArch(.{ .handover = false });
    const Billed = struct {
        pub const name = Base.name;
        pub const caps = Base.caps;
        pub const Config = Base.Config;
        pub const Module = Base.Module;
        pub const claims = Base.claims;
        pub const parse = Base.parse;
        pub const freeConfig = Base.freeConfig;
        pub const shell = Base.shell;
        pub const applySettings = Base.applySettings;
        pub const loadBytes = Base.loadBytes;
        pub const init = Base.init;
        pub const deinit = Base.deinit;
        pub const prefill = Base.prefill;
        pub const step = Base.step;
        pub const position = Base.position;
        pub const handover = {};
        pub const draft_lane = {};
        pub fn bill(_: std.mem.Allocator, _: std.Io, req: *const bill_mod.BillRequest) !bill_mod.MemoryBill {
            return .{ .per_row = req.prompt_tokens };
        }
    };
    const vt = comptime arch.Arch.of(Billed);
    try testing.expect(vt.handover == null and vt.spec == .none and vt.prompt_bytes == null);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var diag: peek.Diag = .{};
    const p = try peek.ConfigPeek.parse(arena.allocator(), "/m", "{\"model_type\":\"fake_arch\"}");
    const cfg = try vt.parse(testing.allocator, &p, &diag);
    defer vt.free_config(testing.allocator, cfg);
    const facts: arch.LoadFacts = .{ .wired_margin_bytes = 0 };
    try testing.expectEqual(@as(u64, 1_000_000_000), try vt.load_bytes(testing.allocator, testing.io, cfg, &facts, 0));
    const req: bill.BillRequest = .{ .peek = &p, .cfg = cfg, .routes = cfg, .prompt_tokens = 77, .max_tokens = 1, .ceiling = 0, .stop = 0 };
    try testing.expectEqual(@as(u64, 77), (try vt.bill.?(testing.allocator, testing.io, &req)).per_row);
}

test "sdk testing: a claims fixture and a group fixture that disagree fail the check" {
    try testing.expectError(error.TestExpectedEqual, expectClaims(FakeArch(.{}).claims, &.{.{ .config = "{\"model_type\":\"fake_arch\"}", .want = .generic }}));
    const Never = struct {
        fn claims(_: *const peek.GroupPeek, why: ?*peek.Diag) ?peek.Priority {
            if (why) |d| d.set("never", .{});
            return null;
        }
    };
    try expectGroupClaims(Never.claims, &.{.{ .quantization = "{}", .hidden = 1, .inter = 1, .want = null }});
    try testing.expectError(error.TestExpectedEqual, expectGroupClaims(Never.claims, &.{.{ .quantization = "null", .hidden = 1, .inter = 1, .want = .native }}));
}
