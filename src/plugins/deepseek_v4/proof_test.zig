//! Parity proof for the DeepSeek plugins: the table's forward on a tiny random pack equals the deleted legacy forward
//! (`Transformer.forward` through the module seam) at every step, and its verify block the module's own verify
//! primitive at the same state, as recorded in the tapes in `fixtures/` (`engines/mlx/plugin/tape.zig`). Shared by
//! both plugins (`expectSteps`).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const mlx = @import("../../mlx/ffi.zig");
const ops = @import("../../mlx/ops.zig");
const sdk = @import("../../sdk/root.zig");
const fixture = @import("fixture.zig");
const v4 = @import("plugin.zig");
const Tape = @import("../../engines/mlx/plugin/tape.zig").Tape;

const testing = std.testing;

pub const Step = struct { mode: abi.Mode, ids: []const u32 };

/// Prefill, then decodes, a verify block, a later chunk: every arm the module seam takes.
pub const steps = [_]Step{
    .{ .mode = .prefill, .ids = &.{ 3, 17, 42, 9, 7, 25, 1 } },
    .{ .mode = .decode, .ids = &.{8} },
    .{ .mode = .decode, .ids = &.{55} },
    .{ .mode = .verify, .ids = &.{ 5, 61, 20 } },
    .{ .mode = .decode, .ids = &.{12} },
    .{ .mode = .prefill, .ids = &.{ 4, 33, 7, 40 } },
    .{ .mode = .decode, .ids = &.{2} },
};

/// A pack on disk loaded by plugin `P`'s table, its expected values played back from a tape.
pub fn Twin(comptime P: type) type {
    return struct {
        tmp: testing.TmpDir,
        parsed: abi.Parsed = .{},
        m: *anyopaque,
        s: mlx.mlx_stream,
        tape: Tape,

        /// `tape`: the recorded legacy values (empty: a test that compares none).
        pub fn open(files: []const sdk.fixture.File, tape: []const u8) !*@This() {
            const a = testing.allocator;
            const io = std.Io.Threaded.global_single_threaded.io();
            const t = try a.create(@This());
            errdefer a.destroy(t);
            t.tape = if (tape.len > 0) try Tape.player(tape) else .{};
            t.tmp = testing.tmpDir(.{});
            errdefer t.tmp.cleanup();
            for (files) |f| try t.tmp.dir.writeFile(io, .{ .sub_path = f.name, .data = f.bytes });
            var buf: [512]u8 = undefined;
            const dir = buf[0..try t.tmp.dir.realPath(io, &buf)];
            t.s = mlx.gpuStream();
            const mem: sdk.fixture.MemDir = .{ .files = files };
            const pd = mem.packDir();
            try testing.expect(P.table.parse(&pd, .of(files[0].bytes), .{}, null, false, &t.parsed).isOk());
            var why: abi.Verdict = .{};
            t.m = P.table.load(&.{ .model_dir = .of(dir), .dir = &pd, .parsed = &t.parsed, .stream = .{ .ctx = t.s.ctx } }, &why) orelse {
                std.debug.print("load: {s}\n", .{why.why.slice()});
                return error.PluginLoad;
            };
            return t;
        }

        pub fn close(t: *@This()) void {
            P.table.unload(t.m);
            P.table.free_parsed(&t.parsed);
            t.tmp.cleanup();
            testing.allocator.destroy(t);
        }

        /// The plugin's logits for one step at `pos`.
        pub fn plugin(t: *@This(), st: Step, pos: u32) !mlx.mlx_array {
            var out: abi.ForwardOut = .{};
            const v = P.table.forward(t.m, &.{ .mode = st.mode, .logits = if (st.mode == .verify) .all_rows else .last_row, .rows = @intCast(st.ids.len), .tokens = .of(st.ids), .positions = .of(&.{pos}) }, &out);
            if (!v.isOk()) {
                std.debug.print("forward: {s}\n", .{v.why.slice()});
                return error.PluginForward;
            }
            return .{ .ctx = out.logits.ctx };
        }

        /// The plugin's `got` (freed here) against the next recorded value.
        pub fn check(t: *@This(), got: mlx.mlx_array) !void {
            defer _ = mlx.mlx_array_free(got);
            try t.tape.expectNext(got, false);
        }
    };
}

/// Runs `steps` against the tape.
pub fn expectSteps(comptime P: type, t: *Twin(P)) !void {
    var pos: u32 = 0;
    for (steps) |st| {
        try t.check(try t.plugin(st, pos));
        pos += @intCast(st.ids.len);
    }
    try t.tape.expectDone();
}

test "deepseek_v4 plugin: prefill, decode (lazy on the GPU), a verify block and a later chunk equal the legacy seam bit for bit" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(v4).open(try fixture.v4Files(arena.allocator(), 11), @embedFile("fixtures/steps.tape"));
    defer t.close();
    try expectSteps(v4, t);
}

test "deepseek_v4 plugin: a position that does not follow the model's state is refused by name" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(v4).open(try fixture.v4Files(arena.allocator(), 11), "");
    defer t.close();
    const l = try t.plugin(.{ .mode = .prefill, .ids = &.{ 1, 2, 3 } }, 0);
    _ = mlx.mlx_array_free(l);
    var out: abi.ForwardOut = .{};
    const v = v4.table.forward(t.m, &.{ .mode = .decode, .rows = 1, .tokens = .of(&.{4}), .positions = .of(&.{5}) }, &out);
    try testing.expect(!v.isOk() and out.logits.ctx == null);
}

/// A decode step whose token arrives as a device array (`Forward.ids`), as the pipelined decode hands it.
pub fn expectDeviceDecode(comptime P: type, t: *Twin(P)) !void {
    const first = [_]u32{ 3, 17, 42, 9, 7 };
    const a = try t.plugin(.{ .mode = .prefill, .ids = &first }, 0);
    try t.check(a);
    var pos: u32 = first.len;
    for ([_]u32{ 8, 55, 12 }) |id| {
        const dev = mlx.mlx_array_new_data(&id, &[_]c_int{ 1, 1 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(dev);
        var out: abi.ForwardOut = .{};
        const v = P.table.forward(t.m, &.{ .mode = .decode, .rows = 1, .tokens = .{}, .ids = .{ .ctx = dev.ctx }, .positions = .of(&.{pos}) }, &out);
        try testing.expect(v.isOk());
        try t.check(.{ .ctx = out.logits.ctx });
        pos += 1;
    }
    try t.tape.expectDone();
}

test "deepseek_v4 plugin: a decode token on the device equals the legacy decode" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const t = try Twin(v4).open(try fixture.v4Files(arena.allocator(), 11), @embedFile("fixtures/device-decode.tape"));
    defer t.close();
    try expectDeviceDecode(v4, t);
}
