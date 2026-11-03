//! A media plugin's fixture staged on disk and loaded through its table on the CPU stream, for plugin tests: run a
//! job and read what `encode` returns.

const std = @import("std");
const media_abi = @import("../../abi/media.zig");
const fixture = @import("fixture.zig");
const mlx = @import("../../mlx/ffi.zig");

const testing = std.testing;

/// Whether this machine has less RAM than plugin root `M`'s fixture job needs (`pub const test_min_ram_gib`); says so
/// by name when it does, so the caller skips.
pub fn ramShort(comptime M: type) bool {
    if (!@hasDecl(M, "test_min_ram_gib")) return false;
    const ram = std.process.totalSystemMemory() catch return false;
    if (ram >= @as(u64, M.test_min_ram_gib) << 30) return false;
    std.debug.print("skip: {s}'s fixture job needs {d} GiB of RAM, this machine has {d}\n", .{ M.plugin.name, M.test_min_ram_gib, ram >> 30 });
    return true;
}

pub fn MediaPack(comptime M: type) type {
    return struct {
        tmp: testing.TmpDir,
        mem: fixture.MemDir,
        stream: mlx.mlx_stream,
        engine: *anyopaque,
        fx: fixture.Fixture,

        /// `a` must outlive the pack (an arena).
        pub fn open(a: std.mem.Allocator) !@This() {
            if (ramShort(M)) return error.SkipZigTest;
            const fx = try M.conformanceFixture(a);
            var tmp = testing.tmpDir(.{});
            errdefer tmp.cleanup();
            for (fx.files) |f| {
                if (std.fs.path.dirname(f.name)) |d| try tmp.dir.createDirPath(testing.io, d);
                try tmp.dir.writeFile(testing.io, .{ .sub_path = f.name, .data = f.bytes });
            }
            const buf = try a.alloc(u8, std.fs.max_path_bytes);
            const path = buf[0..try tmp.dir.realPath(testing.io, buf)];
            var self: @This() = .{ .tmp = tmp, .mem = .{ .files = fx.files }, .stream = mlx.mlx_default_cpu_stream_new(), .engine = undefined, .fx = fx };
            const dir = self.mem.packDir();
            var why: media_abi.Verdict = .{};
            self.engine = M.table.load(&.{ .model_dir = .of(path), .dir = &dir, .stream = .{ .ctx = self.stream.ctx } }, &why) orelse {
                std.debug.print("media pack {s}: load: {s}\n", .{ M.plugin.name, why.why.slice() });
                _ = mlx.mlx_stream_free(self.stream);
                return error.LoadFailed;
            };
            return self;
        }

        pub fn close(self: *@This()) void {
            M.table.unload(self.engine);
            _ = mlx.mlx_stream_free(self.stream);
            self.tmp.cleanup();
        }

        /// One job on `body`; the encoded result (owned by the plugin until its next job), or the refusal.
        pub fn run(self: *@This(), body: []const u8) !Outcome {
            return self.runInputs(body, &.{});
        }

        pub const Outcome = union(enum) { ok: []const u8, refused: media_abi.Verdict };

        /// A job with uploads (`media.Request.inputs`).
        pub fn runInputs(self: *@This(), body: []const u8, inputs: []const media_abi.Str) !Outcome {
            const Quiet = struct {
                fn report(_: ?*anyopaque, _: media_abi.Str, _: u32, _: u32) callconv(.c) void {}
                fn boundary(_: ?*anyopaque) callconv(.c) bool {
                    return false;
                }
            };
            const progress: media_abi.Progress = .{ .ctx = null, .report = Quiet.report, .boundary = Quiet.boundary };
            const dir = self.mem.packDir();
            const req: media_abi.Request = .{ .body = .of(body), .inputs = .{ .ptr = inputs.ptr, .len = inputs.len }, .dir = &dir };
            const v = M.table.step(self.engine, &req, &progress);
            if (!v.isOk()) return .{ .refused = v };
            var out: media_abi.Str = .{};
            const e = M.table.encode(self.engine, &out);
            if (!e.isOk()) return .{ .refused = e };
            return .{ .ok = out.slice() };
        }
    };
}
