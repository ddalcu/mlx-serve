//! The Engram table reader: on-disk tables (~100 GB a layer or external bin)
//! read per lookup from disk, never loaded into memory.

const std = @import("std");
const builtin = @import("builtin");
const crt = @import("../util/crt.zig");
const mlx = @import("../../mlx/ffi.zig");
const QuantMode = @import("../quant/params.zig").QuantMode;
const gguf_root = @import("../quant/gguf/root.zig");

/// One layer's table, read per lookup from disk: mxfp8 records (code bytes,
/// then one e8m0 scale per 32), or an MLX affine table (weight / scales / biases).
pub const Table = struct {
    fd: c_int,
    rows: u64,
    head_dim: u32,
    mode: QuantMode,
    bits: u32,
    group_size: u32,
    w_off: u64,
    w_stride: u64,
    wlen: usize,
    s_off: u64,
    s_stride: u64,
    slen: usize,
    b_off: u64 = 0,
    in_shards: bool = false,

    /// Rows `ids` as they sit on disk (host only).
    pub const Raw = struct {
        w: []align(4) u8,
        s: []align(4) u8,
        b: []align(4) u8,

        pub fn deinit(self: Raw, gpa: std.mem.Allocator) void {
            gpa.free(self.w);
            gpa.free(self.s);
            gpa.free(self.b);
        }
    };

    pub fn close(self: *Table) void {
        crt.close(self.fd);
    }

    /// Bytes this table occupies inside the pack's billed safetensors shards.
    pub fn shardBytes(self: *const Table) u64 {
        if (!self.in_shards) return 0;
        return self.rows * (self.w_stride + self.s_stride * @as(u64, if (self.mode == .affine) 2 else 1));
    }

    /// Reads rows `ids` from disk into a `Raw` buffer using pread.
    pub fn read(self: *const Table, gpa: std.mem.Allocator, ids: []const u32) !Raw {
        const n = ids.len;
        const w = try gpa.alignedAlloc(u8, .@"4", n * self.wlen);
        errdefer gpa.free(w);
        const sc = try gpa.alignedAlloc(u8, .@"4", n * self.slen);
        errdefer gpa.free(sc);
        const bi = try gpa.alignedAlloc(u8, .@"4", if (self.mode == .affine) n * self.slen else 0);
        errdefer gpa.free(bi);

        for (ids, 0..) |row_u32, i| {
            const r: u64 = row_u32;
            if (r >= self.rows) return error.EngramRowOutOfRange;
            if (crt.pread(self.fd, w[i * self.wlen ..][0..self.wlen].ptr, self.wlen, @intCast(self.w_off + r * self.w_stride)) != @as(isize, @intCast(self.wlen))) {
                return error.EngramRead;
            }
            if (self.slen > 0) {
                if (crt.pread(self.fd, sc[i * self.slen ..][0..self.slen].ptr, self.slen, @intCast(self.s_off + r * self.s_stride)) != @as(isize, @intCast(self.slen))) {
                    return error.EngramRead;
                }
            }
            if (self.mode == .affine and self.slen > 0) {
                if (crt.pread(self.fd, bi[i * self.slen ..][0..self.slen].ptr, self.slen, @intCast(self.b_off + r * self.s_stride)) != @as(isize, @intCast(self.slen))) {
                    return error.EngramRead;
                }
            }
        }
        return .{ .w = w, .s = sc, .b = bi };
    }

    /// The first `n` rows of `raw` dequantized: f32 `[n, head_dim]`.
    pub fn dequant(self: *const Table, raw: Raw, n: usize, s: mlx.mlx_stream) !mlx.mlx_array {
        if (self.mode == .gguf) {
            const kernels = gguf_root.kernels;
            const rb = kernels.rowBytes(.iq4_nl, self.head_dim);
            const tmp = try std.heap.page_allocator.alloc(u8, n * rb);
            defer std.heap.page_allocator.free(tmp);
            kernels.packRows(.iq4_nl, self.head_dim, tmp, raw.w[0 .. n * self.wlen]);
            const wa = mlx.mlx_array_new_data(tmp.ptr, &[_]c_int{ @intCast(n), @intCast(rb) }, 2, .uint8);
            defer _ = mlx.mlx_array_free(wa);
            return kernels.dequantize(.iq4_nl, wa, .float32, s);
        }
        const wcols: c_int = @intCast(self.wlen / 4);
        const scols: c_int = @intCast(if (self.mode == .affine) self.slen / 2 else self.slen);
        const rows: c_int = @intCast(n);
        const wa = mlx.mlx_array_new_data(raw.w.ptr, &[_]c_int{ rows, wcols }, 2, .uint32);
        defer _ = mlx.mlx_array_free(wa);
        const sdt: mlx.mlx_dtype = if (self.mode == .affine) .bfloat16 else .uint8;
        const sa = mlx.mlx_array_new_data(raw.s.ptr, &[_]c_int{ rows, scols }, 2, sdt);
        defer _ = mlx.mlx_array_free(sa);
        const ba: mlx.mlx_array = if (self.mode == .affine) mlx.mlx_array_new_data(raw.b.ptr, &[_]c_int{ rows, scols }, 2, .bfloat16) else .{ .ctx = null };
        defer if (ba.ctx != null) {
            _ = mlx.mlx_array_free(ba);
        };
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_dequantize(&out, wa, sa, ba, mlx.mlx_optional_int.some(@intCast(self.group_size)), mlx.mlx_optional_int.some(@intCast(self.bits)), self.mode.cstr(), .{ .ctx = null }, .{ .value = .float32, .has_value = true }, s));
        return out;
    }

    /// Rows `ids`, dequantized: f32 `[ids.len, head_dim]`.
    pub fn gather(self: *const Table, gpa: std.mem.Allocator, ids: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
        const raw = try self.read(gpa, ids);
        defer raw.deinit(gpa);
        return self.dequant(raw, ids.len, s);
    }
};

const testing = std.testing;

test "engram table: affine gather dequantizes rows correctly" {
    const io = std.Io.Threaded.global_single_threaded.io();
    var td = testing.tmpDir(.{});
    defer td.cleanup();
    // 4 rows, head_dim 32: weight U32 [4,4], scales/biases BF16 [4,1].
    const hdr = "{\"weight\":{\"dtype\":\"U32\",\"shape\":[4,4],\"data_offsets\":[0,64]}," ++
        "\"scales\":{\"dtype\":\"BF16\",\"shape\":[4,1],\"data_offsets\":[64,72]}," ++
        "\"biases\":{\"dtype\":\"BF16\",\"shape\":[4,1],\"data_offsets\":[72,80]}}";
    const pad = (8 - hdr.len % 8) % 8;
    var bin: [8 + hdr.len + pad + 80]u8 = @splat(0);
    std.mem.writeInt(u64, bin[0..8], hdr.len + pad, .little);
    @memcpy(bin[8..][0..hdr.len], hdr);
    @memset(bin[8 + hdr.len ..][0..pad], ' ');
    const data = bin[8 + hdr.len + pad ..];
    for (data[0..64]) |*x| x.* = 0x11; // all nibbles = 1
    for (0..4) |r| {
        std.mem.writeInt(u16, data[64 + 2 * r ..][0..2], 0x4000, .little); // scale 2
        std.mem.writeInt(u16, data[72 + 2 * r ..][0..2], 0x3f00, .little); // bias 0.5
    }
    try td.dir.writeFile(io, .{ .sub_path = "table.bin", .data = &bin });

    var buf: [512]u8 = undefined;
    const dir = buf[0..try td.dir.realPath(io, &buf)];
    const path = try std.fs.path.join(testing.allocator, &.{ dir, "table.bin" });
    defer testing.allocator.free(path);

    const fd = crt.openRead(@ptrCast(path.ptr));
    defer crt.close(fd);
    try testing.expect(fd >= 0);

    const base: u64 = 8 + hdr.len + pad;
    const t: Table = .{
        .fd = fd,
        .rows = 4,
        .head_dim = 32,
        .mode = .affine,
        .bits = 4,
        .group_size = 32,
        .w_off = base,
        .w_stride = 16,
        .wlen = 16,
        .s_off = base + 64,
        .s_stride = 2,
        .slen = 2,
        .b_off = base + 72,
    };

    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    const got = try t.gather(testing.allocator, &.{ 3, 0 }, s);
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    const v = mlx.mlx_array_data_float32(got).?;
    for (0..64) |i| try testing.expectEqual(@as(f32, 2.5), v[i]); // 1*2 + 0.5
}
