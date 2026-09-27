//! qwen4_exp n-gram PLE on the GPU: the whole `ngram_table.bin` mapping wrapped as ONE no-copy
//! Metal buffer, and one kernel that hashes the token ids, gathers the rows and rounds them to
//! bf16 exactly like `NgramHash.rowIds` + `NgramTable.gather` + `bf16Rne`. The kernel reads the
//! ids on the GPU, so a forward built on lazy ids needs no host read before it runs.

const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const qwen4 = @import("qwen4_exp.zig");

/// Why the loader picked its arm. Everything but `.gpu` serves the host gather.
pub const Arm = enum { gpu, off, bits, misaligned, too_large, low_memory };

/// `--ple-gpu`, set once in `main()`. Off by default: the first GPU forward makes the whole
/// table resident beside the weights, and the host gather faults in only the rows it reads.
pub var enabled: bool = false;

pub const Budget = struct {
    page: usize,
    max_buffer: u64,
    working_set: u64,
    model_bytes: u64,
};

/// Room the working set keeps past weights + table for KV, transients and the buffer pool.
pub const HEADROOM: u64 = 16 << 30;

/// Kernel dispatches since start; the engagement counter the tests read.
pub var dispatches: u64 = 0;
/// Mappings unmapped by MLX dropping their buffer (MLX may drop it off the inference thread).
pub var unmaps = std.atomic.Value(u64).init(0);

pub fn chooseArm(on: bool, bits: u32, base: usize, len: usize, b: Budget) Arm {
    if (!on) return .off;
    if (!kernelBits(bits)) return .bits;
    // MLX falls back to a malloc + copy of the WHOLE table when Metal refuses the no-copy buffer.
    if (b.page == 0 or base % b.page != 0) return .misaligned;
    const bytes = std.mem.alignForward(u64, len, b.page);
    if (bytes > b.max_buffer) return .too_large;
    if (b.model_bytes +| bytes +| HEADROOM > b.working_set) return .low_memory;
    return .gpu;
}

fn kernelBits(bits: u32) bool {
    return switch (bits) {
        2, 3, 4, 5, 6, 8, 16 => true,
        else => false,
    };
}

pub const Table = struct {
    arr: mlx.mlx_array,

    pub fn release(self: Table) void {
        _ = mlx.mlx_array_free(self.arr);
    }
};

/// What MLX's deleter unmaps. `armed` stays false until `wrap` proved the buffer is the
/// mapping: MLX's copy fallback calls the deleter at once, while the host gather still reads it.
const Mapping = struct {
    map: []align(std.heap.page_size_min) const u8,
    armed: bool = false,
    fired: bool = false,
};

fn onRelease(payload: ?*anyopaque) callconv(.c) void {
    const m: *Mapping = @ptrCast(@alignCast(payload.?));
    if (!m.armed) {
        m.fired = true;
        return;
    }
    std.posix.munmap(m.map);
    _ = unmaps.fetchAdd(1, .monotonic);
    std.heap.page_allocator.destroy(m);
}

const ROW: usize = 4096;

/// `table`'s mapping as one no-copy GPU buffer. On success MLX owns the munmap, and the
/// table stops unmapping it on close.
pub fn wrap(table: *qwen4.NgramTable) !Table {
    const t = try wrapMap(table.map);
    table.gpu_owns_map = true;
    return t;
}

/// `map` as one no-copy uint8 `[pages, 4096]` array (no dim past int32). The caller has
/// checked the base and length (`chooseArm`); on success MLX owns the munmap.
fn wrapMap(map: []const u8) !Table {
    const page = std.heap.pageSize();
    if (@intFromPtr(map.ptr) % page != 0) return error.PleMapMisaligned;
    const len = std.mem.alignForward(usize, map.len, page);
    if (len / ROW > std.math.maxInt(c_int)) return error.PleMapTooLarge;
    const m = try std.heap.page_allocator.create(Mapping);
    errdefer std.heap.page_allocator.destroy(m);
    m.* = .{ .map = @alignCast(map) };
    const shape = [_]c_int{ @intCast(len / ROW), @intCast(ROW) };
    const arr = mlx.mlx_array_new_data_managed_payload(@constCast(map.ptr), &shape, 2, .uint8, m, onRelease);
    if (arr.ctx == null) return error.PleWrapFailed;
    if (m.fired) {
        _ = mlx.mlx_array_free(arr);
        return error.PleWrapCopied;
    }
    m.armed = true;
    return .{ .arr = arr };
}

fn gb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1073741824.0;
}

/// Load-time arm choice for `table`; logs one line naming the arm. `on` is `--ple-gpu`,
/// `model_bytes` the weights already resident. Null = the host gather.
pub fn load(table: *qwen4.NgramTable, on: bool, model_bytes: u64) ?Table {
    if (mlx.noGpuBackend()) {
        log.info("[qwen4] ple gather: cpu (no GPU backend)\n", .{});
        return null;
    }
    const b: Budget = .{
        .page = std.heap.pageSize(),
        .max_buffer = mlx.maxBufferLength(),
        .working_set = mlx.maxRecommendedWorkingSet(),
        .model_bytes = model_bytes,
    };
    const arm = chooseArm(on, table.bits, @intFromPtr(table.map.ptr), table.map.len, b);
    if (arm == .off) {
        log.info("[qwen4] ple gather: cpu (--ple-gpu keeps the {d:.1} GB table resident for the GPU gather)\n", .{gb(table.map.len)});
        return null;
    }
    if (arm != .gpu) {
        log.info("[qwen4] ple gather: cpu ({s}: weights {d:.1} GB + table {d:.1} GB + headroom {d:.0} GB vs working set {d:.1} GB, max buffer {d:.1} GB)\n", .{ @tagName(arm), gb(model_bytes), gb(table.map.len), gb(HEADROOM), gb(b.working_set), gb(b.max_buffer) });
        return null;
    }
    const tbl = wrap(table) catch |e| {
        log.warn("[qwen4] ple gather: cpu (no-copy wrap failed: {s})\n", .{@errorName(e)});
        return null;
    };
    log.info("[qwen4] ple gather: gpu (no-copy {d:.1} GB table buffer, weights {d:.1} GB, working set {d:.1} GB)\n", .{ gb(table.map.len), gb(model_bytes), gb(b.working_set) });
    return tbl;
}

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const in_names = [_][*:0]const u8{ "table", "ids", "prev", "params" };
    const out_names = [_][*:0]const u8{"out"};
    const ins = mlx.mlx_vector_string_new_data(&in_names, in_names.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outs = mlx.mlx_vector_string_new_data(&out_names, out_names.len);
    defer _ = mlx.mlx_vector_string_free(outs);
    const k = mlx.mlx_fast_metal_kernel_new("msv_ple_gather", ins, outs, @embedFile("kernels/ple_gather.metal"), @embedFile("kernels/ple_gather_header.metal"), true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// `params` slots the kernel reads; the three tables sit at their `P_*` offsets.
const P_MULT = 13;
const P_VOCAB = P_MULT + qwen4.MAX_NGRAM_SIZE;
const P_OFFSETS = P_VOCAB + qwen4.MAX_HEADS;
const P_LEN = P_OFFSETS + qwen4.MAX_HEADS;

/// bf16 `[S, n_heads * dim]` for the `S` ids of `ids` (any integer dtype, may be lazy),
/// hashed against `prev` (the `ngram_size - 1` tokens before them). GPU stream only.
pub fn embed(s: mlx.mlx_stream, tbl: Table, h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, ids: mlx.mlx_array, prev: []const u32) !mlx.mlx_array {
    std.debug.assert(prev.len == h.ngram_size - 1);
    const n: usize = mlx.mlx_array_size(ids);
    const width: usize = @as(usize, h.n_heads) * t.dim;
    if (n * width > std.math.maxInt(c_int)) return error.PleChunkTooWide;
    var p: [P_LEN]i64 = @splat(0);
    const head = [_]i64{ @intCast(n), h.n_heads, h.heads_per_ngram, h.ngram_size, h.eos, t.dim, t.bits, t.group_size, t.wcols, t.scols, @intCast(t.w_off), @intCast(t.s_off), @intCast(t.b_off) };
    @memcpy(p[0..P_MULT], &head);
    @memcpy(p[P_MULT..P_VOCAB], &h.multipliers);
    @memcpy(p[P_VOCAB..P_OFFSETS], &h.vocab);
    @memcpy(p[P_OFFSETS..P_LEN], &h.offsets);
    const params = mlx.mlx_array_new_data(&p, &[_]c_int{P_LEN}, 1, .int64);
    defer _ = mlx.mlx_array_free(params);
    var prev_i: [qwen4.MAX_NGRAM_SIZE]i32 = undefined;
    for (prev, 0..) |v, k| prev_i[k] = @intCast(v);
    const prev_arr = mlx.mlx_array_new_data(&prev_i, &[_]c_int{@intCast(prev.len)}, 1, .int32);
    defer _ = mlx.mlx_array_free(prev_arr);
    var ids_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_i);
    try mlx.check(mlx.mlx_astype(&ids_i, ids, .int32, s));

    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ @intCast(n), @intCast(width) };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, @intCast(n * width), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    const ins = mlx.mlx_vector_array_new_data(&.{ tbl.arr, ids_i, prev_arr, params }, 4);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try getKernel(), ins, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    dispatches += 1;
    return out;
}

// ── tests ──

const testing = std.testing;

/// A synthetic table file for tests; `deinit` closes the table and removes the file.
pub const Fixture = struct {
    td: std.testing.TmpDir,
    table: qwen4.NgramTable,

    pub fn deinit(self: *Fixture) void {
        self.table.close();
        self.td.cleanup();
    }
};

fn randBf16(r: std.Random) u16 {
    // Normal values only (exponent 2^-17 .. 2^13): real tables carry no NaN, Inf or subnormal.
    const sign: u16 = @as(u16, r.int(u1)) << 15;
    const exp: u16 = r.intRangeAtMost(u16, 110, 140);
    return sign | (exp << 7) | r.int(u7);
}

/// A `ngram_table.bin` with random packed words, scales and biases. The header length is odd so
/// every region starts off a 4-byte boundary, as nothing in the format promises alignment.
pub fn writeFixture(bits: u32, rows: u64, dim: u32, gs: u32, seed: u64) !Fixture {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const a = testing.allocator;
    const hlen: usize = 509;
    const raw = bits == 16;
    const wbytes: u64 = if (raw) rows * dim * 2 else rows * (dim * bits / 32) * 4;
    const sbytes: u64 = if (raw) 0 else rows * (dim / gs) * 2;
    var hbuf: [hlen]u8 = @splat(' ');
    if (raw)
        _ = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"16\",\"group_size\":\"{d}\"}},\"weight\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}}}}", .{ gs, rows, dim, wbytes })
    else
        _ = try std.fmt.bufPrint(&hbuf, "{{\"__metadata__\":{{\"format\":\"mlx-serve-ngram\",\"bits\":\"{d}\",\"group_size\":\"{d}\"}},\"weight\":{{\"dtype\":\"U32\",\"shape\":[{d},{d}],\"data_offsets\":[0,{d}]}},\"scales\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}},\"biases\":{{\"dtype\":\"BF16\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}}}", .{ bits, gs, rows, dim * bits / 32, wbytes, rows, dim / gs, wbytes, wbytes + sbytes, rows, dim / gs, wbytes + sbytes, wbytes + 2 * sbytes });
    const total: usize = @intCast(8 + hlen + wbytes + 2 * sbytes);
    const buf = try a.alloc(u8, total);
    defer a.free(buf);
    std.mem.writeInt(u64, buf[0..8], hlen, .little);
    @memcpy(buf[8 .. 8 + hlen], &hbuf);
    const data = buf[8 + hlen ..];
    if (raw) {
        var i: usize = 0;
        while (i < wbytes) : (i += 2) std.mem.writeInt(u16, data[i..][0..2], randBf16(r), .little);
    } else {
        r.bytes(data[0..@intCast(wbytes)]);
        var i: usize = @intCast(wbytes);
        while (i < data.len) : (i += 2) std.mem.writeInt(u16, data[i..][0..2], randBf16(r), .little);
    }
    var td = std.testing.tmpDir(.{});
    errdefer td.cleanup();
    const io = std.Io.Threaded.global_single_threaded.io();
    try td.dir.writeFile(io, .{ .sub_path = "ngram_table.bin", .data = buf });
    var pbuf: [std.fs.max_path_bytes]u8 = undefined;
    const root_len = try td.dir.realPath(io, &pbuf);
    var full: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&full, "{s}/ngram_table.bin", .{pbuf[0..root_len]});
    qwen4.warm_override = false;
    defer qwen4.warm_override = null;
    return .{ .td = td, .table = try qwen4.NgramTable.open(path) };
}

fn cpuRows(h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, prev: []const u32, ids: []const u32) ![]u16 {
    const a = testing.allocator;
    const rows = try a.alloc(i64, ids.len * h.n_heads);
    defer a.free(rows);
    h.rowIds(prev, ids, rows);
    const host = try a.alloc(f32, rows.len * t.dim);
    defer a.free(host);
    t.gather(rows, host, 0);
    const out = try a.alloc(u16, host.len);
    for (host, out) |v, *o| o.* = qwen4.bf16Rne(v);
    return out;
}

fn gpuRows(tbl: Table, h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, prev: []const u32, ids: []const u32) ![]u16 {
    const a = testing.allocator;
    const ids_i = try a.alloc(i32, ids.len);
    defer a.free(ids_i);
    for (ids, ids_i) |v, *o| o.* = @intCast(v);
    const shape = [_]c_int{@intCast(ids.len)};
    const arr = mlx.mlx_array_new_data(ids_i.ptr, &shape, 1, .int32);
    defer _ = mlx.mlx_array_free(arr);
    const out = try embed(mlx.gpuStream(), tbl, h, t, arr, prev);
    defer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    const want = ids.len * h.n_heads * t.dim;
    try testing.expectEqual(want, mlx.mlx_array_size(out));
    const d = mlx.mlx_array_data_bfloat16(out) orelse return error.MlxArrayDataNull;
    return a.dupe(u16, d[0..want]);
}

/// Both arms over one table; the GPU bits must equal the CPU bits.
fn expectArmsEqual(tbl: Table, h: *const qwen4.NgramHash, t: *const qwen4.NgramTable, prev: []const u32, ids: []const u32) !void {
    const want = try cpuRows(h, t, prev, ids);
    defer testing.allocator.free(want);
    const got = try gpuRows(tbl, h, t, prev, ids);
    defer testing.allocator.free(got);
    try testing.expectEqualSlices(u16, want, got);
}

const TEST_VOCAB: u32 = 1000;
const TEST_EOS: u32 = 999;

fn testHash(ngram: u32, heads: u32) !qwen4.NgramHash {
    return qwen4.NgramHash.init(TEST_VOCAB, ngram, heads, 500, 1, 1234, 0, TEST_EOS);
}

fn randIds(r: std.Random, out: []u32) void {
    for (out) |*v| v.* = r.uintLessThan(u32, TEST_VOCAB);
}

test "ple gpu: 10k random ids embed bit-identical to the CPU gather on every head" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(3, 8);
    var fx = try writeFixture(4, h.total_rows, 64, 32, 1);
    defer fx.deinit();
    const tbl = try wrap(&fx.table);
    defer tbl.release();
    var prng = std.Random.DefaultPrng.init(7);
    const ids = try testing.allocator.alloc(u32, 10_000);
    defer testing.allocator.free(ids);
    randIds(prng.random(), ids);
    const prev = [_]u32{ 17, 404 };
    try expectArmsEqual(tbl, &h, &fx.table, &prev, ids);
}

test "ple gpu: an eos at every chunk position and inside prev hashes like rowIds" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(4, 4);
    var fx = try writeFixture(4, h.total_rows, 64, 32, 2);
    defer fx.deinit();
    const tbl = try wrap(&fx.table);
    defer tbl.release();
    var prng = std.Random.DefaultPrng.init(8);
    const r = prng.random();
    var ids: [8]u32 = undefined;
    var prev: [3]u32 = undefined;
    for (0..prev.len + 1) |pe| {
        randIds(r, &prev);
        if (pe < prev.len) prev[pe] = TEST_EOS;
        for (0..ids.len) |p| {
            randIds(r, &ids);
            ids[p] = TEST_EOS;
            try expectArmsEqual(tbl, &h, &fx.table, &prev, &ids);
        }
    }
    // Back-to-back eos and a fresh (all-eos) history.
    const fresh = [_]u32{ TEST_EOS, TEST_EOS, TEST_EOS };
    const run = [_]u32{ 5, TEST_EOS, TEST_EOS, 7, 8, TEST_EOS, 9, 10 };
    try expectArmsEqual(tbl, &h, &fx.table, &fresh, &run);
}

test "ple gpu: every shipped width dequantizes like the CPU gather" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(3, 8);
    var prng = std.Random.DefaultPrng.init(9);
    var ids: [512]u32 = undefined;
    randIds(prng.random(), &ids);
    const prev = [_]u32{ 3, 4 };
    for ([_]u32{ 2, 3, 4, 5, 6, 8, 16 }) |bits| {
        var fx = try writeFixture(bits, h.total_rows, 64, 32, 100 + bits);
        defer fx.deinit();
        try testing.expectEqual(bits, fx.table.bits);
        const tbl = try wrap(&fx.table);
        defer tbl.release();
        try expectArmsEqual(tbl, &h, &fx.table, &prev, &ids);
    }
}

test "ple gpu: an 8192-token chunk that reaches the last table row matches the CPU gather" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(3, 8);
    var fx = try writeFixture(4, h.total_rows, 64, 32, 3);
    defer fx.deinit();
    const tbl = try wrap(&fx.table);
    defer tbl.release();
    var prng = std.Random.DefaultPrng.init(10);
    const ids = try testing.allocator.alloc(u32, 8192);
    defer testing.allocator.free(ids);
    randIds(prng.random(), ids);
    // Pick the last token so its widest head lands on the table's final row.
    const last_row: i64 = @intCast(h.total_rows - 1);
    var row: [16]i64 = undefined;
    var x: u32 = 0;
    while (true) : (x += 1) {
        h.rowIds(ids[8189..8191], &[_]u32{x}, &row);
        if (row[15] == last_row) break;
    }
    ids[8191] = x;
    try expectArmsEqual(tbl, &h, &fx.table, &[_]u32{ 1, 2 }, ids);
}

test "ple gpu arm gate: off unless --ple-gpu, a width the kernel lacks, a misaligned base, an over-long buffer, a tight working set" {
    const GB: u64 = 1 << 30;
    const b: Budget = .{ .page = 16384, .max_buffer = 64 * GB, .working_set = 200 * GB, .model_bytes = 70 * GB };
    const len: usize = 32 * GB;
    try testing.expectEqual(Arm.gpu, chooseArm(true, 4, 16384 * 7, len, b));
    try testing.expectEqual(Arm.gpu, chooseArm(true, 16, 16384 * 7, len, b));
    try testing.expectEqual(Arm.off, chooseArm(false, 4, 16384 * 7, len, b));
    try testing.expectEqual(Arm.bits, chooseArm(true, 7, 16384 * 7, len, b));
    try testing.expectEqual(Arm.misaligned, chooseArm(true, 4, 16384 * 7 + 4096, len, b));
    try testing.expectEqual(Arm.too_large, chooseArm(true, 4, 16384 * 7, 65 * GB, b));
    // The length rounds up to the page before the maxBufferLength check.
    try testing.expectEqual(Arm.too_large, chooseArm(true, 4, 16384 * 7, 64 * GB - 1, .{ .page = 16384, .max_buffer = 64 * GB - 1, .working_set = 200 * GB, .model_bytes = 0 }));
    var tight = b;
    tight.working_set = 70 * GB + 32 * GB + HEADROOM - 1;
    try testing.expectEqual(Arm.low_memory, chooseArm(true, 4, 16384 * 7, len, tight));
    tight.working_set += 1;
    try testing.expectEqual(Arm.gpu, chooseArm(true, 4, 16384 * 7, len, tight));
    tight.working_set = 0; // an unknown working set never pins 32 GB
    try testing.expectEqual(Arm.low_memory, chooseArm(true, 4, 16384 * 7, len, tight));
}

test "ple gpu wrap: the buffer IS the mapping, and a misaligned base never reaches MLX" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(3, 8);
    var fx = try writeFixture(4, h.total_rows, 64, 32, 4);
    defer fx.deinit();
    const tbl = try wrap(&fx.table);
    defer tbl.release();
    const d = mlx.mlx_array_data_uint8(tbl.arr) orelse return error.MlxArrayDataNull;
    try testing.expectEqual(@intFromPtr(fx.table.map.ptr), @intFromPtr(d));
    try testing.expect(mlx.mlx_array_size(tbl.arr) >= fx.table.map.len);
    try testing.expectError(error.PleMapMisaligned, wrapMap(fx.table.map[16..]));
}

test "ple gpu wrap: the mapping outlives the table until MLX drops its last reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const h = try testHash(3, 8);
    var fx = try writeFixture(4, h.total_rows, 64, 32, 5);
    const tbl = try wrap(&fx.table);
    var extra = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&extra, tbl.arr));
    const before = unmaps.load(.monotonic);
    fx.deinit(); // closes the table: fd and pool go, the mapping stays
    tbl.release();
    try testing.expectEqual(before, unmaps.load(.monotonic));
    _ = mlx.mlx_array_free(extra);
    try testing.expectEqual(before + 1, unmaps.load(.monotonic));
}

test "ple gpu: the real ngram table embeds bit-identical to the CPU gather past 4 GB offsets (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const model_mod = @import("model.zig");
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| a.free(p);
    const h = try qwen4.NgramHash.init(config.vocab_size, config.ngram_size, config.heads_per_ngram, config.ngram_vocab_base, config.ngram_vocab_divisor, config.ngram_seed, 0, config.ngram_eos);
    qwen4.warm_override = false;
    defer qwen4.warm_override = null;
    var table = try qwen4.NgramTable.open(config.ngram_table_path orelse return error.MissingNgramTable);
    // Straight to `wrap`: this bar is the bits, not whether this Mac's working set fits the table.
    const tbl = try wrap(&table);
    defer {
        table.close();
        tbl.release();
    }
    var prng = std.Random.DefaultPrng.init(11);
    const r = prng.random();
    const ids = try a.alloc(u32, 10_000);
    defer a.free(ids);
    for (ids) |*v| v.* = r.uintLessThan(u32, config.vocab_size);
    for (ids[0..64]) |*v| v.* = config.ngram_eos;
    const prev = [_]u32{ r.uintLessThan(u32, config.vocab_size), config.ngram_eos, 7, 7, 7, 7, 7 };
    const rows = try a.alloc(i64, ids.len * h.n_heads);
    defer a.free(rows);
    h.rowIds(prev[0 .. h.ngram_size - 1], ids, rows);
    const row_bytes: u64 = if (table.bits == 16) table.dim * 2 else table.wcols * 4;
    try testing.expect(std.mem.max(i64, rows) * @as(i64, @intCast(row_bytes)) > 1 << 32);
    try expectArmsEqual(tbl, &h, &table, prev[0 .. h.ngram_size - 1], ids);
    try expectArmsEqual(tbl, &h, &table, prev[0 .. h.ngram_size - 1], ids[0..8192]);
}
