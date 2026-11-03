//! A media pack's weights as `--w8a8` leaves them resident, from its files alone (MLX-free: a plugin's `config.zig`
//! bills with it): the DiT shards re-billed at their int8 copies.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const w8a8_bill = @import("../blocks/dit/w8a8_bill.zig");
const bill = @import("bill.zig");

const PackDir = abi.PackDir;

const list_cap = 64 << 10;

fn isShard(name: []const u8) bool {
    return std.mem.endsWith(u8, name, ".safetensors");
}

fn join(buf: []u8, folder: []const u8, name: []const u8) ?[]const u8 {
    if (folder.len == 0) return name;
    return std.fmt.bufPrint(buf, "{s}/{s}", .{ folder, name }) catch null;
}

/// A top-level shard that holds a DiT: `transformer*` (Krea, H3, LTX, Music3), ACE-Step's `model` or Hunyuan3D's
/// `dit`.
fn isDitShardName(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "transformer") or std.mem.eql(u8, name, "model.safetensors") or std.mem.eql(u8, name, "dit.safetensors");
}

fn eligible(k: u32, n: u32) bool {
    return w8a8_bill.shapeEligible(k, n);
}

/// Shard `rel` resident under the int8 route; null when its header does not read.
pub fn int8Resident(dir: *const PackDir, rel: []const u8) ?u64 {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const hdr = a.alloc(u8, 64 << 20) catch return null;
    var len: usize = 0;
    if (!dir.safetensors_header(dir.ctx, .of(rel), hdr.ptr, hdr.len, &len).isOk()) return null;
    return w8a8_bill.residentFromHeader(a, hdr[0..len], eligible);
}

/// The DiT's shards (`transformer/*.safetensors`, else the top-level ones `isDitShardName` names): their file
/// bytes, or resident under `w8a8`.
pub fn ditBytes(dir: *const PackDir, w8a8: bool) u64 {
    const folder: []const u8 = if (dir.has("transformer")) "transformer" else "";
    var buf: [list_cap]u8 = undefined;
    var it = dir.entries(folder, &buf) orelse return 0;
    var total: u64 = 0;
    var pbuf: [1024]u8 = undefined;
    while (it.next()) |e| {
        if (!isShard(e) or (folder.len == 0 and !isDitShardName(e))) continue;
        const rel = join(&pbuf, folder, e) orelse continue;
        const size = dir.fileBytes(rel);
        if (size == 0) continue;
        total += if (w8a8) int8Resident(dir, rel) orelse size else size;
    }
    return total;
}

/// The pack's weights (`bill.safetensorsIn`) as a backend whose DiT loaders convert under `--w8a8` holds them
/// (`converts`).
pub fn residentWeights(dir: *const PackDir, m: *const abi.Machine, converts: bool) u64 {
    const sum = bill.safetensorsIn(dir, "");
    if (!converts or !m.w8a8) return sum;
    return sum -| ditBytes(dir, false) + ditBytes(dir, true);
}

// ── tests ──

const testing = std.testing;
const fixture = @import("../testing/fixture.zig");
const tensors = @import("../testing/tensors.zig");

test "pack bytes: a converted DiT bills its int8 copy, the rest of the pack its files" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var dit = tensors.Tensors.init(a, 1);
    try dit.rand("transformer_blocks.0.attn.to_q.weight", .{}, &.{ 64, 256 }, .BF16, 1);
    try dit.rand("x_embedder.weight", .{}, &.{ 8, 8 }, .BF16, 1);
    const dit_bytes = try dit.bytes();
    var vae = tensors.Tensors.init(a, 2);
    try vae.rand("conv.weight", .{}, &.{4}, .F32, 1);
    const vae_bytes = try vae.bytes();
    const mem: fixture.MemDir = .{ .files = &.{
        .{ .name = "transformer/diffusion_pytorch_model.safetensors", .bytes = dit_bytes },
        .{ .name = "vae/a/deep.safetensors", .bytes = vae_bytes },
        .{ .name = "vae/model.safetensors", .bytes = vae_bytes },
        .{ .name = "config.json", .bytes = "{}" },
    } };
    const dir = mem.packDir();
    const files: u64 = dit_bytes.len + vae_bytes.len;
    try testing.expectEqual(@as(u64, dit_bytes.len), ditBytes(&dir, false));
    const off: abi.Machine = .{};
    try testing.expectEqual(files, residentWeights(&dir, &off, true));
    const on: abi.Machine = .{ .w8a8 = true };
    // The 64x256 bf16 projection (32 KiB) becomes 64x256 int8 + 4 bytes per row.
    try testing.expectEqual(files - 64 * 256 * 2 + 64 * 256 + 4 * 64, residentWeights(&dir, &on, true));
    try testing.expectEqual(files, residentWeights(&dir, &on, false));
}
