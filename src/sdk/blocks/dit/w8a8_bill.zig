//! The W8A8 route's resident bill from a safetensors header, MLX-free (a media plugin's `config.zig` bills with it):
//! which linears the DiT loaders convert, and a shard's bytes once each converted linear is re-billed at its int8
//! copy (N*K + 4N), its packed originals released.

const std = @import("std");

/// The rotation block K must divide and the store tile N must divide.
pub const block_k: u32 = 256;
pub const tile_n: u32 = 64;

/// Group size read for quantized weights when billing: every shipped media pack is g64
/// (another group mis-bills K by the ratio).
const BILL_GROUP: u64 = 64;

/// The weight shape the int8 kernels take (the hardware half of eligibility is the caller's).
pub fn shapeEligible(k: u32, n: u32) bool {
    return k % block_k == 0 and n % tile_n == 0;
}

/// The block attention and MLP linears of every DiT the loaders convert (Krea `blocks.N.attn.wq`,
/// Qwen-Image and MageFlow `transformer_blocks.N.img_mlp...`, H3 `blocks.N.mlp.fc1` and its token
/// refiner, FLUX.2 `transformer_blocks.N.ff.linear_in` and `single_transformer_blocks.N.attn.to_out`,
/// LTX `transformer.transformer_blocks.N.audio_attn1.to_q`);
/// modulation, embedders and Krea's text-fusion stack stay packed.
pub fn converted(name: []const u8) bool {
    const stacks = [_][]const u8{ "blocks.", "transformer_blocks.", "single_transformer_blocks.", "transformer.transformer_blocks.", "token_refiner.blocks.", "decoder.layers." };
    const in_stack = for (stacks) |p| {
        if (std.mem.startsWith(u8, name, p)) break true;
    } else false;
    if (!in_stack) return false;
    const stems = [_][]const u8{ "attn", ".mlp.", ".img_mlp.", ".txt_mlp.", ".ff.", "_ff.", ".ff_context.", ".ff_", ".shared.", ".skip." };
    for (stems) |t| if (std.mem.indexOf(u8, name, t) != null) return true;
    return false;
}

fn tensorBytes(v: std.json.Value) u64 {
    const off = (if (v == .object) v.object.get("data_offsets") else null) orelse return 0;
    if (off != .array or off.array.items.len != 2 or off.array.items[0] != .integer or off.array.items[1] != .integer) return 0;
    return @intCast(@max(off.array.items[1].integer - off.array.items[0].integer, 0));
}

fn shape2(v: std.json.Value) ?[2]u64 {
    const sh = (if (v == .object) v.object.get("shape") else null) orelse return null;
    if (sh != .array or sh.array.items.len != 2) return null;
    for (sh.array.items) |d| if (d != .integer or d.integer <= 0) return null;
    return .{ @intCast(sh.array.items[0].integer), @intCast(sh.array.items[1].integer) };
}

/// The int8 copy's bytes for the 2-D weight `wkey` when the loaders would convert it.
fn int8Bytes(h: std.json.ObjectMap, wkey: []const u8, a: std.mem.Allocator, eligible: *const fn (u32, u32) bool) ?u64 {
    if (!converted(wkey)) return null;
    const w = h.get(wkey) orelse return null;
    const wsh = shape2(w) orelse return null;
    const skey = std.fmt.allocPrint(a, "{s}.scales", .{wkey[0 .. wkey.len - 7]}) catch return null;
    const k: u64 = if (h.get(skey)) |sc| (shape2(sc) orelse return null)[1] * BILL_GROUP else blk: {
        const dt = (if (w == .object) w.object.get("dtype") else null) orelse return null;
        if (dt != .string or !(std.mem.eql(u8, dt.string, "BF16") or std.mem.eql(u8, dt.string, "F16") or std.mem.eql(u8, dt.string, "F32"))) return null;
        break :blk wsh[1];
    };
    if (!eligible(@intCast(k), @intCast(wsh[0]))) return null;
    return wsh[0] * k + 4 * wsh[0];
}

/// A shard's resident bytes from its JSON header (`header_len` = the header's byte length): 8 + header + every
/// tensor, a converted linear at its int8 copy. `eligible(k, n)` is the route's per-linear answer. Null when the
/// header does not parse.
pub fn residentFromHeader(a: std.mem.Allocator, header: []const u8, eligible: *const fn (u32, u32) bool) ?u64 {
    const h = std.json.parseFromSliceLeaky(std.json.Value, a, header, .{}) catch return null;
    if (h != .object) return null;
    var total: u64 = 8 + header.len;
    var it = h.object.iterator();
    while (it.next()) |e| {
        const key = e.key_ptr.*;
        if (std.mem.eql(u8, key, "__metadata__")) continue;
        if (std.mem.endsWith(u8, key, ".weight")) {
            total += int8Bytes(h.object, key, a, eligible) orelse tensorBytes(e.value_ptr.*);
        } else if (std.mem.endsWith(u8, key, ".scales") or std.mem.endsWith(u8, key, ".biases")) {
            const base = std.fmt.allocPrint(a, "{s}.weight", .{key[0 .. key.len - 7]}) catch return null;
            if (int8Bytes(h.object, base, a, eligible) == null) total += tensorBytes(e.value_ptr.*);
        } else total += tensorBytes(e.value_ptr.*);
    }
    return total;
}
