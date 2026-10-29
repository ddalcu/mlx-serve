//! D1's claims, caps and bills, MLX-free: an LFM2-VL `config.json` whose `auto_map` names the card's own
//! `modeling_d1.D1Model` (what tells it from a chat LFM2.5-VL; a marker claim outranks lfm2_vl's type claim), the
//! decision route, the weights resident.

const std = @import("std");
const abi = @import("../../abi/media.zig");
const bill = @import("../../sdk/media/bill.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "d1",
    .kind = .media,
    .claims = &model_types,
    .gate_models = &.{"LiquidAI/d1-3B"},
};

pub const model_types = [_][]const u8{"d1"};

pub fn claims(model_type: abi.Str, dir: *const abi.PackDir) callconv(.c) abi.Claim {
    _ = model_type;
    var buf: [64 << 10]u8 = undefined;
    var why: abi.Verdict = .{};
    const cfg = dir.readFile("config.json", &buf, &why) orelse return .no;
    var fba_buf: [256 << 10]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&fba_buf);
    const v = std.json.parseFromSliceLeaky(std.json.Value, fba.allocator(), cfg, .{}) catch return .no;
    if (v != .object) return .no;
    const map = v.object.get("auto_map") orelse return .no;
    if (map != .object) return .no;
    const cls = map.object.get("AutoModel") orelse return .no;
    return if (cls == .string and std.mem.eql(u8, cls.string, "modeling_d1.D1Model")) .by_marker else .no;
}

/// No completeness marker: the pack counts as discovered (legacy had none).
pub const required_file = "";

pub fn caps(dir: *const abi.PackDir, out: *abi.MediaCaps) callconv(.c) void {
    _ = dir;
    out.* = .{};
}

pub fn billLoad(dir: *const abi.PackDir, m: *const abi.Machine) callconv(.c) u64 {
    _ = m;
    return bill.safetensorsIn(dir, "");
}

pub fn billRequest(dir: *const abi.PackDir, m: *const abi.Machine, body: abi.Str) callconv(.c) u64 {
    _ = .{ dir, m, body };
    return 0;
}
