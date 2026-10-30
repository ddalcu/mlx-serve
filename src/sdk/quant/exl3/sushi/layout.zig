//! Rate-grouped pack layout: `.gN` tensor names and the per-group shapes they must agree on.
const std = @import("std");
const format = @import("expert_exl3.zig");

pub const max_groups = 32;
pub const Dtype = enum { u16, f16, other };
pub const Name = struct {
    group: u32,
    projection: u32,
    part: u32,
    grouped: bool,

    pub fn parse(name: []const u8) !Name {
        var it = std.mem.splitScalar(u8, name, '.');
        const proj = it.next() orelse return error.Exl3GroupNameInvalid;
        const p: u32 = if (std.mem.eql(u8, proj, "gate_proj")) 0 else if (std.mem.eql(u8, proj, "up_proj")) 1 else if (std.mem.eql(u8, proj, "down_proj")) 2 else return error.Exl3GroupNameInvalid;
        var leaf = it.next() orelse return error.Exl3GroupNameInvalid;
        var group: u32 = 0;
        const grouped = std.mem.startsWith(u8, leaf, "g");
        if (grouped) {
            if (leaf.len < 2 or (leaf.len > 2 and leaf[1] == '0')) return error.Exl3GroupNameInvalid;
            for (leaf[1..]) |c| if (c < '0' or c > '9') return error.Exl3GroupNameInvalid;
            group = std.fmt.parseInt(u32, leaf[1..], 10) catch return error.Exl3GroupNameInvalid;
            if (group >= max_groups) return error.Exl3GroupNameInvalid;
            leaf = it.next() orelse return error.Exl3GroupNameInvalid;
        }
        const part: u32 = if (std.mem.eql(u8, leaf, "trellis")) 0 else if (std.mem.eql(u8, leaf, "suh")) 1 else if (std.mem.eql(u8, leaf, "svh")) 2 else return error.Exl3GroupNameInvalid;
        if (it.next() != null) return error.Exl3GroupNameInvalid;
        return .{ .group = group, .projection = p, .part = part, .grouped = grouped };
    }
};

pub const Group = struct {
    experts: u32 = 0,
    rates: [3]format.Rate = @splat(.{ .n = 0 }),
    mask: u16 = 0,
};

pub const Layout = struct {
    hidden: u32,
    intermediate: u32,
    ceiling: format.Rate,
    /// The bank rates the pack's codebook table has a row for (`Spec.rates`); null admits any up to `ceiling`.
    rates: ?u128 = null,
    groups: [max_groups]Group = @splat(.{}),
    count: usize = 0,
    grouped: ?bool = null,
    /// The streaming engine serves rate groups (pruned layers still refuse it).
    streams_groups: bool = false,
    bytes: u64 = 0,

    pub fn init(hidden: u32, intermediate: u32, ceiling: format.Rate) Layout {
        return .{ .hidden = hidden, .intermediate = intermediate, .ceiling = ceiling };
    }

    pub fn add(self: *Layout, suffix: []const u8, shape: []const u64, dtype: Dtype) !Name {
        const name = try Name.parse(suffix);
        if (self.grouped) |g| if (g != name.grouped) return error.Exl3MixedGroupLayout;
        if (dtype != (if (name.part == 0) Dtype.u16 else Dtype.f16)) return error.Exl3GroupDtype;
        const input = if (name.projection == 2) self.intermediate else self.hidden;
        const output = if (name.projection == 2) self.hidden else self.intermediate;
        if (input == 0 or output == 0 or input % 128 != 0 or output % 128 != 0) return error.Exl3GroupGeometry;
        if (shape.len != (if (name.part == 0) @as(usize, 4) else 2)) return error.Exl3GroupGeometry;
        if (shape[0] == 0 or shape[0] > std.math.maxInt(c_int)) return error.Exl3GroupGeometry;
        const group = &self.groups[name.group];
        if (group.experts != 0 and group.experts != shape[0]) return error.Exl3GroupGeometry;
        var rate: format.Rate = .{ .n = 0 };
        if (name.part == 0) {
            if (shape[1] != input / 16 or shape[2] != output / 16) return error.Exl3GroupGeometry;
            rate = format.kFromPackedDim(shape[3]) orelse return error.Exl3TrellisGeometry;
            if (rate.n > self.ceiling.n) return error.Exl3TrellisGeometry;
            if (!rate.in(self.rates)) return error.Exl3CodebookTableRateMissing;
        } else if (shape[1] != (if (name.part == 1) input else output)) return error.Exl3GroupGeometry;
        const bit = @as(u16, 1) << @intCast(name.projection * 3 + name.part);
        if (group.mask & bit != 0) return error.Exl3GroupNameInvalid;
        var bytes: u64 = 2;
        for (shape) |d| bytes = std.math.mul(u64, bytes, d) catch return error.ResidentBytesOverflow;
        self.bytes = std.math.add(u64, self.bytes, bytes) catch return error.ResidentBytesOverflow;
        self.grouped = name.grouped;
        self.count = @max(self.count, name.group + 1);
        group.experts = @intCast(shape[0]);
        group.mask |= bit;
        if (name.part == 0) group.rates[name.projection] = rate;
        return name;
    }

    pub fn finish(self: *const Layout, router_experts: u32, topk: u32, streaming: bool, configured_experts: u32) !void {
        if (self.count == 0) return error.MissingWeight;
        var total: u64 = 0;
        for (self.groups[0..self.count]) |g| {
            if (g.mask != 511) return if (self.grouped.?) error.Exl3GroupMissing else error.MissingWeight;
            total += g.experts;
        }
        if (total != router_experts or total > std.math.maxInt(c_int)) return error.Exl3RouterWidthMismatch;
        if (topk > router_experts) return error.Exl3TopKExceedsExperts;
        if (topk > 32) return error.Exl3TopKExceedsReduceBank;
        if (streaming and ((self.grouped.? and !self.streams_groups) or router_experts != configured_experts)) return error.Exl3RaggedStreamingUnsupported;
    }

    pub fn locate(self: *const Layout, expert: u32) !struct { group: u32, local: u32 } {
        var offset: u32 = 0;
        for (self.groups[0..self.count], 0..) |g, i| {
            if (expert >= offset and expert - offset < g.experts) return .{ .group = @intCast(i), .local = expert - offset };
            offset += g.experts;
        }
        return error.ExpertOutOfRange;
    }
};
