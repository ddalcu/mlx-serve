const std = @import("std");
const testing = std.testing;

pub const Scheme = enum { off, affine };

pub const KVQuantConfig = struct {
    scheme: Scheme,
    /// Affine: 4 or 8. Ignored when `scheme == .off`.
    bits: u8,
    /// Affine group size — number of consecutive elements that share one
    /// scale+bias pair along the last axis. mlx-c convention is 64 for
    /// 4-bit and 8-bit weights; we match that.
    group_size: u32,

    pub const dense: KVQuantConfig = .{ .scheme = .off, .bits = 0, .group_size = 0 };

    pub fn affine(bits: u8) KVQuantConfig {
        std.debug.assert(bits == 4 or bits == 8);
        return .{ .scheme = .affine, .bits = bits, .group_size = 64 };
    }

    pub fn isQuant(self: KVQuantConfig) bool {
        return self.scheme != .off;
    }

    /// The wire vocabulary shared by the per-request `kv_quant` body field and
    /// `model-settings.json`: "off"/0, "4", "8". Null = unrecognized.
    pub fn fromJsonValue(v: std.json.Value) ?KVQuantConfig {
        switch (v) {
            .string => |s| {
                if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "0")) return dense;
                if (std.mem.eql(u8, s, "4")) return affine(4);
                if (std.mem.eql(u8, s, "8")) return affine(8);
                return null;
            },
            .integer => |i| {
                if (i == 0) return dense;
                if (i == 4) return affine(4);
                if (i == 8) return affine(8);
                return null;
            },
            else => return null,
        }
    }

    /// The same vocabulary, for reporting (`/v1/models` `meta.kv_quant`).
    pub fn wireName(self: KVQuantConfig) []const u8 {
        return switch (self.scheme) {
            .off => "off",
            .affine => if (self.bits == 4) "4" else "8",
        };
    }
};

test "KVQuantConfig.affine builds a sane config" {
    const c4 = KVQuantConfig.affine(4);
    try testing.expectEqual(Scheme.affine, c4.scheme);
    try testing.expectEqual(@as(u8, 4), c4.bits);
    try testing.expectEqual(@as(u32, 64), c4.group_size);

    const c8 = KVQuantConfig.affine(8);
    try testing.expectEqual(@as(u8, 8), c8.bits);

    const cd = KVQuantConfig.dense;
    try testing.expectEqual(Scheme.off, cd.scheme);
}
