//! mlx/kernel_cache.
//! Owns QsaCfgCache, KernelCfgCache.

const std = @import("std");
const mlx = @import("ffi.zig");

/// Keyed on everything the config depends on and nothing else: `kv`/`kb` are runtime shapes,
/// `NSPLIT` changes the grid and both output shapes.
/// A small LRU of metal_kernel configs: adaptive MTP alternates verify widths within one
/// request, and a single slot rebuilt on every alternation. Handles are borrowed.
pub fn QsaCfgCache(comptime Key: type, comptime NCFG: usize) type {
    return KernelCfgCache(Key, NCFG, 4);
}

pub fn KernelCfgCache(comptime Key: type, comptime NCFG: usize, comptime slots: usize) type {
    return struct {
        const Self = @This();
        pub const SLOTS: usize = slots;
        /// Named so the default can be written as an array literal.
        const MaybeCfg = ?mlx.mlx_fast_metal_kernel_config;
        const Entry = struct {
            key: Key = std.mem.zeroes(Key),
            cfg: [NCFG]MaybeCfg = @splat(null),
            stamp: u64 = 0,
        };
        entries: [SLOTS]Entry = @splat(.{}),
        clock: u64 = 0,
        /// Build counter.
        builds: usize = 0,

        pub fn get(self: *Self, key: Key) ?[NCFG]mlx.mlx_fast_metal_kernel_config {
            for (&self.entries) |*e| {
                if (e.cfg[0] == null) continue;
                if (!std.meta.eql(e.key, key)) continue;
                self.clock += 1;
                e.stamp = self.clock;
                var out: [NCFG]mlx.mlx_fast_metal_kernel_config = undefined;
                for (e.cfg, 0..) |c, i| out[i] = c.?;
                return out;
            }
            return null;
        }

        pub fn put(self: *Self, key: Key, cfg: [NCFG]mlx.mlx_fast_metal_kernel_config) void {
            var victim: usize = 0;
            var oldest: u64 = std.math.maxInt(u64);
            for (&self.entries, 0..) |*e, i| {
                if (e.cfg[0] == null) {
                    victim = i;
                    oldest = 0;
                    break;
                }
                if (e.stamp < oldest) {
                    oldest = e.stamp;
                    victim = i;
                }
            }
            const e = &self.entries[victim];
            for (&e.cfg) |*c| {
                if (c.*) |cc| _ = mlx.mlx_fast_metal_kernel_config_free(cc);
                c.* = null;
            }
            e.key = key;
            for (cfg, 0..) |c, i| e.cfg[i] = c;
            self.clock += 1;
            e.stamp = self.clock;
            self.builds += 1;
        }
    };
}
