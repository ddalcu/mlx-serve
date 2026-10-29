//! Which tensors one cluster rank loads: a pipeline rank's layer span, a tensor-parallel rank's expert slice.

const std = @import("std");

/// The trunk layers one pipeline rank holds: `[start, end)` of `total`. The rank
/// loads them renumbered from 0, so its config, caches and bills are those of a
/// short model; the hidden stream crosses ranks at `end`.
pub const LayerSpan = struct {
    start: u32,
    end: u32,
    total: u32,

    pub fn len(self: LayerSpan) u32 {
        return self.end - self.start;
    }

    /// Rank 0 embeds the tokens, projects the logits and runs the MTP head and vision
    /// tower; a later rank holds only its trunk layers.
    pub fn first(self: LayerSpan) bool {
        return self.start == 0;
    }

    /// The name tensor `key` loads under on this rank, or null when it belongs to
    /// another: `key` itself, or the renumbered trunk name written into `buf`.
    pub fn remap(self: LayerSpan, key: []const u8, buf: []u8) ?[]const u8 {
        if (headOnly(key)) return if (self.first()) key else null;
        const at = trunkLayerAt(key) orelse return key;
        const n = std.fmt.parseInt(u32, key[at.lo..at.hi], 10) catch return key;
        if (n < self.start or n >= self.end) return null;
        if (self.start == 0) return key;
        return std.fmt.bufPrint(buf, "{s}{d}{s}", .{ key[0..at.lo], n - self.start, key[at.hi..] }) catch null;
    }
};

/// Tensor-parallel rank: its slice of every MoE expert's intermediate width (gate/up
/// rows, down columns), the routed experts and the shared one alike.
pub const Shard = struct {
    rank: u32,
    size: u32,

    /// Which projection of an expert `key` names: gate/up split their rows, down its columns.
    pub fn expertAxis(key: []const u8) ?enum { rows, cols } {
        if (headOnly(key)) return null;
        if (std.mem.indexOf(u8, key, ".switch_mlp.") == null and std.mem.indexOf(u8, key, ".shared_expert.") == null) return null;
        const tail = for ([_][]const u8{ ".weight", ".scales", ".biases" }) |sfx| {
            if (std.mem.endsWith(u8, key, sfx)) break key[0 .. key.len - sfx.len];
        } else return null;
        if (std.mem.endsWith(u8, tail, ".gate_proj") or std.mem.endsWith(u8, tail, ".up_proj")) return .rows;
        if (std.mem.endsWith(u8, tail, ".down_proj")) return .cols;
        return null;
    }

    /// Copies this rank's part of axis `mid` of a row-major `[outer, mid, inner]` buffer.
    pub fn slice(self: Shard, out: []u8, buf: []const u8, outer: usize, mid: usize, inner: usize) void {
        const part = mid / self.size;
        for (0..outer) |o| {
            const src = (o * mid + self.rank * part) * inner;
            @memcpy(out[o * part * inner ..][0 .. part * inner], buf[src..][0 .. part * inner]);
        }
    }
};

/// Tensors only rank 0 reads: the MTP head and the vision/audio towers.
pub fn headOnly(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "mtp.") or std.mem.indexOf(u8, key, ".mtp.") != null or
        std.mem.indexOf(u8, key, "visual") != null or std.mem.indexOf(u8, key, "vision") != null or
        std.mem.indexOf(u8, key, "audio") != null;
}

/// The digits of the first `layers.<N>.` in `key`.
fn trunkLayerAt(key: []const u8) ?struct { lo: usize, hi: usize } {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, key, from, "layers.")) |i| {
        from = i + 1;
        if (i > 0 and key[i - 1] != '.') continue;
        const lo = i + "layers.".len;
        var hi = lo;
        while (hi < key.len and std.ascii.isDigit(key[hi])) hi += 1;
        if (hi > lo and hi < key.len and key[hi] == '.') return .{ .lo = lo, .hi = hi };
    }
    return null;
}
