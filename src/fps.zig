//! Farthest-point sampling — pure Zig, zero MLX deps. Deterministic greedy FPS
//! with `random_start=false` (start at index 0), the selection the UniRig
//! michelangelo point-cloud encoder uses to pick its 1024 query latents
//! (sal_perceiver.py: `fps(pos, batch, ratio=1/4, random_start=False)`). The
//! reference torch_cluster.fps is not portable; this is the standard greedy
//! algorithm the fixture dump also reproduces (`tests/dump_unirig_fixtures.py`
//! fps_numpy), so the dumped query-index oracle and this module agree.
//!
//! Also reusable by the stage-2 voxel-skin geodesic prior (dossier §4/§9).

const std = @import("std");
const testing = std.testing;

/// Select `n_sample` farthest points from `points` (n×3, xyz interleaved) by the
/// greedy max-min-distance rule, starting at index 0. Returns `n_sample` indices
/// into `points` (caller owns). Ties are broken toward the lowest index (argmax
/// keeps the first maximum), which is deterministic.
///
/// `n_sample` must be ≤ the point count. O(n_sample · n).
pub fn farthestPointSample(alloc: std.mem.Allocator, points: []const f32, n_sample: usize) ![]u32 {
    std.debug.assert(points.len % 3 == 0);
    const n = points.len / 3;
    std.debug.assert(n_sample <= n and n_sample > 0);

    const sel = try alloc.alloc(u32, n_sample);
    errdefer alloc.free(sel);
    const dist = try alloc.alloc(f32, n);
    defer alloc.free(dist);
    @memset(dist, std.math.floatMax(f32));

    var far: usize = 0; // random_start = false
    for (0..n_sample) |i| {
        sel[i] = @intCast(far);
        const fx = points[far * 3 + 0];
        const fy = points[far * 3 + 1];
        const fz = points[far * 3 + 2];
        var best: f32 = -1.0;
        var best_idx: usize = 0;
        for (0..n) |j| {
            const dx = points[j * 3 + 0] - fx;
            const dy = points[j * 3 + 1] - fy;
            const dz = points[j * 3 + 2] - fz;
            const d = dx * dx + dy * dy + dz * dz;
            if (d < dist[j]) dist[j] = d;
            if (dist[j] > best) {
                best = dist[j];
                best_idx = j;
            }
        }
        far = best_idx;
    }
    return sel;
}

// ── tests (hermetic) ──────────────────────────────────────────────────────────

test "fps: picks the two endpoints of a line first" {
    const a = testing.allocator;
    // 5 collinear points on x∈{0,1,2,3,4}; start at index 0 → farthest is index 4,
    // then the midpoint, etc.
    const pts = [_]f32{ 0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0 };
    const sel = try farthestPointSample(a, &pts, 3);
    defer a.free(sel);
    try testing.expectEqual(@as(u32, 0), sel[0]); // start
    try testing.expectEqual(@as(u32, 4), sel[1]); // farthest from 0
    try testing.expectEqual(@as(u32, 2), sel[2]); // midpoint maximizes min-dist to {0,4}
}

test "fps: deterministic and selects distinct indices" {
    const a = testing.allocator;
    var pts: [30 * 3]f32 = undefined;
    var s: u64 = 12345;
    var rng = std.Random.DefaultPrng.init(s);
    _ = &s;
    for (&pts) |*p| p.* = rng.random().float(f32) * 2 - 1;
    const s1 = try farthestPointSample(a, &pts, 10);
    defer a.free(s1);
    const s2 = try farthestPointSample(a, &pts, 10);
    defer a.free(s2);
    try testing.expectEqualSlices(u32, s1, s2); // deterministic
    // all distinct
    for (0..s1.len) |i| {
        for (i + 1..s1.len) |j| try testing.expect(s1[i] != s1[j]);
    }
}

test "fps: full sample returns a permutation of all indices" {
    const a = testing.allocator;
    const pts = [_]f32{ 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    const sel = try farthestPointSample(a, &pts, 4);
    defer a.free(sel);
    var seen = [_]bool{false} ** 4;
    for (sel) |idx| seen[idx] = true;
    for (seen) |b| try testing.expect(b);
}
