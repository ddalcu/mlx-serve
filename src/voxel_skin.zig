//! Non-neural geodesic skinning ("voxel_skin"), a pure-Zig port of UniRig's
//! `VertexGroupVoxelSkin` (`src/data/vertex_group.py:130` +
//! `voxel_skin(...)` `:434`). This is Phase 3's shippable v1 skinning path: it
//! turns an untextured mesh + a predicted joint skeleton into per-vertex,
//! top-4-normalized bone weights WITHOUT the PTv3 neural refiner — the neural
//! model only refines this same prior, so the prior alone is a usable rig
//! (dossier §4, §9). Zero MLX, std only. Output feeds `glb.zig`'s
//! `writeGlbRigged` (`Skeleton.{joints,weights}`).
//!
//! Reference algorithm (faithful port; `file:line` = UniRig shallow clone):
//!   1. Normalize verts + joints into [-1,1] using the MESH bbox only, center
//!      `(min+max)/2`, `scale = max_extent/2` (RADIUS, not diameter — the
//!      independent voxel_skin convention, `vertex_group.py:153`; dossier §7).
//!   2. Voxelize the mesh into occupied grid cells (`voxelization`, `:282`).
//!      The reference carves a solid via 6-view pyrender depth OR an open3d
//!      "occupied along >=2 of 3 axis spans" fill; both are OpenGL/library
//!      heavy, so we REPLACE them (dossier §8 HIGH item) with our own
//!      triangle-sampling surface voxelizer + the open3d >=2-of-3 span fill
//!      (`vertex_group.py:393-428`), which is deterministic, hermetic, and
//!      preserves concave gaps (a U-tube opening is spanned along only ONE
//!      axis, so it stays hollow — no cross-gap shortcut).
//!   3. Build an undirected graph over [mesh verts | occupied voxels]:
//!      mesh edges (`:494-496`), near-duplicate vert links (`:472-479`),
//!      6-connected grid links weighted x`grid_weight` (`:461-470`), and
//!      grid->nearest-vertex links (`:481-486`). Attach each JOINT POINT to its
//!      nearest combined node (`:491`).
//!   4. Per-joint Dijkstra (`shortest_path`, `:507`) -> (J,N) geodesic distance;
//!      euclidean fallback for verts unreachable from every joint (`:511-519`);
//!      inf -> finite max, clamp 1e-6 (`:521-524`).
//!   5. distance -> weight: square `(1/((1-a)d + a d^2))^2` or exp
//!      `exp(-d/max*20)` (`:526-531`), normalize over joints per vertex
//!      (`:532`).
//!   6. Top-4 per vertex, renormalize by the top-4 sum (`merge.py:277-279`,
//!      `group_per_vertex=4`).
//!
//! DEVIATION (flagged loudly): the task brief hypothesised seeding from BONE
//! SEGMENTS (parent,child). The reference `voxel_skin` does NOT — it seeds from
//! joint POINTS (`combined_tree.query(joints)`, `:491`); bone-segment nearest
//! is a DIFFERENT vertex group (`get_geodesic_distance`, `:215`). We follow the
//! reference: joint points. `parents` is accepted for API symmetry with
//! `glb.Skeleton` but is unused by the voxel_skin math.

const std = @import("std");

pub const Mode = enum { square, exp };

/// Reference defaults from `configs/transform/inference_skin_transform.yaml`
/// (`voxel_skin` kwargs) + `merge.py` export (`group_per_vertex=4`).
pub const VoxelSkinOpts = struct {
    grid: u32 = 196, // voxel resolution per axis; memory ~ O(grid^3)
    alpha: f32 = 0.5, // square-mode falloff mix (linear vs quadratic)
    link_dis: f32 = 0.00001, // near-duplicate vertex weld radius
    grid_query: u32 = 7, // reference kNN incl. self => 6-connectivity (unused knob; kept for parity)
    vertex_query: u32 = 1, // grid->vertex links per voxel
    grid_weight: f32 = 3.0, // multiplier on grid-edge cost (discourages volume shortcuts)
    mode: Mode = .square,
    top_k: usize = 4, // merge.py group_per_vertex
    interior_fill: bool = true, // open3d >=2-of-3 span solid fill
};

/// Per-vertex top-4 bone influences, ready for `glb.Skeleton`.
pub const SkinWeights = struct {
    joints: []u16, // N*4 joint indices (0-padded slots for verts with <4 influences)
    weights: []f32, // N*4 weights, each vertex's 4 summing to ~1
    allocator: std.mem.Allocator,

    pub fn deinit(self: *SkinWeights) void {
        self.allocator.free(self.joints);
        self.allocator.free(self.weights);
        self.* = undefined;
    }
};

const Edge = struct { u: u32, v: u32, w: f32 };

/// Compute geodesic voxel-skin weights. `positions` = N*3 ORIGINAL mesh coords,
/// `indices` = M*3 triangle indices, `joints` = J*3 in the SAME coord space,
/// `parents` = J parent ids (-1 root; accepted for API symmetry, unused here).
pub fn computeSkinWeights(
    allocator: std.mem.Allocator,
    positions: []const f32,
    indices: []const u32,
    joints: []const f32,
    parents: []const i32,
    opts: VoxelSkinOpts,
) !SkinWeights {
    _ = parents; // see module DEVIATION note: voxel_skin seeds from joint points, not bones.
    std.debug.assert(positions.len % 3 == 0);
    std.debug.assert(indices.len % 3 == 0);
    std.debug.assert(joints.len % 3 == 0);

    const n: usize = positions.len / 3;
    const j_count: usize = joints.len / 3;
    const grid: usize = opts.grid;
    std.debug.assert(grid >= 2);

    // --- 1. Normalize into [-1,1] by the MESH bbox (radius scale). ---
    var bmin = [3]f32{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
    var bmax = [3]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
    {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            for (0..3) |c| {
                const val = positions[i * 3 + c];
                bmin[c] = @min(bmin[c], val);
                bmax[c] = @max(bmax[c], val);
            }
        }
    }
    if (n == 0) {
        bmin = .{ 0, 0, 0 };
        bmax = .{ 0, 0, 0 };
    }
    var center: [3]f32 = undefined;
    var extent: f32 = 0;
    for (0..3) |c| {
        center[c] = (bmin[c] + bmax[c]) / 2;
        extent = @max(extent, bmax[c] - bmin[c]);
    }
    const scale: f32 = if (extent > 0) extent / 2 else 1; // radius; guard degenerate

    const nv = try allocator.alloc(f32, n * 3);
    defer allocator.free(nv);
    const nj = try allocator.alloc(f32, j_count * 3);
    defer allocator.free(nj);
    for (0..n) |i| for (0..3) |c| {
        nv[i * 3 + c] = (positions[i * 3 + c] - center[c]) / scale;
    };
    for (0..j_count) |jj| for (0..3) |c| {
        nj[jj * 3 + c] = (joints[jj * 3 + c] - center[c]) / scale;
    };

    const vsize: f32 = 2.0 / @as(f32, @floatFromInt(grid)); // voxel edge in normalized space

    // --- 2. Voxelize: surface via triangle sampling, then >=2-of-3 span fill. ---
    // `node_of_voxel[idx] = -1` (empty) or the graph node id (>= n) of an
    // occupied voxel. Also our occupancy oracle for grid links.
    const gcubed = grid * grid * grid;
    const node_of_voxel = try allocator.alloc(i32, gcubed);
    defer allocator.free(node_of_voxel);
    @memset(node_of_voxel, -1);

    var voxel_centers: std.ArrayList(f32) = .empty; // M*3, node order n.. n+M-1
    defer voxel_centers.deinit(allocator);
    {
        const surf = try allocator.alloc(bool, gcubed);
        defer allocator.free(surf);
        @memset(surf, false);
        rasterizeSurface(surf, nv, indices, grid, vsize);

        const occ = try allocator.alloc(bool, gcubed);
        defer allocator.free(occ);
        if (opts.interior_fill) {
            try fillInterior(allocator, surf, occ, grid);
        } else {
            @memcpy(occ, surf);
        }

        var node: u32 = @intCast(n);
        var vi: usize = 0;
        while (vi < grid) : (vi += 1) {
            var vj: usize = 0;
            while (vj < grid) : (vj += 1) {
                var vk: usize = 0;
                while (vk < grid) : (vk += 1) {
                    const idx = (vi * grid + vj) * grid + vk;
                    if (!occ[idx]) continue;
                    node_of_voxel[idx] = @intCast(node);
                    node += 1;
                    try voxel_centers.append(allocator, voxelCenter(vi, vsize));
                    try voxel_centers.append(allocator, voxelCenter(vj, vsize));
                    try voxel_centers.append(allocator, voxelCenter(vk, vsize));
                }
            }
        }
    }
    const m: usize = voxel_centers.items.len / 3; // occupied voxel count
    const num_nodes: usize = n + m;

    // Spatial hash over normalized mesh vertices for nearest / radius queries.
    var vhash = try VHash.init(allocator, nv, n, vsize);
    defer vhash.deinit(allocator);

    // --- 3. Build the undirected graph. ---
    var edges: std.ArrayList(Edge) = .empty;
    defer edges.deinit(allocator);

    // (a) mesh edges: each triangle contributes its 3 edges (`:494-496`).
    {
        var f: usize = 0;
        while (f < indices.len) : (f += 3) {
            const a = indices[f + 0];
            const b = indices[f + 1];
            const c = indices[f + 2];
            try edges.append(allocator, .{ .u = a, .v = b, .w = distVerts(nv, a, b) });
            try edges.append(allocator, .{ .u = b, .v = c, .w = distVerts(nv, b, c) });
            try edges.append(allocator, .{ .u = c, .v = a, .w = distVerts(nv, c, a) });
        }
    }

    // (b) near-duplicate vertex welds within link_dis (`:472-479`).
    {
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const q = [3]f32{ nv[i * 3], nv[i * 3 + 1], nv[i * 3 + 2] };
            var it = vhash.radiusIter(q, opts.link_dis * 1.001);
            while (it.next()) |cand| {
                if (cand.idx == i) continue;
                if (cand.dist > 0 and cand.dist < opts.link_dis * 1.001) {
                    try edges.append(allocator, .{ .u = @intCast(i), .v = cand.idx, .w = cand.dist });
                }
            }
        }
    }

    // (c) 6-connected grid links, cost x grid_weight (`:461-470`). The reference
    //     kNN threshold `< 2/grid*1.001` admits only face-adjacent voxels.
    {
        var vi: usize = 0;
        while (vi < grid) : (vi += 1) {
            var vj: usize = 0;
            while (vj < grid) : (vj += 1) {
                var vk: usize = 0;
                while (vk < grid) : (vk += 1) {
                    const idx = (vi * grid + vj) * grid + vk;
                    const nu = node_of_voxel[idx];
                    if (nu < 0) continue;
                    // +axis neighbors only (each undirected edge once).
                    if (vi + 1 < grid) try linkVoxel(&edges, allocator, node_of_voxel, nu, (((vi + 1) * grid + vj) * grid + vk), vsize, opts.grid_weight);
                    if (vj + 1 < grid) try linkVoxel(&edges, allocator, node_of_voxel, nu, ((vi * grid + (vj + 1)) * grid + vk), vsize, opts.grid_weight);
                    if (vk + 1 < grid) try linkVoxel(&edges, allocator, node_of_voxel, nu, ((vi * grid + vj) * grid + (vk + 1)), vsize, opts.grid_weight);
                }
            }
        }
    }

    // (d) grid -> nearest mesh vertex within one voxel (vertex_query=1, `:481-486`).
    {
        var voxel_node: u32 = @intCast(n);
        var mi: usize = 0;
        while (mi < m) : (mi += 1) {
            const q = [3]f32{ voxel_centers.items[mi * 3], voxel_centers.items[mi * 3 + 1], voxel_centers.items[mi * 3 + 2] };
            if (vhash.nearest(q)) |near| {
                if (near.dist > 0 and near.dist < vsize * 1.001) {
                    try edges.append(allocator, .{ .u = voxel_node, .v = near.idx, .w = near.dist });
                }
            }
            voxel_node += 1;
        }
    }

    // --- Attach each joint POINT to its nearest combined node (`:491`). ---
    const joint_seed = try allocator.alloc(u32, j_count);
    defer allocator.free(joint_seed);
    for (0..j_count) |jj| {
        const q = [3]f32{ nj[jj * 3], nj[jj * 3 + 1], nj[jj * 3 + 2] };
        var best_node: u32 = 0;
        var best_d: f32 = std.math.inf(f32);
        if (vhash.nearest(q)) |near| {
            best_node = near.idx;
            best_d = near.dist;
        }
        if (nearestVoxelNode(node_of_voxel, grid, vsize, q)) |vn| {
            if (vn.dist < best_d) {
                best_node = vn.node;
                best_d = vn.dist;
            }
        }
        joint_seed[jj] = best_node;
    }

    // --- Build CSR adjacency (undirected: each edge both ways). ---
    const adj_start = try allocator.alloc(usize, num_nodes + 1);
    defer allocator.free(adj_start);
    @memset(adj_start, 0);
    for (edges.items) |e| {
        adj_start[e.u] += 1;
        adj_start[e.v] += 1;
    }
    var acc: usize = 0;
    for (0..num_nodes) |i| {
        const d = adj_start[i];
        adj_start[i] = acc;
        acc += d;
    }
    adj_start[num_nodes] = acc;
    const adj_to = try allocator.alloc(u32, acc);
    defer allocator.free(adj_to);
    const adj_w = try allocator.alloc(f32, acc);
    defer allocator.free(adj_w);
    {
        const cursor = try allocator.alloc(usize, num_nodes);
        defer allocator.free(cursor);
        for (0..num_nodes) |i| cursor[i] = adj_start[i];
        for (edges.items) |e| {
            adj_to[cursor[e.u]] = e.v;
            adj_w[cursor[e.u]] = e.w;
            cursor[e.u] += 1;
            adj_to[cursor[e.v]] = e.u;
            adj_w[cursor[e.v]] = e.w;
            cursor[e.v] += 1;
        }
    }

    // --- 4. Per-joint Dijkstra -> (J, N) geodesic distance. ---
    const geo = try allocator.alloc(f32, j_count * n);
    defer allocator.free(geo);
    {
        var dijk = try Dijkstra.init(allocator, num_nodes);
        defer dijk.deinit(allocator);
        for (0..j_count) |jj| {
            try dijk.run(adj_start, adj_to, adj_w, joint_seed[jj]);
            for (0..n) |v| geo[jj * n + v] = dijk.dist[v];
        }
    }

    // Euclidean fallback for verts unreachable from EVERY joint (`:511-519`).
    const kfb: usize = @min(j_count, 3);
    for (0..n) |v| {
        var all_inf = true;
        for (0..j_count) |jj| {
            if (!std.math.isInf(geo[jj * n + v])) {
                all_inf = false;
                break;
            }
        }
        if (!all_inf) continue;
        // k nearest joints (euclidean) get a real distance; others stay inf.
        var sel_j: [3]usize = .{ 0, 0, 0 };
        var sel_d: [3]f32 = .{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
        for (0..j_count) |jj| {
            const dx = nv[v * 3] - nj[jj * 3];
            const dy = nv[v * 3 + 1] - nj[jj * 3 + 1];
            const dz = nv[v * 3 + 2] - nj[jj * 3 + 2];
            const d = @sqrt(dx * dx + dy * dy + dz * dz);
            var worst: usize = 0;
            for (1..kfb) |s| if (sel_d[s] > sel_d[worst]) {
                worst = s;
            };
            if (d < sel_d[worst]) {
                sel_d[worst] = d;
                sel_j[worst] = jj;
            }
        }
        for (0..kfb) |s| geo[sel_j[s] * n + v] = sel_d[s];
    }

    // inf/NaN -> finite max, then clamp to 1e-6 (`:521-524`).
    var max_dis: f32 = 0;
    var any_finite = false;
    for (geo) |d| {
        if (std.math.isFinite(d)) {
            max_dis = @max(max_dis, d);
            any_finite = true;
        }
    }
    if (!any_finite) max_dis = 1;
    for (geo) |*d| {
        if (!std.math.isFinite(d.*)) d.* = max_dis;
        d.* = @max(d.*, 1e-6);
    }

    // --- 5+6. distance -> weight, per-vertex joint-normalize, top-4. ---
    const out_joints = try allocator.alloc(u16, n * 4);
    errdefer allocator.free(out_joints);
    const out_weights = try allocator.alloc(f32, n * 4);
    errdefer allocator.free(out_weights);
    @memset(out_joints, 0);
    @memset(out_weights, 0);

    const wj = try allocator.alloc(f32, @max(j_count, 1));
    defer allocator.free(wj);
    for (0..n) |v| {
        var sum_w: f32 = 0;
        for (0..j_count) |jj| {
            const d = geo[jj * n + v];
            const w = switch (opts.mode) {
                .square => blk: {
                    const denom = (1 - opts.alpha) * d + opts.alpha * d * d;
                    const inv = 1.0 / denom;
                    break :blk inv * inv;
                },
                .exp => @exp(-d / max_dis * 20.0),
            };
            wj[jj] = w;
            sum_w += w;
        }
        if (sum_w > 0) for (0..j_count) |jj| {
            wj[jj] /= sum_w;
        };

        // Top-4 (merge.py:277-279): pick the 4 largest, renormalize by their sum.
        var sel_w: [4]f32 = .{ -1, -1, -1, -1 };
        var sel_j: [4]u16 = .{ 0, 0, 0, 0 };
        for (0..j_count) |jj| {
            var worst: usize = 0;
            for (1..4) |s| if (sel_w[s] < sel_w[worst]) {
                worst = s;
            };
            if (wj[jj] > sel_w[worst]) {
                sel_w[worst] = wj[jj];
                sel_j[worst] = @intCast(jj);
            }
        }
        var top_sum: f32 = 0;
        for (0..4) |s| top_sum += @max(sel_w[s], 0);
        for (0..4) |s| {
            const w = @max(sel_w[s], 0);
            out_joints[v * 4 + s] = sel_j[s];
            out_weights[v * 4 + s] = if (top_sum > 0) w / top_sum else 0;
        }
    }

    return .{ .joints = out_joints, .weights = out_weights, .allocator = allocator };
}

fn voxelCenter(idx: usize, vsize: f32) f32 {
    // Center convention: (idx+0.5)*vsize - 1 (`vertex_group.py:357/431`).
    return (@as(f32, @floatFromInt(idx)) + 0.5) * vsize - 1.0;
}

fn distVerts(nv: []const f32, a: u32, b: u32) f32 {
    const dx = nv[a * 3] - nv[b * 3];
    const dy = nv[a * 3 + 1] - nv[b * 3 + 1];
    const dz = nv[a * 3 + 2] - nv[b * 3 + 2];
    return @sqrt(dx * dx + dy * dy + dz * dz);
}

fn linkVoxel(edges: *std.ArrayList(Edge), alloc: std.mem.Allocator, node_of_voxel: []const i32, nu: i32, nbr_idx: usize, vsize: f32, grid_weight: f32) !void {
    const nv2 = node_of_voxel[nbr_idx];
    if (nv2 < 0) return;
    // Face-adjacent centers are exactly `vsize` apart.
    try edges.append(alloc, .{ .u = @intCast(nu), .v = @intCast(nv2), .w = vsize * grid_weight });
}

/// Mark every voxel a triangle passes through by dense barycentric point
/// sampling at <= vsize/2 spacing (a robust, conservative surface voxelizer;
/// our replacement for pyrender/open3d `voxelization`, dossier §8).
fn rasterizeSurface(surf: []bool, nv: []const f32, indices: []const u32, grid: usize, vsize: f32) void {
    var f: usize = 0;
    while (f < indices.len) : (f += 3) {
        const a = indices[f + 0];
        const b = indices[f + 1];
        const c = indices[f + 2];
        const va = [3]f32{ nv[a * 3], nv[a * 3 + 1], nv[a * 3 + 2] };
        const vb = [3]f32{ nv[b * 3], nv[b * 3 + 1], nv[b * 3 + 2] };
        const vc = [3]f32{ nv[c * 3], nv[c * 3 + 1], nv[c * 3 + 2] };
        var e1: [3]f32 = undefined;
        var e2: [3]f32 = undefined;
        var maxlen: f32 = 0;
        for (0..3) |d| {
            e1[d] = vb[d] - va[d];
            e2[d] = vc[d] - va[d];
        }
        const l1 = @sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
        const l2 = @sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
        const dcx = vb[0] - vc[0];
        const dcy = vb[1] - vc[1];
        const dcz = vb[2] - vc[2];
        const l3 = @sqrt(dcx * dcx + dcy * dcy + dcz * dcz);
        maxlen = @max(l1, @max(l2, l3));
        var steps: usize = @intFromFloat(@ceil(maxlen / (vsize * 0.5)));
        if (steps < 1) steps = 1;
        if (steps > 4 * grid) steps = 4 * grid; // guard pathological giant triangles
        const fsteps: f32 = @floatFromInt(steps);
        var i: usize = 0;
        while (i <= steps) : (i += 1) {
            const u = @as(f32, @floatFromInt(i)) / fsteps;
            var jj: usize = 0;
            while (i + jj <= steps) : (jj += 1) {
                const vv = @as(f32, @floatFromInt(jj)) / fsteps;
                const p = [3]f32{
                    va[0] + u * e1[0] + vv * e2[0],
                    va[1] + u * e1[1] + vv * e2[1],
                    va[2] + u * e1[2] + vv * e2[2],
                };
                const idx = voxelIndexOf(p, grid, vsize) orelse continue;
                surf[idx] = true;
            }
        }
    }
}

fn voxelIndexOf(p: [3]f32, grid: usize, vsize: f32) ?usize {
    var ijk: [3]usize = undefined;
    for (0..3) |d| {
        const fi = @floor((p[d] + 1.0) / vsize);
        if (fi < 0) {
            ijk[d] = 0;
        } else if (fi >= @as(f32, @floatFromInt(grid))) {
            ijk[d] = grid - 1;
        } else {
            ijk[d] = @intFromFloat(fi);
        }
    }
    return (ijk[0] * grid + ijk[1]) * grid + ijk[2];
}

/// open3d-style solid fill (`vertex_group.py:393-428`): a voxel is occupied if
/// it is `surf` OR sits between the min/max occupied index along >= 2 of the 3
/// axes. Preserves concave gaps (spanned along only 1 axis -> stays hollow).
fn fillInterior(alloc: std.mem.Allocator, surf: []const bool, occ: []bool, grid: usize) !void {
    const g2 = grid * grid;
    // Per-column min/max occupied index along each axis; -1 = empty column.
    const xmin = try alloc.alloc(i32, g2); // key (j,k)
    defer alloc.free(xmin);
    const xmax = try alloc.alloc(i32, g2);
    defer alloc.free(xmax);
    const ymin = try alloc.alloc(i32, g2); // key (i,k)
    defer alloc.free(ymin);
    const ymax = try alloc.alloc(i32, g2);
    defer alloc.free(ymax);
    const zmin = try alloc.alloc(i32, g2); // key (i,j)
    defer alloc.free(zmin);
    const zmax = try alloc.alloc(i32, g2);
    defer alloc.free(zmax);
    @memset(xmin, -1);
    @memset(xmax, -1);
    @memset(ymin, -1);
    @memset(ymax, -1);
    @memset(zmin, -1);
    @memset(zmax, -1);

    var i: usize = 0;
    while (i < grid) : (i += 1) {
        var j: usize = 0;
        while (j < grid) : (j += 1) {
            var k: usize = 0;
            while (k < grid) : (k += 1) {
                if (!surf[(i * grid + j) * grid + k]) continue;
                const ii: i32 = @intCast(i);
                const jj: i32 = @intCast(j);
                const kk: i32 = @intCast(k);
                updSpan(xmin, xmax, j * grid + k, ii);
                updSpan(ymin, ymax, i * grid + k, jj);
                updSpan(zmin, zmax, i * grid + j, kk);
            }
        }
    }

    i = 0;
    while (i < grid) : (i += 1) {
        var j: usize = 0;
        while (j < grid) : (j += 1) {
            var k: usize = 0;
            while (k < grid) : (k += 1) {
                const idx = (i * grid + j) * grid + k;
                if (surf[idx]) {
                    occ[idx] = true;
                    continue;
                }
                const ii: i32 = @intCast(i);
                const jj: i32 = @intCast(j);
                const kk: i32 = @intCast(k);
                var count: u8 = 0;
                if (inSpan(xmin, xmax, j * grid + k, ii)) count += 1;
                if (inSpan(ymin, ymax, i * grid + k, jj)) count += 1;
                if (inSpan(zmin, zmax, i * grid + j, kk)) count += 1;
                occ[idx] = count >= 2;
            }
        }
    }
}

fn updSpan(mins: []i32, maxs: []i32, key: usize, v: i32) void {
    if (mins[key] < 0 or v < mins[key]) mins[key] = v;
    if (maxs[key] < 0 or v > maxs[key]) maxs[key] = v;
}

fn inSpan(mins: []const i32, maxs: []const i32, key: usize, v: i32) bool {
    if (mins[key] < 0) return false;
    return v >= mins[key] and v <= maxs[key];
}

/// Nearest occupied voxel to a normalized point, searched as expanding
/// Chebyshev shells over the integer voxel grid.
fn nearestVoxelNode(node_of_voxel: []const i32, grid: usize, vsize: f32, q: [3]f32) ?struct { node: u32, dist: f32 } {
    var home: [3]i64 = undefined;
    for (0..3) |d| {
        const fi = @floor((q[d] + 1.0) / vsize);
        var idx: i64 = @intFromFloat(fi);
        if (idx < 0) idx = 0;
        if (idx >= @as(i64, @intCast(grid))) idx = @as(i64, @intCast(grid)) - 1;
        home[d] = idx;
    }
    const gi: i64 = @intCast(grid);
    var best_node: u32 = 0;
    var best_d: f32 = std.math.inf(f32);
    var found = false;
    var r: i64 = 0;
    while (r < gi) : (r += 1) {
        var di: i64 = -r;
        while (di <= r) : (di += 1) {
            var dj: i64 = -r;
            while (dj <= r) : (dj += 1) {
                var dk: i64 = -r;
                while (dk <= r) : (dk += 1) {
                    const cheb = @max(@abs(di), @max(@abs(dj), @abs(dk)));
                    if (cheb != r) continue; // shell only
                    const ci = home[0] + di;
                    const cj = home[1] + dj;
                    const ck = home[2] + dk;
                    if (ci < 0 or cj < 0 or ck < 0 or ci >= gi or cj >= gi or ck >= gi) continue;
                    const idx: usize = @intCast((ci * gi + cj) * gi + ck);
                    const nn = node_of_voxel[idx];
                    if (nn < 0) continue;
                    const cx = voxelCenter(@intCast(ci), vsize);
                    const cy = voxelCenter(@intCast(cj), vsize);
                    const cz = voxelCenter(@intCast(ck), vsize);
                    const dx = cx - q[0];
                    const dy = cy - q[1];
                    const dz = cz - q[2];
                    const dd = @sqrt(dx * dx + dy * dy + dz * dz);
                    if (dd < best_d) {
                        best_d = dd;
                        best_node = @intCast(nn);
                        found = true;
                    }
                }
            }
        }
        // Points in shell r+1 are >= r*vsize away; stop once that can't beat best.
        if (found and @as(f32, @floatFromInt(r)) * vsize > best_d) break;
    }
    if (!found) return null;
    return .{ .node = best_node, .dist = best_d };
}

// ---------------------------------------------------------------------------
// Uniform spatial hash over mesh vertices (nearest / radius queries).
// ---------------------------------------------------------------------------

const VHash = struct {
    pts: []const f32,
    npts: usize,
    cell: f32,
    origin: [3]f32,
    dims: [3]usize,
    starts: []usize, // len = prod(dims)+1
    items: []u32, // vertex ids grouped by cell

    fn init(alloc: std.mem.Allocator, pts: []const f32, npts: usize, cell: f32) !VHash {
        var mn = [3]f32{ std.math.inf(f32), std.math.inf(f32), std.math.inf(f32) };
        var mx = [3]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
        for (0..npts) |i| for (0..3) |c| {
            mn[c] = @min(mn[c], pts[i * 3 + c]);
            mx[c] = @max(mx[c], pts[i * 3 + c]);
        };
        if (npts == 0) {
            mn = .{ 0, 0, 0 };
            mx = .{ 0, 0, 0 };
        }
        const cs = if (cell > 0) cell else 1;
        var dims: [3]usize = undefined;
        for (0..3) |c| {
            const span = mx[c] - mn[c];
            const dc: usize = @intFromFloat(@floor(span / cs));
            dims[c] = dc + 1;
        }
        const ncells = dims[0] * dims[1] * dims[2];
        const starts = try alloc.alloc(usize, ncells + 1);
        @memset(starts, 0);
        var self = VHash{
            .pts = pts,
            .npts = npts,
            .cell = cs,
            .origin = mn,
            .dims = dims,
            .starts = starts,
            .items = &[_]u32{},
        };
        for (0..npts) |i| {
            const cidx = self.cellLinear(.{ pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2] });
            self.starts[cidx] += 1;
        }
        var acc: usize = 0;
        for (0..ncells) |c| {
            const d = self.starts[c];
            self.starts[c] = acc;
            acc += d;
        }
        self.starts[ncells] = acc;
        const items = try alloc.alloc(u32, npts);
        const cursor = try alloc.alloc(usize, ncells);
        defer alloc.free(cursor);
        for (0..ncells) |c| cursor[c] = self.starts[c];
        for (0..npts) |i| {
            const cidx = self.cellLinear(.{ pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2] });
            items[cursor[cidx]] = @intCast(i);
            cursor[cidx] += 1;
        }
        self.items = items;
        return self;
    }

    fn deinit(self: *VHash, alloc: std.mem.Allocator) void {
        alloc.free(self.starts);
        if (self.items.len > 0) alloc.free(self.items);
        self.* = undefined;
    }

    fn cellCoord(self: *const VHash, p: [3]f32) [3]usize {
        var c: [3]usize = undefined;
        for (0..3) |d| {
            const fi = @floor((p[d] - self.origin[d]) / self.cell);
            if (fi < 0) {
                c[d] = 0;
            } else if (fi >= @as(f32, @floatFromInt(self.dims[d]))) {
                c[d] = self.dims[d] - 1;
            } else {
                c[d] = @intFromFloat(fi);
            }
        }
        return c;
    }

    fn cellLinear(self: *const VHash, p: [3]f32) usize {
        const c = self.cellCoord(p);
        return (c[0] * self.dims[1] + c[1]) * self.dims[2] + c[2];
    }

    fn nearest(self: *const VHash, q: [3]f32) ?struct { idx: u32, dist: f32 } {
        if (self.npts == 0) return null;
        const home = self.cellCoord(q);
        var best_idx: u32 = 0;
        var best_d: f32 = std.math.inf(f32);
        var found = false;
        const maxdim = @max(self.dims[0], @max(self.dims[1], self.dims[2]));
        var r: usize = 0;
        while (r <= maxdim) : (r += 1) {
            self.scanShell(home, r, q, &best_idx, &best_d, &found);
            if (found and @as(f32, @floatFromInt(r)) * self.cell > best_d) break;
        }
        if (!found) return null;
        return .{ .idx = best_idx, .dist = best_d };
    }

    fn scanShell(self: *const VHash, home: [3]usize, r: usize, q: [3]f32, best_idx: *u32, best_d: *f32, found: *bool) void {
        const ri: i64 = @intCast(r);
        var di: i64 = -ri;
        while (di <= ri) : (di += 1) {
            var dj: i64 = -ri;
            while (dj <= ri) : (dj += 1) {
                var dk: i64 = -ri;
                while (dk <= ri) : (dk += 1) {
                    if (@max(@abs(di), @max(@abs(dj), @abs(dk))) != ri) continue;
                    const ci = @as(i64, @intCast(home[0])) + di;
                    const cj = @as(i64, @intCast(home[1])) + dj;
                    const ck = @as(i64, @intCast(home[2])) + dk;
                    if (ci < 0 or cj < 0 or ck < 0) continue;
                    if (ci >= @as(i64, @intCast(self.dims[0])) or cj >= @as(i64, @intCast(self.dims[1])) or ck >= @as(i64, @intCast(self.dims[2]))) continue;
                    const cidx: usize = (@as(usize, @intCast(ci)) * self.dims[1] + @as(usize, @intCast(cj))) * self.dims[2] + @as(usize, @intCast(ck));
                    var s = self.starts[cidx];
                    const e = self.starts[cidx + 1];
                    while (s < e) : (s += 1) {
                        const vi = self.items[s];
                        const dx = self.pts[vi * 3] - q[0];
                        const dy = self.pts[vi * 3 + 1] - q[1];
                        const dz = self.pts[vi * 3 + 2] - q[2];
                        const dd = @sqrt(dx * dx + dy * dy + dz * dz);
                        if (dd < best_d.*) {
                            best_d.* = dd;
                            best_idx.* = vi;
                            found.* = true;
                        }
                    }
                }
            }
        }
    }

    const RadiusIter = struct {
        h: *const VHash,
        q: [3]f32,
        radius: f32,
        lo: [3]i64,
        hi: [3]i64,
        cur: [3]i64,
        // iterate current cell's items
        s: usize,
        e: usize,

        fn next(self: *RadiusIter) ?struct { idx: u32, dist: f32 } {
            while (true) {
                while (self.s < self.e) {
                    const vi = self.h.items[self.s];
                    self.s += 1;
                    const dx = self.h.pts[vi * 3] - self.q[0];
                    const dy = self.h.pts[vi * 3 + 1] - self.q[1];
                    const dz = self.h.pts[vi * 3 + 2] - self.q[2];
                    const dd = @sqrt(dx * dx + dy * dy + dz * dz);
                    if (dd <= self.radius) return .{ .idx = vi, .dist = dd };
                }
                if (!self.advanceCell()) return null;
            }
        }

        fn advanceCell(self: *RadiusIter) bool {
            while (true) {
                self.cur[2] += 1;
                if (self.cur[2] > self.hi[2]) {
                    self.cur[2] = self.lo[2];
                    self.cur[1] += 1;
                    if (self.cur[1] > self.hi[1]) {
                        self.cur[1] = self.lo[1];
                        self.cur[0] += 1;
                        if (self.cur[0] > self.hi[0]) return false;
                    }
                }
                const ci = self.cur[0];
                const cj = self.cur[1];
                const ck = self.cur[2];
                if (ci < 0 or cj < 0 or ck < 0) continue;
                if (ci >= @as(i64, @intCast(self.h.dims[0])) or cj >= @as(i64, @intCast(self.h.dims[1])) or ck >= @as(i64, @intCast(self.h.dims[2]))) continue;
                const cidx: usize = (@as(usize, @intCast(ci)) * self.h.dims[1] + @as(usize, @intCast(cj))) * self.h.dims[2] + @as(usize, @intCast(ck));
                self.s = self.h.starts[cidx];
                self.e = self.h.starts[cidx + 1];
                return true;
            }
        }
    };

    fn radiusIter(self: *const VHash, q: [3]f32, radius: f32) RadiusIter {
        const home = self.cellCoord(q);
        // radius <= one cell for our use (link_dis << cell), so a 3x3x3 window covers it.
        const reach: i64 = @max(1, @as(i64, @intFromFloat(@ceil(radius / self.cell))));
        var it = RadiusIter{
            .h = self,
            .q = q,
            .radius = radius,
            .lo = .{ @as(i64, @intCast(home[0])) - reach, @as(i64, @intCast(home[1])) - reach, @as(i64, @intCast(home[2])) - reach },
            .hi = .{ @as(i64, @intCast(home[0])) + reach, @as(i64, @intCast(home[1])) + reach, @as(i64, @intCast(home[2])) + reach },
            .cur = undefined,
            .s = 0,
            .e = 0,
        };
        it.cur = .{ it.lo[0], it.lo[1], it.lo[2] - 1 }; // advanceCell steps to the first cell
        _ = it.advanceCell();
        return it;
    }
};

// ---------------------------------------------------------------------------
// Dijkstra with a binary min-heap (reused across joints).
// ---------------------------------------------------------------------------

const Dijkstra = struct {
    dist: []f32,
    heap_d: []f32,
    heap_n: []u32,
    heap_len: usize,

    fn init(alloc: std.mem.Allocator, num_nodes: usize) !Dijkstra {
        return .{
            .dist = try alloc.alloc(f32, num_nodes),
            .heap_d = try alloc.alloc(f32, num_nodes + 1),
            .heap_n = try alloc.alloc(u32, num_nodes + 1),
            .heap_len = 0,
        };
    }
    fn deinit(self: *Dijkstra, alloc: std.mem.Allocator) void {
        alloc.free(self.dist);
        alloc.free(self.heap_d);
        alloc.free(self.heap_n);
        self.* = undefined;
    }

    fn push(self: *Dijkstra, d: f32, node: u32) void {
        var i = self.heap_len;
        self.heap_len += 1;
        self.heap_d[i] = d;
        self.heap_n[i] = node;
        while (i > 0) {
            const parent = (i - 1) / 2;
            if (self.heap_d[parent] <= self.heap_d[i]) break;
            self.swap(parent, i);
            i = parent;
        }
    }
    const HeapTop = struct { d: f32, n: u32 };
    fn pop(self: *Dijkstra) HeapTop {
        const top = HeapTop{ .d = self.heap_d[0], .n = self.heap_n[0] };
        self.heap_len -= 1;
        self.heap_d[0] = self.heap_d[self.heap_len];
        self.heap_n[0] = self.heap_n[self.heap_len];
        var i: usize = 0;
        while (true) {
            const l = 2 * i + 1;
            const r = 2 * i + 2;
            var smallest = i;
            if (l < self.heap_len and self.heap_d[l] < self.heap_d[smallest]) smallest = l;
            if (r < self.heap_len and self.heap_d[r] < self.heap_d[smallest]) smallest = r;
            if (smallest == i) break;
            self.swap(i, smallest);
            i = smallest;
        }
        return top;
    }
    fn swap(self: *Dijkstra, a: usize, b: usize) void {
        std.mem.swap(f32, &self.heap_d[a], &self.heap_d[b]);
        std.mem.swap(u32, &self.heap_n[a], &self.heap_n[b]);
    }

    fn run(self: *Dijkstra, adj_start: []const usize, adj_to: []const u32, adj_w: []const f32, seed: u32) !void {
        @memset(self.dist, std.math.inf(f32));
        self.heap_len = 0;
        self.dist[seed] = 0;
        self.push(0, seed);
        while (self.heap_len > 0) {
            const cur = self.pop();
            if (cur.d > self.dist[cur.n]) continue; // stale
            const s = adj_start[cur.n];
            const e = adj_start[cur.n + 1];
            var k = s;
            while (k < e) : (k += 1) {
                const to = adj_to[k];
                const nd = cur.d + adj_w[k];
                if (nd < self.dist[to]) {
                    self.dist[to] = nd;
                    self.push(nd, to);
                }
            }
        }
    }
};

// ===========================================================================
// Tests (hermetic — no weights, no MLX). Build/iterate:
//   zig test src/voxel_skin.zig
// ===========================================================================

const testing = std.testing;

/// Build a tessellated open cylinder along +Y: `rings` rings of `k` points at
/// radius `rad`, connected by side quads (two tris each). Returns positions
/// (interleaved) + indices, caller owns.
fn buildCylinder(alloc: std.mem.Allocator, rings: usize, k: usize, rad: f32, height: f32) !struct { pos: []f32, idx: []u32 } {
    var pos: std.ArrayList(f32) = .empty;
    errdefer pos.deinit(alloc);
    var idx: std.ArrayList(u32) = .empty;
    errdefer idx.deinit(alloc);
    for (0..rings) |ri| {
        const y = height * @as(f32, @floatFromInt(ri)) / @as(f32, @floatFromInt(rings - 1));
        for (0..k) |ki| {
            const ang = 2.0 * std.math.pi * @as(f32, @floatFromInt(ki)) / @as(f32, @floatFromInt(k));
            try pos.append(alloc, rad * @cos(ang));
            try pos.append(alloc, y);
            try pos.append(alloc, rad * @sin(ang));
        }
    }
    for (0..rings - 1) |ri| {
        for (0..k) |ki| {
            const a: u32 = @intCast(ri * k + ki);
            const b: u32 = @intCast(ri * k + (ki + 1) % k);
            const c: u32 = @intCast((ri + 1) * k + ki);
            const d: u32 = @intCast((ri + 1) * k + (ki + 1) % k);
            try idx.appendSlice(alloc, &[_]u32{ a, b, c, b, d, c });
        }
    }
    return .{ .pos = try pos.toOwnedSlice(alloc), .idx = try idx.toOwnedSlice(alloc) };
}

/// Build a thin ribbon (two chains offset in Z) that follows a U-shaped
/// centerline in the XY plane: left arm down, bottom across, right arm up.
/// The two arm tips end euclidean-close but geodesic-far. Returns pos+idx and
/// the vertex ids of the left-tip and right-tip lower verts.
fn buildURibbon(alloc: std.mem.Allocator, seg: usize, gap: f32, arm_h: f32, width: f32) !struct { pos: []f32, idx: []u32, left_tip: u32, right_tip: u32 } {
    var center: std.ArrayList([2]f32) = .empty;
    defer center.deinit(alloc);
    const hx = gap / 2; // half-gap: tips at x = -hx (left) and +hx (right)
    // Left arm: from top (-hx, arm_h) down to base (-hx, 0).
    for (0..seg + 1) |i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(seg));
        try center.append(alloc, .{ -hx, arm_h * (1 - t) });
    }
    // Bottom: from (-hx,0) across to (+hx,0), skip the shared first point.
    for (1..seg + 1) |i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(seg));
        try center.append(alloc, .{ -hx + (2 * hx) * t, 0 });
    }
    // Right arm: from base (+hx,0) up to top (+hx, arm_h), skip shared point.
    for (1..seg + 1) |i| {
        const t = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(seg));
        try center.append(alloc, .{ hx, arm_h * t });
    }
    const cc = center.items.len;

    var pos: std.ArrayList(f32) = .empty;
    errdefer pos.deinit(alloc);
    var idx: std.ArrayList(u32) = .empty;
    errdefer idx.deinit(alloc);
    for (center.items) |p| {
        try pos.append(alloc, p[0]);
        try pos.append(alloc, p[1]);
        try pos.append(alloc, -width);
        try pos.append(alloc, p[0]);
        try pos.append(alloc, p[1]);
        try pos.append(alloc, width);
    }
    // Quads between consecutive cross-sections (each has 2 verts: lo=2i, hi=2i+1).
    for (0..cc - 1) |i| {
        const lo0: u32 = @intCast(2 * i);
        const hi0: u32 = @intCast(2 * i + 1);
        const lo1: u32 = @intCast(2 * (i + 1));
        const hi1: u32 = @intCast(2 * (i + 1) + 1);
        try idx.appendSlice(alloc, &[_]u32{ lo0, hi0, lo1, hi0, hi1, lo1 });
    }
    return .{
        .pos = try pos.toOwnedSlice(alloc),
        .idx = try idx.toOwnedSlice(alloc),
        .left_tip = 0, // lo vert of the first cross-section (left top)
        .right_tip = @intCast(2 * (cc - 1)), // lo vert of the last cross-section (right top)
    };
}

fn weightOf(sw: *const SkinWeights, vert: usize, joint: u16) f32 {
    var total: f32 = 0;
    for (0..4) |s| {
        if (sw.joints[vert * 4 + s] == joint) total += sw.weights[vert * 4 + s];
    }
    return total;
}

test "voxel_skin: two-joint cylinder splits bottom->bone0, top->bone1, mid mixed" {
    const a = testing.allocator;
    const rings: usize = 7;
    const k: usize = 12;
    const cyl = try buildCylinder(a, rings, k, 0.5, 3.0);
    defer a.free(cyl.pos);
    defer a.free(cyl.idx);

    const joints = [_]f32{ 0, 0.4, 0, 0, 2.6, 0 };
    const parents = [_]i32{ -1, 0 };
    var sw = try computeSkinWeights(a, cyl.pos, cyl.idx, &joints, &parents, .{ .grid = 40 });
    defer sw.deinit();

    // Bottom ring (ids 0..k-1) leans to bone 0; top ring (last ring) to bone 1.
    for (0..k) |ki| {
        try testing.expect(weightOf(&sw, ki, 0) > weightOf(&sw, ki, 1));
        const top = (rings - 1) * k + ki;
        try testing.expect(weightOf(&sw, top, 1) > weightOf(&sw, top, 0));
    }
    // Mid ring: both bones present with non-trivial weight.
    const mid_ring = rings / 2;
    for (0..k) |ki| {
        const v = mid_ring * k + ki;
        try testing.expect(weightOf(&sw, v, 0) > 0.05);
        try testing.expect(weightOf(&sw, v, 1) > 0.05);
    }
    // Every row sums to ~1.
    for (0..cyl.pos.len / 3) |v| {
        var s: f32 = 0;
        for (0..4) |c| s += sw.weights[v * 4 + c];
        try testing.expectApproxEqAbs(@as(f32, 1.0), s, 1e-5);
    }
}

test "voxel_skin: geodesic beats euclidean on a U-tube (euclidean-near bone ~0)" {
    const a = testing.allocator;
    // Tips separated by gap=0.2, arms 1.0 tall; ribbon thin in Z.
    const u = try buildURibbon(a, 14, 0.2, 1.0, 0.03);
    defer a.free(u.pos);
    defer a.free(u.idx);

    // Bone 0 down at the LEFT base; bone 1 at the RIGHT tip. The left-tip vertex
    // is euclidean-CLOSER to bone 1 (across the small gap) but geodesic-closer
    // to bone 0 (down the same arm). A euclidean skinner mis-assigns it.
    const j0 = [3]f32{ -0.1, 0.0, 0 }; // left base
    const j1 = [3]f32{ 0.1, 1.0, 0 }; // right tip
    const joints = [_]f32{ j0[0], j0[1], j0[2], j1[0], j1[1], j1[2] };
    const parents = [_]i32{ -1, 0 };

    // Sanity: the probe (left tip) is euclidean-nearer to bone 1 than bone 0.
    const lt = [3]f32{ u.pos[u.left_tip * 3], u.pos[u.left_tip * 3 + 1], u.pos[u.left_tip * 3 + 2] };
    const e0 = @sqrt((lt[0] - j0[0]) * (lt[0] - j0[0]) + (lt[1] - j0[1]) * (lt[1] - j0[1]) + (lt[2] - j0[2]) * (lt[2] - j0[2]));
    const e1 = @sqrt((lt[0] - j1[0]) * (lt[0] - j1[0]) + (lt[1] - j1[1]) * (lt[1] - j1[1]) + (lt[2] - j1[2]) * (lt[2] - j1[2]));
    try testing.expect(e1 < e0); // euclidean would prefer the WRONG bone (1)

    var sw = try computeSkinWeights(a, u.pos, u.idx, &joints, &parents, .{ .grid = 44 });
    defer sw.deinit();

    // Geodesic gives the left tip almost entirely to bone 0; the euclidean-near
    // but geodesic-far bone 1 gets ~0.
    try testing.expect(weightOf(&sw, u.left_tip, 0) > 0.8);
    try testing.expect(weightOf(&sw, u.left_tip, 1) < 0.15);
}

test "voxel_skin: weights valid — [0,1], rows sum to 1, joint ids < J" {
    const a = testing.allocator;
    const cyl = try buildCylinder(a, 6, 10, 0.5, 2.5);
    defer a.free(cyl.pos);
    defer a.free(cyl.idx);
    const joints = [_]f32{ 0, 0.2, 0, 0, 1.25, 0, 0, 2.3, 0 };
    const parents = [_]i32{ -1, 0, 1 };
    const jcount: u16 = 3;
    var sw = try computeSkinWeights(a, cyl.pos, cyl.idx, &joints, &parents, .{ .grid = 36 });
    defer sw.deinit();

    for (0..cyl.pos.len / 3) |v| {
        var s: f32 = 0;
        for (0..4) |c| {
            const w = sw.weights[v * 4 + c];
            try testing.expect(w >= 0 and w <= 1.0 + 1e-6);
            try testing.expect(sw.joints[v * 4 + c] < jcount);
            s += w;
        }
        try testing.expectApproxEqAbs(@as(f32, 1.0), s, 1e-5);
    }
}

test "voxel_skin: deterministic across runs" {
    const a = testing.allocator;
    const cyl = try buildCylinder(a, 6, 10, 0.5, 2.5);
    defer a.free(cyl.pos);
    defer a.free(cyl.idx);
    const joints = [_]f32{ 0, 0.3, 0, 0, 2.2, 0 };
    const parents = [_]i32{ -1, 0 };

    var s1 = try computeSkinWeights(a, cyl.pos, cyl.idx, &joints, &parents, .{ .grid = 32 });
    defer s1.deinit();
    var s2 = try computeSkinWeights(a, cyl.pos, cyl.idx, &joints, &parents, .{ .grid = 32 });
    defer s2.deinit();
    try testing.expectEqualSlices(u16, s1.joints, s2.joints);
    try testing.expectEqualSlices(f32, s1.weights, s2.weights);
}

test "voxel_skin: single joint => all weight on bone 0" {
    const a = testing.allocator;
    const cyl = try buildCylinder(a, 5, 8, 0.5, 2.0);
    defer a.free(cyl.pos);
    defer a.free(cyl.idx);
    const joints = [_]f32{ 0, 1.0, 0 };
    const parents = [_]i32{-1};
    var sw = try computeSkinWeights(a, cyl.pos, cyl.idx, &joints, &parents, .{ .grid = 28 });
    defer sw.deinit();
    for (0..cyl.pos.len / 3) |v| {
        try testing.expectEqual(@as(u16, 0), sw.joints[v * 4 + 0]);
        try testing.expectApproxEqAbs(@as(f32, 1.0), sw.weights[v * 4 + 0], 1e-5);
    }
}
