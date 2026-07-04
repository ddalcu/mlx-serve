//! UniRig skeleton tokenizer — pure Zig, zero MLX deps. The deterministic "brain"
//! of the stage-1 skeleton decode: coordinate discretization, the finite-state
//! grammar that constrains generation to parseable skeletons, and detokenization
//! (token stream → joints / parents / tails via nearest-previous-joint parenting
//! and tail extrusion). The MLX autoregressive engine (src/unirig_skeleton.zig,
//! in progress) wraps this; this module knows nothing about MLX or the decoder.
//!
//! Verbatim port of the reference `src/tokenizer/tokenizer_part.py` +
//! `src/tokenizer/spec.py:make_skeleton` (UniRig, MIT). Vocabulary offsets and
//! discretize/undiscretize semantics are pinned by
//! `tests/unirig_weights_contract.md` §3 and validated here against the oracle
//! token sequence dumped by `tests/dump_unirig_fixtures.py`.

const std = @import("std");
const testing = std.testing;

/// Token vocabulary (tokenizer_parts_articulationxl_256.yaml + tokenizer_part.py).
pub const Tok = struct {
    pub const num_discrete: u16 = 256; // coordinate bins; ids 0..255
    pub const branch: u16 = 256; // topology jump: next joint reparents to an explicit parent
    pub const bos: u16 = 257;
    pub const eos: u16 = 258;
    pub const pad: u16 = 259;
    pub const spring: u16 = 260; // unnamed part separator
    pub const body: u16 = 261;
    pub const hand: u16 = 262;
    pub const cls_none: u16 = 263;
    pub const cls_vroid: u16 = 264;
    pub const cls_mixamo: u16 = 265;
    pub const cls_articulationxl: u16 = 266;
    pub const vocab_size: usize = 267;
};

/// continuous_range = [-1, 1] (tokenizer_parts_articulationxl_256.yaml).
pub const range_lo: f32 = -1.0;
pub const range_hi: f32 = 1.0;

/// Coordinate → discrete bin: round(clip((t-lo)/(hi-lo)*N, 0, N-1)).
pub fn discretize(t: f32) u16 {
    const n: f32 = @floatFromInt(Tok.num_discrete);
    var v = (t - range_lo) / (range_hi - range_lo) * n;
    v = @round(v);
    if (v < 0) v = 0;
    if (v > n - 1) v = n - 1;
    return @intFromFloat(v);
}

/// Discrete bin → coordinate (bin-center dequant): (t+0.5)/N*(hi-lo)+lo.
pub fn undiscretize(t: u16) f32 {
    const n: f32 = @floatFromInt(Tok.num_discrete);
    const tf: f32 = @floatFromInt(t);
    return (tf + 0.5) / n * (range_hi - range_lo) + range_lo;
}

fn isClsToken(id: u16) bool {
    return id == Tok.cls_vroid or id == Tok.cls_mixamo or id == Tok.cls_articulationxl;
}

fn isPartToken(id: u16) bool {
    return id == Tok.spring or id == Tok.body or id == Tok.hand;
}

// ── grammar finite-state machine (tokenizer_part.next_posible_token) ──────────
//
// A hard finite-state grammar over the token classes that guarantees a
// topologically valid, parseable skeleton: only legal next tokens are unmasked.
// State names mirror the reference verbatim.

pub const State = enum {
    expect_bos,
    expect_cls_or_part_or_joint,
    expect_part_or_joint,
    expect_joint_2,
    expect_joint_3,
    expect_branch_or_part_or_joint,
    expect_joint,
};

pub const Grammar = struct {
    state: State = .expect_bos,

    /// Advance the state by one consumed token. Mirrors the per-id loop in
    /// `next_posible_token` / `bones_in_sequence`.
    pub fn advance(self: *Grammar, id: u16) void {
        switch (self.state) {
            .expect_bos => self.state = .expect_cls_or_part_or_joint, // reference asserts id==bos
            .expect_cls_or_part_or_joint => {
                if (id < Tok.num_discrete) {
                    self.state = .expect_joint_2;
                } else if (id == Tok.cls_none or isClsToken(id)) {
                    self.state = .expect_part_or_joint;
                } else { // a part
                    self.state = .expect_joint;
                }
            },
            .expect_part_or_joint => {
                self.state = if (id < Tok.num_discrete) .expect_joint_2 else .expect_part_or_joint;
            },
            .expect_joint_2 => self.state = .expect_joint_3,
            .expect_joint_3 => self.state = .expect_branch_or_part_or_joint,
            .expect_branch_or_part_or_joint => {
                if (id == Tok.branch) {
                    self.state = .expect_joint;
                } else if (id < Tok.num_discrete) {
                    self.state = .expect_joint_2;
                } else { // a part
                    self.state = .expect_joint;
                }
            },
            .expect_joint => self.state = .expect_joint_2,
        }
    }

    /// Fill `out` (length Tok.vocab_size) with the legal next tokens for the
    /// current state: `out[id] = true` iff `id` may follow. All others false.
    pub fn allowed(self: Grammar, out: *[Tok.vocab_size]bool) void {
        @memset(out, false);
        const addCls = struct {
            fn f(o: *[Tok.vocab_size]bool) void {
                o[Tok.cls_none] = true;
                o[Tok.cls_vroid] = true;
                o[Tok.cls_mixamo] = true;
                o[Tok.cls_articulationxl] = true;
            }
        }.f;
        const addPart = struct {
            fn f(o: *[Tok.vocab_size]bool) void {
                o[Tok.spring] = true;
                o[Tok.body] = true;
                o[Tok.hand] = true;
            }
        }.f;
        const addJoint = struct {
            fn f(o: *[Tok.vocab_size]bool) void {
                var i: u16 = 0;
                while (i < Tok.num_discrete) : (i += 1) o[i] = true;
            }
        }.f;
        switch (self.state) {
            .expect_bos => out[Tok.bos] = true,
            .expect_cls_or_part_or_joint => {
                addCls(out);
                addPart(out);
                addJoint(out);
            },
            .expect_part_or_joint => {
                addPart(out);
                addJoint(out);
                out[Tok.eos] = true;
            },
            .expect_joint_2, .expect_joint_3, .expect_joint => addJoint(out),
            .expect_branch_or_part_or_joint => {
                addJoint(out);
                addPart(out);
                out[Tok.branch] = true;
                out[Tok.eos] = true;
            },
        }
    }
};

/// Legal next-token mask given the sequence so far (must start with bos). Runs a
/// fresh Grammar over `ids` and fills `out`. Matches `next_posible_token`; an
/// empty `ids` yields only bos.
pub fn nextPossibleTokens(ids: []const u16, out: *[Tok.vocab_size]bool) void {
    var g = Grammar{};
    for (ids) |id| g.advance(id);
    g.allowed(out);
}

/// Count completed bones in a token sequence (a bone completes on the third joint
/// coordinate; stops at eos). Mirrors `bones_in_sequence`.
pub fn bonesInSequence(ids: []const u16) usize {
    var g = Grammar{};
    var count: usize = 0;
    for (ids) |id| {
        // a bone completes when leaving expect_joint_3 (the 3rd coordinate)
        if (g.state == .expect_joint_3) count += 1;
        g.advance(id);
        if (id == Tok.eos) break;
    }
    return count;
}

// ── detokenize → joints / parents / tails ─────────────────────────────────────

pub const Joint = [3]f32;

/// A reconstructed skeleton: per-bone head position, parent index (null = root),
/// and tail position (bone direction). Caller owns the slices.
pub const Skeleton = struct {
    joints: []Joint, // bone head positions
    parents: []?usize, // parent bone index, null for the root
    tails: []Joint, // bone tail positions (direction indicator)

    pub fn deinit(self: *Skeleton, alloc: std.mem.Allocator) void {
        alloc.free(self.joints);
        alloc.free(self.parents);
        alloc.free(self.tails);
        self.* = undefined;
    }
};

pub const DetokenizeError = error{ MissingBos, MissingEos, BadToken, OutOfMemory };

fn distSq(a: Joint, b: Joint) f32 {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return dx * dx + dy * dy + dz * dz;
}

/// Parse a full token sequence (bos … eos, optional trailing pad) into a skeleton.
/// Mirrors `TokenizerPart.detokenize` + `make_skeleton` with the AR-inference
/// flags (convert_leaf_bones_to_tails=false, extrude_tail_for_leaf/branch=true,
/// extrude_scale=0.5). Bone NAMES (order.make_names) are a separate follow-up.
pub fn detokenize(alloc: std.mem.Allocator, ids: []const u16) DetokenizeError!Skeleton {
    if (ids.len == 0 or ids[0] != Tok.bos) return error.MissingBos;
    // strip trailing pad, require eos before it
    var end = ids.len;
    while (end > 0 and ids[end - 1] == Tok.pad) end -= 1;
    if (end == 0 or ids[end - 1] != Tok.eos) return error.MissingEos;
    const body_ids = ids[1 .. end - 1]; // strip bos and eos(+pad)

    var joints: std.ArrayList(Joint) = .empty;
    defer joints.deinit(alloc);
    var p_joints: std.ArrayList(Joint) = .empty;
    defer p_joints.deinit(alloc);

    var i: usize = 0;
    var is_branch = false;
    var last_joint: ?Joint = null;
    while (i < body_ids.len) {
        const id = body_ids[i];
        if (id < Tok.num_discrete) {
            if (i + 2 >= body_ids.len) return error.BadToken; // need 3 coords
            var current: Joint = undefined;
            var p_joint: Joint = undefined;
            if (is_branch) {
                if (i + 5 >= body_ids.len) return error.BadToken; // need 6 coords
                p_joint = .{ undiscretize(body_ids[i]), undiscretize(body_ids[i + 1]), undiscretize(body_ids[i + 2]) };
                current = .{ undiscretize(body_ids[i + 3]), undiscretize(body_ids[i + 4]), undiscretize(body_ids[i + 5]) };
                i += 6;
            } else {
                current = .{ undiscretize(body_ids[i]), undiscretize(body_ids[i + 1]), undiscretize(body_ids[i + 2]) };
                p_joint = if (last_joint) |lj| lj else current; // root parents itself
                i += 3;
            }
            try joints.append(alloc, current);
            try p_joints.append(alloc, p_joint);
            last_joint = current;
            is_branch = false;
        } else if (id == Tok.branch) {
            is_branch = true;
            last_joint = null;
            i += 1;
        } else if (id == Tok.spring or isPartToken(id) or id == Tok.cls_none or isClsToken(id)) {
            i += 1; // parts / cls annotations don't affect joint geometry here
        } else {
            return error.BadToken;
        }
    }

    return makeSkeleton(alloc, joints.items, p_joints.items);
}

/// Build bones + parents (nearest-previous-joint) + tails (leaf/branch extrusion,
/// single-child = child head) from the walked joints/p_joints. Port of
/// spec.make_skeleton with the AR-inference flags. The successor-derived tails
/// from the walk are unused: make_skeleton's three passes (leaf / branch /
/// single-child) reassign a tail for EVERY bone, so they always win.
fn makeSkeleton(
    alloc: std.mem.Allocator,
    joints: []const Joint,
    p_joints: []const Joint,
) DetokenizeError!Skeleton {
    const n = joints.len;
    const out_joints = try alloc.alloc(Joint, n);
    errdefer alloc.free(out_joints);
    const parents = try alloc.alloc(?usize, n);
    errdefer alloc.free(parents);
    const tails = try alloc.alloc(Joint, n);
    errdefer alloc.free(tails);
    @memcpy(out_joints, joints);

    // parents: nearest previous joint HEAD to this bone's p_joint (ties → most recent)
    for (0..n) |i| {
        if (i == 0) {
            parents[i] = null;
            continue;
        }
        var best: f32 = std.math.floatMax(f32);
        var pid: ?usize = null;
        var j: usize = i;
        while (j > 0) {
            j -= 1;
            const d = distSq(joints[j], p_joints[i]);
            if (d < best) {
                best = d;
                pid = j;
            }
        }
        parents[i] = pid;
    }

    // child counts
    const child_count = try alloc.alloc(usize, n);
    defer alloc.free(child_count);
    const first_child = try alloc.alloc(?usize, n);
    defer alloc.free(first_child);
    @memset(child_count, 0);
    @memset(first_child, null);
    for (0..n) |i| {
        if (parents[i]) |p| {
            if (child_count[p] == 0) first_child[p] = i;
            child_count[p] += 1;
        }
    }

    const extrude_scale: f32 = 0.5;
    for (0..n) |i| {
        if (child_count[i] == 0) {
            // leaf → extrude a tail along parent→head direction
            const head = joints[i];
            var d: Joint = .{ 0, 0, 1 };
            if (parents[i]) |p| {
                d = .{ head[0] - joints[p][0], head[1] - joints[p][1], head[2] - joints[p][2] };
                if (@sqrt(distSq(head, joints[p])) <= 1e-9) d = .{ 0, 0, 1 };
            }
            tails[i] = .{ head[0] + d[0] * extrude_scale, head[1] + d[1] * extrude_scale, head[2] + d[2] * extrude_scale };
        } else if (child_count[i] == 1) {
            // single child → tail = child head
            tails[i] = joints[first_child[i].?];
        } else {
            // branch (>1 children) → extrude along parent→head (root: +Z by mean child length)
            const head = joints[i];
            if (parents[i]) |p| {
                var d: Joint = .{ head[0] - joints[p][0], head[1] - joints[p][1], head[2] - joints[p][2] };
                if (@sqrt(distSq(head, joints[p])) <= 1e-9) d = .{ 0, 0, 1 };
                tails[i] = .{ head[0] + d[0] * extrude_scale, head[1] + d[1] * extrude_scale, head[2] + d[2] * extrude_scale };
            } else {
                // root branch: mean distance to children, extrude +Z
                var av: f32 = 0;
                var cnt: f32 = 0;
                for (0..n) |c| {
                    if (parents[c] == 0) {
                        av += @sqrt(distSq(head, joints[c]));
                        cnt += 1;
                    }
                }
                if (cnt > 0) av /= cnt;
                tails[i] = .{ head[0], head[1], head[2] + extrude_scale * av };
            }
        }
    }

    return .{ .joints = out_joints, .parents = parents, .tails = tails };
}

// ── tests (hermetic; no weights, no MLX) ──────────────────────────────────────

test "unirig tokenizer: discretize/undiscretize round-trip and edges" {
    try testing.expectEqual(@as(u16, 128), discretize(0.0));
    try testing.expectEqual(@as(u16, 0), discretize(-1.0));
    try testing.expectEqual(@as(u16, 255), discretize(1.0));
    try testing.expectEqual(@as(u16, 0), discretize(-5.0)); // clamped
    try testing.expectEqual(@as(u16, 255), discretize(5.0)); // clamped
    // bin-center dequant matches (t+0.5)/256*2-1
    try testing.expect(@abs(undiscretize(128) - 0.00390625) < 1e-6);
    try testing.expect(@abs(undiscretize(0) - (-0.99609375)) < 1e-6);
    // round-trip within one bin half-width
    var t: f32 = -0.9;
    while (t < 0.9) : (t += 0.05) {
        const back = undiscretize(discretize(t));
        try testing.expect(@abs(back - t) < (2.0 / 256.0));
    }
}

test "unirig tokenizer: grammar first-token allows cls/part/joint after bos" {
    var mask: [Tok.vocab_size]bool = undefined;
    // empty → only bos
    nextPossibleTokens(&[_]u16{}, &mask);
    try testing.expect(mask[Tok.bos]);
    var only_bos = true;
    for (0..Tok.vocab_size) |k| {
        if (k != Tok.bos and mask[k]) only_bos = false;
    }
    try testing.expect(only_bos);

    // after [bos] → cls tokens, part tokens, all 256 joints (NOT eos/branch/pad)
    nextPossibleTokens(&[_]u16{Tok.bos}, &mask);
    try testing.expect(mask[Tok.cls_articulationxl] and mask[Tok.cls_none]);
    try testing.expect(mask[Tok.spring] and mask[Tok.body] and mask[Tok.hand]);
    try testing.expect(mask[0] and mask[255]);
    try testing.expect(!mask[Tok.eos] and !mask[Tok.branch] and !mask[Tok.pad] and !mask[Tok.bos]);
}

test "unirig tokenizer: grammar reaches branch/eos after a completed bone" {
    var mask: [Tok.vocab_size]bool = undefined;
    // bos, cls, spring, then a full joint triple → expect_branch_or_part_or_joint
    const seq = [_]u16{ Tok.bos, Tok.cls_articulationxl, Tok.spring, 128, 128, 25 };
    nextPossibleTokens(&seq, &mask);
    try testing.expect(mask[Tok.branch]); // may start a topology jump
    try testing.expect(mask[Tok.eos]); // may terminate
    try testing.expect(mask[0] and mask[255]); // or continue with another joint
    try testing.expect(mask[Tok.spring]); // or a new part
    try testing.expect(!mask[Tok.bos] and !mask[Tok.cls_articulationxl]);

    // mid-joint (after only 1 coord) → ONLY joints
    const mid = [_]u16{ Tok.bos, Tok.cls_articulationxl, 128 };
    nextPossibleTokens(&mid, &mask);
    try testing.expect(mask[0] and mask[255]);
    try testing.expect(!mask[Tok.eos] and !mask[Tok.branch] and !mask[Tok.spring]);
}

test "unirig tokenizer: detokenize the oracle E2E token sequence into a 5-bone chain" {
    // The exact greedy-decode output dumped by tests/dump_unirig_fixtures.py for
    // the synthetic sphere: bos, cls, spring, then 5 joint triples, eos.
    const a = testing.allocator;
    const ids = [_]u16{ 257, 266, 260, 128, 128, 25, 128, 128, 40, 128, 128, 81, 128, 128, 128, 128, 128, 169, 258 };

    try testing.expectEqual(@as(usize, 5), bonesInSequence(&ids));

    var skel = try detokenize(a, &ids);
    defer skel.deinit(a);
    try testing.expectEqual(@as(usize, 5), skel.joints.len);
    // a pure chain: root, then each parented to the immediately previous joint
    try testing.expectEqual(@as(?usize, null), skel.parents[0]);
    try testing.expectEqual(@as(?usize, 0), skel.parents[1]);
    try testing.expectEqual(@as(?usize, 1), skel.parents[2]);
    try testing.expectEqual(@as(?usize, 2), skel.parents[3]);
    try testing.expectEqual(@as(?usize, 3), skel.parents[4]);
    // first joint z-coord = undiscretize(25)
    try testing.expect(@abs(skel.joints[0][2] - undiscretize(25)) < 1e-6);
    try testing.expect(@abs(skel.joints[0][0] - undiscretize(128)) < 1e-6);
    // last joint (leaf) tail extruded past its head along parent→head
    const leaf_head = skel.joints[4];
    const leaf_tail = skel.tails[4];
    try testing.expect(!std.meta.eql(leaf_head, leaf_tail));
    // single-child bones: tail == child head (bone 0's tail is bone 1's head)
    try testing.expect(std.meta.eql(skel.tails[0], skel.joints[1]));
}

test "unirig tokenizer: detokenize rejects malformed sequences" {
    const a = testing.allocator;
    try testing.expectError(error.MissingBos, detokenize(a, &[_]u16{ 128, 128, 128, 258 }));
    try testing.expectError(error.MissingEos, detokenize(a, &[_]u16{ 257, 128, 128, 128 }));
}

test "unirig tokenizer: branch token reparents (non-chain topology)" {
    // bos, cls, joint0(root), branch, <parent coords>, <joint1 coords>, eos.
    // The branch supplies an explicit parent position for joint1.
    const a = testing.allocator;
    const ids = [_]u16{
        Tok.bos,        Tok.cls_articulationxl,
        128,            128,
        128, // joint0 (root) at (0,0,0)-ish
        Tok.branch,     10,
        10,             10, // explicit parent pos
        200,            200,
        200, // joint1 head
        Tok.eos,
    };
    var skel = try detokenize(a, &ids);
    defer skel.deinit(a);
    try testing.expectEqual(@as(usize, 2), skel.joints.len);
    // joint1's p_joint (10,10,10) is nearest to joint0's head → parent 0
    try testing.expectEqual(@as(?usize, null), skel.parents[0]);
    try testing.expectEqual(@as(?usize, 0), skel.parents[1]);
}
