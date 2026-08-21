//! SeedVR2 NaDiT weight manifest — the exact tensor names and shapes the 3B
//! checkpoint must contain, as pure string/integer logic with zero MLX.
//!
//! Transcribed from ByteDance-Seed/SeedVR `models/dit_v2/*` (Apache-2.0); spec
//! in `docs/seedvr2-arch.md` §1.
//!
//! WHY THIS EXISTS AS A SEPARATE, TESTED LAYER: `MMModule` stores its weights
//! under `.vid`+`.txt` for layers 0..9 and under a single `.all` for layers
//! 10..31 (`shared_weights = not (i < mm_layers)`). Every tensor SHAPE is
//! identical either way. A loader that picks one naming for all 32 layers finds
//! nothing for 22 of them and — depending on how forgiving the weight map is —
//! either loads zeros or silently reuses a stale buffer. There is no shape
//! check anywhere downstream that can catch it, and the model still emits an
//! image. So the boundary is asserted here, by name, before any weight is read.
//!
//! This manifest is a TRANSCRIPTION, not an oracle: it was derived by reading
//! the reference module tree, not by dumping `state_dict().keys()` from the
//! real checkpoint. `tests/dump_seedvr2_fixtures.py --manifest` emits the true
//! key list; `validate` is what reconciles the two, and when they disagree the
//! checkpoint wins.

const std = @import("std");

/// 3B geometry. Everything derived, nothing hardcoded twice.
/// Which MLP a block runs. The 3B is gated SwiGLU; the 7B is a plain
/// GELU MLP, which differs in TENSOR SET (two linears WITH biases, no
/// `proj_in_gate`), in hidden width, and in the forward math.
pub const MlpType = enum { swiglu, normal };

/// Which basis the rope frequency table was built on. We LOAD the table from
/// the checkpoint either way, so this does not change the frequencies — it
/// changes the POSITIONS they multiply: `lang` walks integer positions, while
/// `pixel` normalises each axis onto `linspace(-1, 1, extent)`.
pub const RopeFreqs = enum { lang, pixel };

pub const Config = struct {
    vid_dim: u32 = 2560,
    txt_in_dim: u32 = 5120,
    heads: u32 = 20,
    head_dim: u32 = 128,
    num_layers: u32 = 32,
    /// Layers with index < this get separate vid/txt weights.
    mm_layers: u32 = 10,
    expand_ratio: u32 = 4,
    /// SwiGLU hidden is rounded UP to a multiple of this.
    mlp_multiple_of: u32 = 256,
    vid_in_channels: u32 = 33,
    vid_out_channels: u32 = 16,
    patch_t: u32 = 1,
    patch_h: u32 = 2,
    patch_w: u32 = 2,
    sinusoidal_dim: u32 = 256,
    rope_dim: u32 = 128,
    /// 3B: gated SwiGLU. 7B: plain GELU with biases.
    mlp_type: MlpType = .swiglu,
    /// 3B applies rope to both streams; the 7B rotates video only.
    rope_on_text: bool = true,
    rope_freqs_for: RopeFreqs = .lang,
    /// 3B closes with `vid_out_norm` + an `out_shift`/`out_scale` modulation.
    /// The 7B ships NEITHER tensor and skips both.
    use_output_ada: bool = true,
    /// 3B's last layer runs its txt stream through no MLP. The 7B's does.
    last_layer_vid_only: bool = true,

    pub fn txtDim(self: Config) u32 {
        return self.vid_dim;
    }

    /// `emb_dim == 6 * dim` — 2 layers (attn, mlp) x 3 (shift, scale, gate).
    /// `AdaSingle` asserts this; so do we.
    pub fn embDim(self: Config) u32 {
        return 6 * self.vid_dim;
    }

    pub fn innerDim(self: Config) u32 {
        return self.heads * self.head_dim;
    }

    pub fn qkvDim(self: Config) u32 {
        return self.innerDim() * 3;
    }

    /// SwiGLU: `ceil_to_256(int(2 * dim * expand_ratio / 3))` — the two thirds
    /// is what keeps a GATED MLP's parameter count level with a plain one.
    /// Plain GELU: the ungated `dim * expand_ratio`, no rounding, which is why
    /// the 7B's is 12288 where the same dim under SwiGLU would give 8192.
    pub fn mlpHidden(self: Config) u32 {
        if (self.mlp_type == .normal) return self.vid_dim * self.expand_ratio;
        const raw = (2 * self.vid_dim * self.expand_ratio) / 3; // int() truncates
        const m = self.mlp_multiple_of;
        return ((raw + m - 1) / m) * m;
    }

    /// rope_dim // 3 (three axes), i.e. 128//3 = 42 in the 3B config.
    pub fn ropeAxisDim(self: Config) u32 {
        return self.rope_dim / 3;
    }

    /// Patchified input width: `in_channels * t * h * w`.
    pub fn patchInDim(self: Config) u32 {
        return self.vid_in_channels * self.patch_t * self.patch_h * self.patch_w;
    }

    pub fn patchOutDim(self: Config) u32 {
        return self.vid_out_channels * self.patch_t * self.patch_h * self.patch_w;
    }
};

/// Number of stored rope frequencies per layer: `arange(0, axis_dim, 2)` has
/// `ceil(axis_dim/2)` entries, and rotary_embedding_torch keeps `[:dim//2]`.
pub fn ropeFreqCount(cfg: Config) u32 {
    return cfg.ropeAxisDim() / 2;
}

/// Which `MMModule` storage a layer uses. THE boundary this file exists for.
pub const Branch = enum {
    /// Layers < mm_layers: separate `.vid` and `.txt` submodules.
    split,
    /// Layers >= mm_layers: one `.all` submodule serving both streams.
    shared,
};

pub fn branchForLayer(cfg: Config, layer: u32) Branch {
    return if (layer < cfg.mm_layers) .split else .shared;
}

/// The submodule key(s) a layer's MMModule weights live under.
/// `.split` yields two ("vid", "txt"); `.shared` yields one ("all").
pub fn branchKeys(b: Branch) []const []const u8 {
    return switch (b) {
        .split => &.{ "vid", "txt" },
        .shared => &.{"all"},
    };
}

/// True when this layer's TXT stream skips the MLP entirely.
///
/// `is_last_layer` sets `vid_only=True` on `mlp_norm`, `mlp` and `ada`. Note
/// this is a FORWARD-PATH fact, not a naming one: by layer 31 the branch is
/// already `.shared`, so the tensors are named `.all` exactly like layers
/// 10..30. Nothing is missing from the checkpoint — the txt stream simply is
/// not run through them. A loader looking for absent tensors here is chasing a
/// bug that does not exist; a forward that runs txt through the last MLP is
/// the real one.
pub fn txtSkipsMlp(cfg: Config, layer: u32) bool {
    return cfg.last_layer_vid_only and layer == cfg.num_layers - 1;
}

/// Read a pack's `config.json` into a `Config`.
///
/// The geometry lives in `transformer_overrides`, which is the argument list
/// the reference transformer's constructor takes — mflux passes it straight to
/// `SeedVR2Transformer(**cfg["transformer_overrides"])`, and the packs are
/// exported against exactly that. An ABSENT block leaves every field at the 3B
/// default, which is what the two 3B packs rely on: `mlx-community/SeedVR2-3B-mlx`
/// declares an EMPTY `transformer_overrides` and our own converter's config
/// carries the geometry as flat top-level keys instead. Both still load.
///
/// Unknown keys are ignored rather than refused: this is a constructor
/// signature, and a future field we do not model yet must not make an existing
/// pack unloadable. Fields we DO model but cannot honour are the loader's
/// problem, not the parser's — it reports what the file says.
pub fn configFromJson(allocator: std.mem.Allocator, bytes: []const u8) Config {
    var cfg = Config{};
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return cfg;
    defer parsed.deinit();
    if (parsed.value != .object) return cfg;
    const root = parsed.value.object;

    // Our own converter writes the geometry flat; mflux-shaped packs nest it.
    applyFields(&cfg, root);
    if (root.get("transformer_overrides")) |ov| {
        if (ov == .object) applyFields(&cfg, ov.object);
    }
    return cfg;
}

fn applyFields(cfg: *Config, o: std.json.ObjectMap) void {
    readU32(o, "vid_dim", &cfg.vid_dim);
    readU32(o, "txt_in_dim", &cfg.txt_in_dim);
    readU32(o, "heads", &cfg.heads);
    readU32(o, "head_dim", &cfg.head_dim);
    readU32(o, "num_layers", &cfg.num_layers);
    readU32(o, "mm_layers", &cfg.mm_layers);
    readU32(o, "expand_ratio", &cfg.expand_ratio);
    readU32(o, "vid_in_channels", &cfg.vid_in_channels);
    readU32(o, "vid_out_channels", &cfg.vid_out_channels);
    readU32(o, "rope_dim", &cfg.rope_dim);
    readBool(o, "rope_on_text", &cfg.rope_on_text);
    readBool(o, "use_output_ada", &cfg.use_output_ada);
    readBool(o, "last_layer_vid_only", &cfg.last_layer_vid_only);
    if (o.get("mlp_type")) |v| {
        if (v == .string and std.mem.eql(u8, v.string, "normal")) cfg.mlp_type = .normal;
        if (v == .string and std.mem.eql(u8, v.string, "swiglu")) cfg.mlp_type = .swiglu;
    }
    if (o.get("rope_freqs_for")) |v| {
        if (v == .string and std.mem.eql(u8, v.string, "pixel")) cfg.rope_freqs_for = .pixel;
        if (v == .string and std.mem.eql(u8, v.string, "lang")) cfg.rope_freqs_for = .lang;
    }
    // `patch_size` is a 3-list, and the ONLY one of these the 3B config states
    // as an array. A malformed entry leaves the default rather than a 0 patch.
    if (o.get("patch_size")) |v| {
        if (v == .array and v.array.items.len == 3) {
            var dims: [3]u32 = .{ cfg.patch_t, cfg.patch_h, cfg.patch_w };
            for (v.array.items, 0..) |it, i| {
                if (it == .integer and it.integer > 0) dims[i] = @intCast(it.integer);
            }
            cfg.patch_t = dims[0];
            cfg.patch_h = dims[1];
            cfg.patch_w = dims[2];
        }
    }
}

fn readU32(o: std.json.ObjectMap, key: []const u8, out: *u32) void {
    const v = o.get(key) orelse return;
    if (v == .integer and v.integer > 0) out.* = @intCast(v.integer);
}

fn readBool(o: std.json.ObjectMap, key: []const u8, out: *bool) void {
    const v = o.get(key) orelse return;
    if (v == .bool) out.* = v.bool;
}

/// One expected tensor.
pub const Tensor = struct {
    name: []const u8,
    /// Up to 2 dims; `rank` says how many are meaningful.
    shape: [2]u32,
    rank: u8,
};

/// Append every tensor the 3B NaDiT checkpoint must contain. Caller owns the
/// list and the names (all names are allocated).
pub fn manifest(allocator: std.mem.Allocator, cfg: Config) ![]Tensor {
    var out: std.ArrayList(Tensor) = .empty;
    errdefer {
        for (out.items) |t| allocator.free(t.name);
        out.deinit(allocator);
    }

    const d = cfg.vid_dim;

    // ---- stem ----
    try push(&out, allocator, "vid_in.proj.weight", .{ d, cfg.patchInDim() }, 2);
    try push(&out, allocator, "vid_in.proj.bias", .{ d, 0 }, 1);
    // txt_in is a Linear only because txt_in_dim (5120) != txt_dim (2560); if
    // they were equal the reference substitutes nn.Identity and these two
    // tensors DO NOT EXIST.
    if (cfg.txt_in_dim != cfg.txtDim()) {
        try push(&out, allocator, "txt_in.weight", .{ cfg.txtDim(), cfg.txt_in_dim }, 2);
        try push(&out, allocator, "txt_in.bias", .{ cfg.txtDim(), 0 }, 1);
    }
    try push(&out, allocator, "emb_in.proj_in.weight", .{ d, cfg.sinusoidal_dim }, 2);
    try push(&out, allocator, "emb_in.proj_in.bias", .{ d, 0 }, 1);
    try push(&out, allocator, "emb_in.proj_hid.weight", .{ d, d }, 2);
    try push(&out, allocator, "emb_in.proj_hid.bias", .{ d, 0 }, 1);
    try push(&out, allocator, "emb_in.proj_out.weight", .{ cfg.embDim(), d }, 2);
    try push(&out, allocator, "emb_in.proj_out.bias", .{ cfg.embDim(), 0 }, 1);

    // ---- blocks ----
    var i: u32 = 0;
    while (i < cfg.num_layers) : (i += 1) {
        // The rope frequency table is a registered BUFFER, stored per layer in
        // the checkpoint rather than recomputed at load. It sits OUTSIDE the
        // MMModule branch (one copy per layer, not per stream) and its double
        // `rope.rope` prefix is the RotaryEmbeddingBase wrapper around
        // rotary_embedding_torch's own module.
        //
        // Its length is the whole mmrope3d derivation in one number:
        // RotaryEmbedding(dim = rope_dim/3 = 128/3 = 42) keeps
        // arange(0, 42, 2)[:21] -> 21 freqs, each later repeated twice to give
        // 42 values per axis, 126 across three axes. head_dim is 128, so the
        // last two dims are NEVER rotated. A [64] table here would mean the
        // rope was built over head_dim instead of rope_dim/3.
        try pushFmt(&out, allocator, "blocks.{d}.attn.rope.rope.freqs", .{i}, .{ ropeFreqCount(cfg), 0 }, 1);
        const keys = branchKeys(branchForLayer(cfg, i));
        for (keys) |k| {
            // attn_norm / mlp_norm are elementwise_affine=FALSE — they hold NO
            // parameters. All per-channel affine lives in `ada`. A checkpoint
            // carrying blocks.N.attn_norm.*.weight means the norm was built
            // affine and the ada params are being double-counted.
            try pushFmt(&out, allocator, "blocks.{d}.attn.proj_qkv.{s}.weight", .{ i, k }, .{ cfg.qkvDim(), d }, 2);
            // qk_bias=False -> proj_qkv has NO bias.
            try pushFmt(&out, allocator, "blocks.{d}.attn.proj_out.{s}.weight", .{ i, k }, .{ d, cfg.innerDim() }, 2);
            try pushFmt(&out, allocator, "blocks.{d}.attn.proj_out.{s}.bias", .{ i, k }, .{ d, 0 }, 1);
            // qk_norm is per-HEAD_DIM, not per-model-dim.
            try pushFmt(&out, allocator, "blocks.{d}.attn.norm_q.{s}.weight", .{ i, k }, .{ cfg.head_dim, 0 }, 1);
            try pushFmt(&out, allocator, "blocks.{d}.attn.norm_k.{s}.weight", .{ i, k }, .{ cfg.head_dim, 0 }, 1);
            // SwiGLU, all bias-free. proj_in_gate is the SiLU'd branch.
            // The plain arm has no gate and DOES carry biases, so this is a
            // different tensor set, not just a different forward.
            if (cfg.mlp_type == .swiglu)
                try pushFmt(&out, allocator, "blocks.{d}.mlp.{s}.proj_in_gate.weight", .{ i, k }, .{ cfg.mlpHidden(), d }, 2);
            try pushFmt(&out, allocator, "blocks.{d}.mlp.{s}.proj_in.weight", .{ i, k }, .{ cfg.mlpHidden(), d }, 2);
            try pushFmt(&out, allocator, "blocks.{d}.mlp.{s}.proj_out.weight", .{ i, k }, .{ d, cfg.mlpHidden() }, 2);
            if (cfg.mlp_type == .normal) {
                try pushFmt(&out, allocator, "blocks.{d}.mlp.{s}.proj_in.bias", .{ i, k }, .{ cfg.mlpHidden(), 0 }, 1);
                try pushFmt(&out, allocator, "blocks.{d}.mlp.{s}.proj_out.bias", .{ i, k }, .{ d, 0 }, 1);
            }
            // AdaSingle: layers ["attn","mlp"] x modes ["in","out"].
            for ([_][]const u8{ "attn", "mlp" }) |layer_name| {
                for ([_][]const u8{ "shift", "scale", "gate" }) |p| {
                    try pushFmt(&out, allocator, "blocks.{d}.ada.{s}.{s}_{s}", .{ i, k, layer_name, p }, .{ d, 0 }, 1);
                }
            }
        }
    }

    // ---- head ----
    // The output norm and its modulation are ONE switch, not two: the
    // reference builds `vid_out_norm`, `out_shift` and `out_scale` inside a
    // single `if use_output_ada`, and the 7B ships none of the three.
    if (cfg.use_output_ada) {
        // vid_out_norm IS affine (elementwise_affine=True at the call site).
        try push(&out, allocator, "vid_out_norm.weight", .{ d, 0 }, 1);
        // vid_out_ada uses modes=["in"] only -> shift+scale, NO gate.
        try push(&out, allocator, "vid_out_ada.out_shift", .{ d, 0 }, 1);
        try push(&out, allocator, "vid_out_ada.out_scale", .{ d, 0 }, 1);
    }
    try push(&out, allocator, "vid_out.proj.weight", .{ cfg.patchOutDim(), d }, 2);
    try push(&out, allocator, "vid_out.proj.bias", .{ cfg.patchOutDim(), 0 }, 1);

    return out.toOwnedSlice(allocator);
}

pub fn freeManifest(allocator: std.mem.Allocator, m: []Tensor) void {
    for (m) |t| allocator.free(t.name);
    allocator.free(m);
}

fn push(out: *std.ArrayList(Tensor), a: std.mem.Allocator, name: []const u8, shape: [2]u32, rank: u8) !void {
    try out.append(a, .{ .name = try a.dupe(u8, name), .shape = shape, .rank = rank });
}

fn pushFmt(out: *std.ArrayList(Tensor), a: std.mem.Allocator, comptime fmt: []const u8, args: anytype, shape: [2]u32, rank: u8) !void {
    try out.append(a, .{ .name = try std.fmt.allocPrint(a, fmt, args), .shape = shape, .rank = rank });
}

// ════════════════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════════════════

const testing = std.testing;

fn hasName(m: []const Tensor, name: []const u8) bool {
    for (m) |t| if (std.mem.eql(u8, t.name, name)) return true;
    return false;
}

fn shapeOf(m: []const Tensor, name: []const u8) ?[2]u32 {
    for (m) |t| if (std.mem.eql(u8, t.name, name)) return t.shape;
    return null;
}

test "seedvr2 7b: the config's transformer_overrides decide the geometry" {
    const a = testing.allocator;
    // benc0/SeedVR2-7B-mlx-int8's config.json, verbatim.
    const seven_b =
        \\{"model_type":"seedvr2","variant":"seedvr2-7b","transformer_overrides":{
        \\"vid_dim":3072,"heads":24,"num_layers":36,"mm_layers":36,"rope_dim":64,
        \\"rope_on_text":false,"rope_freqs_for":"pixel","mlp_type":"normal",
        \\"use_output_ada":false,"last_layer_vid_only":false},
        \\"pos_emb_shape":[58,5120],"dtype":"float16","quantization":{"bits":8,"group_size":64}}
    ;
    const cfg = configFromJson(a, seven_b);
    try testing.expectEqual(@as(u32, 3072), cfg.vid_dim);
    try testing.expectEqual(@as(u32, 24), cfg.heads);
    try testing.expectEqual(@as(u32, 128), cfg.head_dim); // 3072/24, not stated
    try testing.expectEqual(@as(u32, 36), cfg.num_layers);
    try testing.expectEqual(@as(u32, 36), cfg.mm_layers); // every layer split
    try testing.expectEqual(@as(u32, 64), cfg.rope_dim);
    try testing.expectEqual(false, cfg.rope_on_text);
    try testing.expectEqual(RopeFreqs.pixel, cfg.rope_freqs_for);
    try testing.expectEqual(MlpType.normal, cfg.mlp_type);
    try testing.expectEqual(false, cfg.use_output_ada);
    try testing.expectEqual(false, cfg.last_layer_vid_only);

    // The derived numbers the checkpoint can be checked against:
    // rope_dim 64 -> axis 21 -> 10 stored freqs (the 3B's are 21), and a
    // PLAIN mlp is dim*4 = 12288 where SwiGLU on the same dim gives 8192.
    try testing.expectEqual(@as(u32, 10), ropeFreqCount(cfg));
    try testing.expectEqual(@as(u32, 12288), cfg.mlpHidden());
    var swiglu = cfg;
    swiglu.mlp_type = .swiglu;
    try testing.expectEqual(@as(u32, 8192), swiglu.mlpHidden());

    // mm_layers == num_layers means NO shared layer exists at all.
    try testing.expectEqual(Branch.split, branchForLayer(cfg, 35));
    // ...and with last_layer_vid_only off, the last layer keeps its txt MLP.
    try testing.expect(!txtSkipsMlp(cfg, 35));
}

test "seedvr2 3b: a config with no overrides is the 3B, byte for byte" {
    const a = testing.allocator;
    // mlx-community/SeedVR2-3B-mlx declares an EMPTY overrides block; if that
    // were read as "unknown geometry" both 3B packs would stop loading.
    const three_b =
        \\{"model_type":"seedvr2","variant":"seedvr2-3b","transformer_overrides":{},
        \\"pos_emb_shape":[58,5120],"dtype":"float16"}
    ;
    try testing.expectEqual(Config{}, configFromJson(a, three_b));
    // Junk, an absent file, or a config that is not even an object must all
    // fall back rather than half-apply.
    try testing.expectEqual(Config{}, configFromJson(a, "not json at all"));
    try testing.expectEqual(Config{}, configFromJson(a, "[1,2,3]"));
    try testing.expectEqual(Config{}, configFromJson(a, "{}"));

    // Our own converter writes the geometry FLAT, with no overrides block.
    const ours =
        \\{"model_type":"seedvr2","vid_dim":2560,"num_layers":32,"mm_layers":10,
        \\"heads":20,"head_dim":128,"patch_size":[1,2,2],"rope_dim":128}
    ;
    try testing.expectEqual(Config{}, configFromJson(a, ours));

    // An unmodelled key is ignored, not fatal: this block is a constructor
    // signature and will grow.
    const future =
        \\{"transformer_overrides":{"vid_dim":3072,"some_new_switch":true}}
    ;
    try testing.expectEqual(@as(u32, 3072), configFromJson(a, future).vid_dim);
}

test "seedvr2 7b: the manifest names the tensor set the 7B pack actually ships" {
    const a = testing.allocator;
    var cfg = Config{
        .vid_dim = 3072,
        .heads = 24,
        .num_layers = 36,
        .mm_layers = 36,
        .rope_dim = 64,
        .mlp_type = .normal,
        .use_output_ada = false,
        .last_layer_vid_only = false,
    };
    cfg.rope_on_text = false;
    cfg.rope_freqs_for = .pixel;

    const m = try manifest(a, cfg);
    defer freeManifest(a, m);

    // Plain MLP: two linears WITH biases, and no gate anywhere. Checked on the
    // last layer too, since that is where the 3B's txt stream has no MLP at
    // all and an unguarded `last_layer_vid_only` would drop these.
    for ([_][]const u8{ "vid", "txt" }) |k| {
        for ([_]u32{ 0, 35 }) |i| {
            const inw = try std.fmt.allocPrint(a, "blocks.{d}.mlp.{s}.proj_in.weight", .{ i, k });
            defer a.free(inw);
            const inb = try std.fmt.allocPrint(a, "blocks.{d}.mlp.{s}.proj_in.bias", .{ i, k });
            defer a.free(inb);
            const outb = try std.fmt.allocPrint(a, "blocks.{d}.mlp.{s}.proj_out.bias", .{ i, k });
            defer a.free(outb);
            const gate = try std.fmt.allocPrint(a, "blocks.{d}.mlp.{s}.proj_in_gate.weight", .{ i, k });
            defer a.free(gate);
            try testing.expect(hasName(m, inw));
            try testing.expect(hasName(m, inb));
            try testing.expect(hasName(m, outb));
            try testing.expect(!hasName(m, gate));
        }
    }

    // mm_layers == num_layers: every layer is split, so no `.all` name exists.
    for (m) |t| try testing.expect(std.mem.indexOf(u8, t.name, ".all.") == null);

    // The output head the 7B does not ship.
    try testing.expect(!hasName(m, "vid_out_norm.weight"));
    try testing.expect(!hasName(m, "vid_out_ada.out_shift"));
    try testing.expect(!hasName(m, "vid_out_ada.out_scale"));
    // ...but the projection itself is still there.
    try testing.expect(hasName(m, "vid_out.proj.weight"));

    // The 3B keeps every one of those, so the switches cannot be no-ops.
    const three = try manifest(a, Config{});
    defer freeManifest(a, three);
    try testing.expect(hasName(three, "vid_out_norm.weight"));
    try testing.expect(hasName(three, "blocks.0.mlp.vid.proj_in_gate.weight"));
    try testing.expect(!hasName(three, "blocks.0.mlp.vid.proj_in.bias"));
}

test "the shared_weights boundary sits at layer 10 exactly" {
    // TRAP #1. Layers 0..9 are split, 10..31 shared. Every shape matches under
    // either naming, so this boundary is the only thing that can catch a
    // loader using one convention for all 32 layers.
    const cfg: Config = .{};
    try testing.expectEqual(Branch.split, branchForLayer(cfg, 0));
    try testing.expectEqual(Branch.split, branchForLayer(cfg, 9));
    try testing.expectEqual(Branch.shared, branchForLayer(cfg, 10));
    try testing.expectEqual(Branch.shared, branchForLayer(cfg, 31));
}

test "manifest names layers 0-9 vid/txt and 10-31 all" {
    const cfg: Config = .{};
    const m = try manifest(testing.allocator, cfg);
    defer freeManifest(testing.allocator, m);

    // Split side: both branches present, `.all` absent.
    try testing.expect(hasName(m, "blocks.0.attn.proj_qkv.vid.weight"));
    try testing.expect(hasName(m, "blocks.0.attn.proj_qkv.txt.weight"));
    try testing.expect(!hasName(m, "blocks.0.attn.proj_qkv.all.weight"));
    try testing.expect(hasName(m, "blocks.9.mlp.txt.proj_in_gate.weight"));

    // Shared side: `.all` only.
    try testing.expect(hasName(m, "blocks.10.attn.proj_qkv.all.weight"));
    try testing.expect(!hasName(m, "blocks.10.attn.proj_qkv.vid.weight"));
    try testing.expect(!hasName(m, "blocks.10.attn.proj_qkv.txt.weight"));
    try testing.expect(hasName(m, "blocks.31.ada.all.mlp_gate"));
}

test "the last layer's txt stream skips the MLP but its tensors are still named .all" {
    // A subtlety worth pinning: `is_last_layer` makes the txt stream skip
    // mlp_norm/mlp/ada-mlp in FORWARD, but by layer 31 the branch is already
    // shared, so nothing is absent from the checkpoint. Looking for missing
    // tensors here is chasing a bug that does not exist.
    const cfg: Config = .{};
    try testing.expect(txtSkipsMlp(cfg, 31));
    try testing.expect(!txtSkipsMlp(cfg, 30));
    try testing.expect(!txtSkipsMlp(cfg, 0));

    const m = try manifest(testing.allocator, cfg);
    defer freeManifest(testing.allocator, m);
    try testing.expect(hasName(m, "blocks.31.mlp.all.proj_out.weight"));
    try testing.expect(!hasName(m, "blocks.31.mlp.txt.proj_out.weight"));
}

test "oracle: rope freqs is a stored per-layer buffer of 21 values" {
    // ORACLE, from the real checkpoint's key list. 21 is the whole mmrope3d
    // derivation compressed into one observable number: rope_dim/3 = 42,
    // arange(0,42,2)[:21] = 21 freqs, x2 = 42 per axis, x3 axes = 126 of the
    // 128 head dims rotated. A [64] here would mean the rope was built over
    // head_dim; a [42] would mean the `[:dim//2]` slice was dropped.
    const cfg: Config = .{};
    try testing.expectEqual(@as(u32, 42), cfg.ropeAxisDim());
    try testing.expectEqual(@as(u32, 21), ropeFreqCount(cfg));

    const m = try manifest(testing.allocator, cfg);
    defer freeManifest(testing.allocator, m);
    // One per LAYER, not per branch — layer 0 is split but has a single table.
    try testing.expectEqual([2]u32{ 21, 0 }, shapeOf(m, "blocks.0.attn.rope.rope.freqs").?);
    try testing.expect(hasName(m, "blocks.31.attn.rope.rope.freqs"));
    try testing.expect(!hasName(m, "blocks.0.attn.rope.vid.rope.freqs"));
    var n: usize = 0;
    for (m) |t| if (std.mem.indexOf(u8, t.name, "rope.rope.freqs") != null) {
        n += 1;
    };
    try testing.expectEqual(@as(usize, 32), n);
}

test "SwiGLU hidden rounds 6826 up to 6912" {
    // int(2*2560*4/3) = 6826 -> ceil to a multiple of 256 -> 6912. Getting the
    // truncation or the rounding wrong gives a shape mismatch at load, which
    // is the GOOD failure — but only if the number is right here.
    const cfg: Config = .{};
    try testing.expectEqual(@as(u32, 6912), cfg.mlpHidden());
}

test "derived dims match the 3B config" {
    const cfg: Config = .{};
    try testing.expectEqual(@as(u32, 2560), cfg.innerDim()); // 20 * 128
    try testing.expectEqual(@as(u32, 7680), cfg.qkvDim());
    try testing.expectEqual(@as(u32, 15360), cfg.embDim()); // 6 * 2560
    try testing.expectEqual(@as(u32, 132), cfg.patchInDim()); // 33 * 1*2*2
    try testing.expectEqual(@as(u32, 64), cfg.patchOutDim()); // 16 * 1*2*2
}

test "qk_norm is per head_dim, not per model dim" {
    // norm_q/norm_k are built with dim=head_dim. A [2560] tensor here loads
    // 20x too much and the RMS divisor is computed over the wrong axis.
    const cfg: Config = .{};
    const m = try manifest(testing.allocator, cfg);
    defer freeManifest(testing.allocator, m);
    try testing.expectEqual([2]u32{ 128, 0 }, shapeOf(m, "blocks.0.attn.norm_q.vid.weight").?);
    try testing.expectEqual([2]u32{ 128, 0 }, shapeOf(m, "blocks.10.attn.norm_k.all.weight").?);
}

test "attn_norm and mlp_norm carry no parameters" {
    // elementwise_affine=False. A checkpoint with these tensors means the norm
    // was built affine and the ada shift/scale are being applied twice.
    const cfg: Config = .{};
    const m = try manifest(testing.allocator, cfg);
    defer freeManifest(testing.allocator, m);
    for (m) |t| {
        try testing.expect(std.mem.indexOf(u8, t.name, "attn_norm") == null);
        try testing.expect(std.mem.indexOf(u8, t.name, "mlp_norm") == null);
    }
}

test "proj_qkv has no bias but proj_out does" {
    // qk_bias=False on proj_qkv; proj_out is a plain nn.Linear (bias=True).
    const cfg: Config = .{};
    const m = try manifest(testing.allocator, cfg);
    defer freeManifest(testing.allocator, m);
    try testing.expect(!hasName(m, "blocks.0.attn.proj_qkv.vid.bias"));
    try testing.expect(hasName(m, "blocks.0.attn.proj_out.vid.bias"));
    // SwiGLU is entirely bias-free.
    try testing.expect(!hasName(m, "blocks.0.mlp.vid.proj_in.bias"));
    try testing.expect(!hasName(m, "blocks.0.mlp.vid.proj_out.bias"));
}

test "vid_out_ada has shift and scale but no gate" {
    // modes=["in"] only. An out_gate tensor means the head was built with the
    // block-style ada and something is modulating the output twice.
    const cfg: Config = .{};
    const m = try manifest(testing.allocator, cfg);
    defer freeManifest(testing.allocator, m);
    try testing.expect(hasName(m, "vid_out_ada.out_shift"));
    try testing.expect(hasName(m, "vid_out_ada.out_scale"));
    try testing.expect(!hasName(m, "vid_out_ada.out_gate"));
    // vid_out_norm IS affine, unlike the per-block norms.
    try testing.expect(hasName(m, "vid_out_norm.weight"));
}

test "txt_in collapses to Identity when the dims already match" {
    // The reference substitutes nn.Identity when txt_in_dim == txt_dim, so the
    // tensors vanish. Guards a future 7B/variant config rather than the 3B.
    var cfg: Config = .{};
    cfg.txt_in_dim = cfg.vid_dim;
    const m = try manifest(testing.allocator, cfg);
    defer freeManifest(testing.allocator, m);
    try testing.expect(!hasName(m, "txt_in.weight"));

    const m3b = try manifest(testing.allocator, .{});
    defer freeManifest(testing.allocator, m3b);
    try testing.expectEqual([2]u32{ 2560, 5120 }, shapeOf(m3b, "txt_in.weight").?);
}

test "manifest tensor count is exact and every name is unique" {
    // 10 split layers x 2 branches + 22 shared layers x 1 = 42 branch-instances.
    // Per instance: 1 qkv + 2 proj_out + 2 qk_norm + 3 mlp + 6 ada = 14.
    // Stem: 2 vid_in + 2 txt_in + 6 emb_in = 10. Head: 1+2+2 = 5.
    const cfg: Config = .{};
    const m = try manifest(testing.allocator, cfg);
    defer freeManifest(testing.allocator, m);

    const instances: usize = 10 * 2 + 22 * 1;
    try testing.expectEqual(@as(usize, 42), instances);
    // + one rope freqs buffer per LAYER (outside the branch loop).
    try testing.expectEqual(@as(usize, 10 + instances * 14 + cfg.num_layers + 5), m.len);
    // The real checkpoint has exactly this many tensors — see
    // tests/fixtures/seedvr2/dit_manifest.json, dumped from
    // seedvr2_ema_3b_fp16.safetensors.
    try testing.expectEqual(@as(usize, 635), m.len);

    var seen = std.StringHashMap(void).init(testing.allocator);
    defer seen.deinit();
    for (m) |t| {
        const gop = try seen.getOrPut(t.name);
        testing.expect(!gop.found_existing) catch |err| {
            std.debug.print("duplicate tensor name: {s}\n", .{t.name});
            return err;
        };
    }
}

test "every manifest shape is non-zero in its declared rank" {
    // Cheap structural guard: a rank-2 entry with a zero second dim is a
    // rank-1 entry that was mislabelled, and would allocate a zero-column
    // matrix that silently produces zeros.
    const m = try manifest(testing.allocator, .{});
    defer freeManifest(testing.allocator, m);
    for (m) |t| {
        try testing.expect(t.rank == 1 or t.rank == 2);
        try testing.expect(t.shape[0] > 0);
        if (t.rank == 2) try testing.expect(t.shape[1] > 0);
        if (t.rank == 1) try testing.expectEqual(@as(u32, 0), t.shape[1]);
    }
}
