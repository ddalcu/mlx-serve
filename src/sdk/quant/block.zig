//! A config's quantization blocks and the built-in formats' claims and validation, MLX-free (discovery and the
//! load preflight run them): which block a pack carries, which format serves it, the defaults it declares.

const std = @import("std");
const abi = @import("../../abi/quant.zig");
const common = @import("../../abi/common.zig");
const fp8_grid = @import("fp8_block128_gemv.zig");
const sushi_exl3 = @import("sushi_exl3");

const Str = abi.Str;
const Claim = abi.Claim;
const Verdict = abi.Verdict;
const Value = std.json.Value;

/// The config keys a quantization block sits under.
pub const keys = [_][]const u8{ "quantization", "quantization_config", "expert_quant" };

pub const Block = struct { key: []const u8, json: []const u8 };

/// The blocks of `config_json`, in `keys` order, each re-serialized into `a`; none when it does not parse.
pub fn blocksOf(a: std.mem.Allocator, config_json: []const u8) ![]Block {
    const root = parseObject(a, config_json) orelse return &.{};
    var out: std.ArrayList(Block) = .empty;
    for (keys) |k| {
        const v = root.get(k) orelse continue;
        if (v != .object) continue;
        try out.append(a, .{ .key = k, .json = try std.json.Stringify.valueAlloc(a, v, .{}) });
    }
    return out.items;
}

/// The (bits, group size, mode) a block declares model-wide; each weight still resolves from its own geometry.
pub const Declared = struct { bits: u32 = 0, group_size: u32 = 64, mode: []const u8 = "affine" };

/// MLX's affine kernels: an already-quantized pack outside these dies at Metal kernel load.
pub fn affineBits(bits: u32) bool {
    return switch (bits) {
        2, 3, 4, 5, 6, 8 => true,
        else => false,
    };
}

/// The MLX modes a block or a per-weight override may name.
pub fn mlxMode(mode: []const u8) bool {
    for ([_][]const u8{ "affine", "nvfp4", "mxfp4", "mxfp8" }) |m| if (std.mem.eql(u8, mode, m)) return true;
    return false;
}

/// What the block declares; a ModelOpt block is NVFP4 g16 (its MXFP8 linears resolve per weight).
pub fn declared(a: std.mem.Allocator, block_json: []const u8) Declared {
    const o = parseObject(a, block_json) orelse return .{};
    if (eqlField(o, "quant_method", "modelopt")) return .{ .bits = 4, .group_size = 16, .mode = "nvfp4" };
    var d: Declared = .{};
    if (uintField(o, "bits")) |b| d.bits = b;
    if (uintField(o, "group_size")) |g| d.group_size = g;
    if (o.get("mode")) |m| if (m == .string) {
        d.mode = m.string;
    };
    return d;
}

/// An NVIDIA ModelOpt block (`quant_method: modelopt`).
pub fn isModelOpt(block_json: []const u8) bool {
    return methodIs(.of(block_json), "modelopt");
}

/// The Hadamard block every listed module rotates by (Prism's `modules[].block`); null when there is no such list
/// or the modules disagree.
pub fn hadamardBlock(a: std.mem.Allocator, config_json: []const u8) ?u32 {
    const root = parseObject(a, config_json) orelse return null;
    const mods = root.get("modules") orelse return null;
    if (mods != .array) return null;
    var block: u32 = 0;
    for (mods.array.items) |m| {
        if (m != .object) return null;
        const b = uintField(m.object, "block") orelse return null;
        if (b == 0 or (block != 0 and b != block)) return null;
        block = b;
    }
    return if (block == 0) null else block;
}

fn hasModules(a: std.mem.Allocator, config_json: []const u8) bool {
    const root = parseObject(a, config_json) orelse return false;
    const mods = root.get("modules") orelse return false;
    return mods == .array;
}

// ── the built-in formats' claims and validation ──

pub const ClaimsFn = *const fn (Str, Str) callconv(.c) Claim;
pub const ValidateFn = *const fn (Str, Str, *const abi.PackDir) callconv(.c) Verdict;

pub const Builtin = struct { name: []const u8, claims: ClaimsFn, validate: ValidateFn };

/// The built-in formats, in the order `formats.zig` lists their tables.
pub const builtins = [_]Builtin{
    .{ .name = "affine", .claims = &claimsAffine, .validate = &validateMlx },
    .{ .name = "fp8", .claims = &claimsFp8, .validate = &validateFp8 },
    .{ .name = "mxfp8", .claims = &claimsMxfp8, .validate = &validateMlx },
    .{ .name = "nvfp4", .claims = &claimsNvfp4, .validate = &validateMlx },
    .{ .name = "hadamard", .claims = &claimsHadamard, .validate = &validateHadamard },
    .{ .name = "exl3", .claims = &claimsExl3, .validate = &validateExl3 },
    .{ .name = "gguf", .claims = &claimsGguf, .validate = &validateGguf },
};

/// The built-in format serving `block_json`: the highest claim wins, two at that rank are ambiguous.
pub fn claimant(block_json: []const u8, config_json: []const u8) error{Ambiguous}!?usize {
    var got: [builtins.len]Claim = undefined;
    for (builtins, &got) |b, *c| c.* = b.claims(.of(block_json), .of(config_json));
    return common.pick(&got);
}

/// A scratch arena for one call: the C-ABI members take no allocator.
const Scratch = struct {
    arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator),
    fn a(self: *Scratch) std.mem.Allocator {
        return self.arena.allocator();
    }
};

/// An MLX-style block (`bits`, `group_size`, `mode`; no `quant_method`) whose mode is one of `modes`. Without
/// `bits` the pack loads dense.
fn mlxBlockOf(block_json: Str, comptime modes: []const []const u8) bool {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const o = parseObject(sc.a(), block_json.slice()) orelse return false;
    if (o.get("quant_method") != null or uintField(o, "bits") == null) return false;
    const mode: []const u8 = if (o.get("mode")) |m| (if (m == .string) m.string else return false) else "affine";
    for (modes) |m| if (std.mem.eql(u8, mode, m)) return true;
    return false;
}

fn methodIs(block_json: Str, method: []const u8) bool {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const o = parseObject(sc.a(), block_json.slice()) orelse return false;
    return eqlField(o, "quant_method", method);
}

fn claimsAffine(block_json: Str, _: Str) callconv(.c) Claim {
    return if (mlxBlockOf(block_json, &.{"affine"})) .by_type else .no;
}

fn claimsMxfp8(block_json: Str, _: Str) callconv(.c) Claim {
    return if (mlxBlockOf(block_json, &.{"mxfp8"})) .by_type else .no;
}

/// The e2m1 nibble modes (NVFP4 g16, MXFP4 g32) and NVIDIA ModelOpt packs.
fn claimsNvfp4(block_json: Str, _: Str) callconv(.c) Claim {
    return if (mlxBlockOf(block_json, &.{ "nvfp4", "mxfp4" }) or methodIs(block_json, "modelopt")) .by_type else .no;
}

/// Row-scaled e4m3 (Sushi's MiMo trunk, `quant_method: fp8`).
fn claimsFp8(block_json: Str, _: Str) callconv(.c) Claim {
    return if (methodIs(block_json, "fp8")) .by_type else .no;
}

/// An affine block beside Prism's `modules` list: the rotations outrank the block's own mode.
fn claimsHadamard(block_json: Str, config_json: Str) callconv(.c) Claim {
    if (claimsAffine(block_json, config_json) == .no) return .no;
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    return if (hasModules(sc.a(), config_json.slice())) .by_marker else .no;
}

fn validateMlx(block_json: Str, _: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const o = parseObject(sc.a(), block_json.slice()) orelse return Verdict.refuse(.failed, "quantization block does not parse");
    if (eqlField(o, "quant_method", "modelopt")) return .ok;
    const d = declared(sc.a(), block_json.slice());
    if (!mlxMode(d.mode)) return Verdict.refuse(.unsupported, "quantization mode other than affine, nvfp4, mxfp4, mxfp8");
    var it = o.iterator();
    while (it.next()) |kv| {
        if (kv.value_ptr.* != .object) continue;
        const m = kv.value_ptr.object.get("mode") orelse continue;
        if (m == .string and !mlxMode(m.string) and !fp8_grid.isEntry(kv.value_ptr.object)) return Verdict.refuse(.unsupported, "a per-weight quantization mode other than affine, nvfp4, mxfp4, mxfp8, fp8_block128");
    }
    if (std.mem.eql(u8, d.mode, "affine") and d.bits != 0 and !affineBits(d.bits))
        return Verdict.refuse(.unsupported, "affine quantization outside 2, 3, 4, 5, 6, 8 bits");
    return .ok;
}

/// Sushi's adaptation reads e4m3 in 128x128 tiles.
fn validateFp8(block_json: Str, _: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const o = parseObject(sc.a(), block_json.slice()) orelse return Verdict.refuse(.failed, "quantization block does not parse");
    const ok = if (o.get("weight_block_size")) |v| v == .array and v.array.items.len == 2 and
        isUint(v.array.items[0], 128) and isUint(v.array.items[1], 128) else false;
    return if (ok) .ok else Verdict.refuse(.unsupported, "fp8: weight_block_size other than [128, 128]");
}

fn validateHadamard(block_json: Str, config_json: Str, dir: *const abi.PackDir) callconv(.c) Verdict {
    const v = validateMlx(block_json, config_json, dir);
    if (!v.isOk()) return v;
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    return if (hadamardBlock(sc.a(), config_json.slice()) != null) .ok else Verdict.refuse(.unsupported, "hadamard: modules without one shared rotation block");
}

/// Sushi's `expert_quant`: EXL3 (MCG, MUL1) and its STQ codebook family (a named codebook table).
fn claimsExl3(block_json: Str, _: Str) callconv(.c) Claim {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const o = parseObject(sc.a(), block_json.slice()) orelse return .no;
    return if (eqlField(o, "format", "exl3") or eqlField(o, "format", "stq")) .by_type else .no;
}

/// Sushi's own parse: a codebook, window or table this build does not decode, and a top-k past the reduce bank.
fn validateExl3(block_json: Str, config_json: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    var sc: Scratch = .{};
    defer sc.arena.deinit();
    const root = expertQuantRoot(sc.a(), block_json.slice()) orelse return Verdict.refuse(.failed, "expert_quant block does not parse");
    _ = sushi_exl3.parseExpertQuant(root) catch |e| return Verdict.named(.unsupported, @errorName(e));
    const cfg = parseObject(sc.a(), config_json.slice()) orelse return .ok;
    const text = if (cfg.get("text_config")) |t| (if (t == .object) t.object else cfg) else cfg;
    const topk = uintField(cfg, "num_experts_per_tok") orelse uintField(text, "num_experts_per_tok") orelse return .ok;
    sushi_exl3.admitTopK(topk) catch |e| return Verdict.named(.unsupported, @errorName(e));
    return .ok;
}

/// `{"expert_quant": block}`: the root Sushi's parse reads.
pub fn expertQuantRoot(a: std.mem.Allocator, block_json: []const u8) ?std.json.ObjectMap {
    const wrapped = std.fmt.allocPrint(a, "{{\"expert_quant\":{s}}}", .{block_json}) catch return null;
    return parseObject(a, wrapped);
}

/// A GGUF file's rebuilt config (`mode: gguf`); its tensor types were vetted when the file was found servable.
fn claimsGguf(block_json: Str, _: Str) callconv(.c) Claim {
    return if (mlxBlockOf(block_json, &.{"gguf"})) .by_type else .no;
}

fn validateGguf(_: Str, _: Str, _: *const abi.PackDir) callconv(.c) Verdict {
    return .ok;
}

// ── JSON readers ──

fn parseObject(a: std.mem.Allocator, json: []const u8) ?std.json.ObjectMap {
    const v = std.json.parseFromSliceLeaky(Value, a, json, .{}) catch return null;
    return if (v == .object) v.object else null;
}

fn uintField(o: std.json.ObjectMap, key: []const u8) ?u32 {
    const v = o.get(key) orelse return null;
    if (v != .integer) return null;
    return std.math.cast(u32, v.integer);
}

fn eqlField(o: std.json.ObjectMap, key: []const u8, want: []const u8) bool {
    const v = o.get(key) orelse return false;
    return v == .string and std.mem.eql(u8, v.string, want);
}

fn isUint(v: Value, want: i64) bool {
    return v == .integer and v.integer == want;
}
