//! The example's claims and parse: MLX-free (discovery runs them with no device). A llama-shaped config.json in
//! mlx-lm's names; every field is bounds-checked before use (config JSON is untrusted).

const std = @import("std");
const kit = @import("mlx_serve");
const abi = kit.abi.plugin;

pub const model_types = [_][]const u8{"_example"};

pub const claims = kit.sdk.claimsByType(&model_types);

/// What the forward and the bill read.
pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    intermediate_size: u32,
    num_hidden_layers: u32,
    num_attention_heads: u32,
    num_key_value_heads: u32,
    head_dim: u32,
    rms_norm_eps: f32,
    rope_theta: f32,
};

const Raw = struct {
    vocab_size: u32 = 0,
    hidden_size: u32 = 0,
    intermediate_size: u32 = 0,
    num_hidden_layers: u32 = 0,
    num_attention_heads: u32 = 0,
    num_key_value_heads: ?u32 = null,
    head_dim: ?u32 = null,
    rms_norm_eps: f32 = 1e-5,
    rope_theta: f32 = 10000,
    max_position_embeddings: u64 = 4096,
    eos_token_id: ?u32 = null,
};

const gpa = std.heap.c_allocator;

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return abi.Verdict.refuse(.unsupported, "_example: no layer split");
    const raw = std.json.parseFromSlice(Raw, gpa, config_json.slice(), .{ .ignore_unknown_fields = true }) catch
        return abi.Verdict.refuse(.failed, "_example: config.json does not parse");
    defer raw.deinit();
    const r = raw.value;
    const kv = r.num_key_value_heads orelse r.num_attention_heads;
    const hd = r.head_dim orelse if (r.num_attention_heads > 0) r.hidden_size / r.num_attention_heads else 0;
    if (r.vocab_size == 0 or r.vocab_size > 1 << 20 or r.hidden_size == 0 or r.hidden_size > 1 << 14 or
        r.intermediate_size == 0 or r.intermediate_size > 1 << 16 or r.num_hidden_layers == 0 or r.num_hidden_layers > 256 or
        r.num_attention_heads == 0 or r.num_attention_heads > 256 or kv == 0 or r.num_attention_heads % kv != 0 or
        hd == 0 or hd > 512 or hd % 2 != 0 or !(r.rms_norm_eps > 0) or !(r.rope_theta > 0))
        return abi.Verdict.refuse(.failed, "_example: config.json dimensions out of range");

    var p: abi.Parsed = .{ .caps = .{ .batches_decode = true } };
    p.shell = .{
        .vocab_size = r.vocab_size,
        .hidden_size = r.hidden_size,
        .intermediate_size = r.intermediate_size,
        .num_hidden_layers = r.num_hidden_layers,
        .num_attention_heads = r.num_attention_heads,
        .num_key_value_heads = kv,
        .head_dim = hd,
        .max_position_embeddings = r.max_position_embeddings,
        .context_cap = r.max_position_embeddings,
        .act_dtype = .f32,
        .model_type = abi.Str.of(model_types[0]),
    };
    if (r.eos_token_id) |e| {
        p.shell.eos[0] = e;
        p.shell.eos_count = 1;
    }
    // Discovery's stub parse stops at config.json; the load's parse reads the sidecars too.
    if (!stub) kit.generation.apply(dir, &p.shell);
    const cfg = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "_example: out of memory");
    cfg.* = .{
        .vocab_size = r.vocab_size,
        .hidden_size = r.hidden_size,
        .intermediate_size = r.intermediate_size,
        .num_hidden_layers = r.num_hidden_layers,
        .num_attention_heads = r.num_attention_heads,
        .num_key_value_heads = kv,
        .head_dim = hd,
        .rms_norm_eps = r.rms_norm_eps,
        .rope_theta = r.rope_theta,
    };
    p.private = cfg;
    out.* = p;
    return .ok;
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |c| gpa.destroy(@as(*Config, @ptrCast(@alignCast(c))));
    p.private = null;
}

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

// ── tests ──

const testing = std.testing;
const fixture = kit.sdk.fixture;

fn parseFiles(files: []const fixture.File, stub: bool, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = files };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(files[0].bytes), .{}, null, stub, out);
}

test "_example config: head_dim and kv heads default as llama's; the full parse reads generation_config.json" {
    const files = [_]fixture.File{
        .{ .name = "config.json", .bytes = "{\"model_type\":\"_example\",\"vocab_size\":32,\"hidden_size\":16,\"intermediate_size\":32,\"num_hidden_layers\":2,\"num_attention_heads\":2}" },
        .{ .name = "generation_config.json", .bytes = "{\"temperature\":0.6}" },
    };
    var stub: abi.Parsed = .{};
    try testing.expect(parseFiles(&files, true, &stub).isOk());
    defer freeParsed(&stub);
    try testing.expect(std.math.isNan(stub.shell.gen.temperature));
    try testing.expectEqual(@as(u32, 8), of(&stub).head_dim);
    try testing.expectEqual(@as(u32, 2), of(&stub).num_key_value_heads);
    var full: abi.Parsed = .{};
    try testing.expect(parseFiles(&files, false, &full).isOk());
    defer freeParsed(&full);
    try testing.expectEqual(@as(f32, 0.6), full.shell.gen.temperature);
}

test "_example config: bad geometry and broken JSON are refused by name, leaving nothing to free" {
    inline for (.{ "{\"vocab_size\":32,\"hidden_size\":16,\"intermediate_size\":32,\"num_hidden_layers\":2,\"num_attention_heads\":3,\"num_key_value_heads\":2}", "{\"vocab_size\":0}", "{", "[]" }) |json| {
        var p: abi.Parsed = .{};
        const v = parseFiles(&.{.{ .name = "config.json", .bytes = json }}, true, &p);
        try testing.expect(!v.isOk() and v.why.len > 0 and p.private == null);
    }
}
