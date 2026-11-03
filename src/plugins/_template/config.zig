//! The template model's claims and parse. MLX-free (discovery runs them with no device): this file and everything it
//! imports may not reach `src/mlx/`, `src/engines/`, `src/host/` or `src/kernels/` (the build checks the closure).

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");

pub const model_types = [_][]const u8{"_template"};

pub const claims = sdk.claimsByType(&model_types);

/// The template's own fields, read by its bill and its load.
pub const Config = struct {
    vocab_size: u32,
    hidden_size: u32,
    num_hidden_layers: u32,
    rms_norm_eps: f32,
};

/// config.json as written, every field bounds-checked before use (config JSON is untrusted).
const Raw = struct {
    vocab_size: u32 = 0,
    hidden_size: u32 = 0,
    num_hidden_layers: u32 = 1,
    rms_norm_eps: f32 = 1e-6,
    max_position_embeddings: u64 = 4096,
    eos_token_id: ?u32 = null,
};

const GenerationConfig = struct {
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    top_k: ?i32 = null,
};

const gpa = std.heap.c_allocator;

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    _ = span;
    const raw = std.json.parseFromSlice(Raw, gpa, config_json.slice(), .{ .ignore_unknown_fields = true }) catch
        return abi.Verdict.refuse(.failed, "_template: config.json does not parse");
    defer raw.deinit();
    const r = raw.value;
    if (r.vocab_size == 0 or r.vocab_size > 1 << 20 or r.hidden_size == 0 or r.hidden_size > 1 << 14 or
        r.num_hidden_layers == 0 or r.num_hidden_layers > 1024 or !(r.rms_norm_eps > 0))
        return abi.Verdict.refuse(.failed, "_template: config.json dimensions out of range");

    var p: abi.Parsed = .{ .caps = .{ .batches_decode = true } };
    p.shell = .{
        .vocab_size = r.vocab_size,
        .hidden_size = r.hidden_size,
        .num_hidden_layers = r.num_hidden_layers,
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
    if (!stub) {
        var buf: [4096]u8 = undefined;
        var why: abi.Verdict = .{};
        if (dir.readFile("generation_config.json", &buf, &why)) |bytes| {
            const g = std.json.parseFromSlice(GenerationConfig, gpa, bytes, .{ .ignore_unknown_fields = true }) catch
                return abi.Verdict.refuse(.failed, "_template: generation_config.json does not parse");
            defer g.deinit();
            if (g.value.temperature) |t| p.shell.gen.temperature = t;
            if (g.value.top_p) |t| p.shell.gen.top_p = t;
            if (g.value.top_k) |k| p.shell.gen.top_k = k;
        }
    }
    const cfg = gpa.create(Config) catch return abi.Verdict.refuse(.failed, "_template: out of memory");
    cfg.* = .{ .vocab_size = r.vocab_size, .hidden_size = r.hidden_size, .num_hidden_layers = r.num_hidden_layers, .rms_norm_eps = r.rms_norm_eps };
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
const fixture = sdk.fixture;

fn parseFiles(files: []const fixture.File, stub: bool, out: *abi.Parsed) abi.Verdict {
    const mem: fixture.MemDir = .{ .files = files };
    const dir = mem.packDir();
    return parse(&dir, abi.Str.of(files[0].bytes), .{}, null, stub, out);
}

test "_template config: the stub parse reads config.json only; the full parse adds generation_config.json" {
    const files = [_]fixture.File{
        .{ .name = "config.json", .bytes = "{\"model_type\":\"_template\",\"vocab_size\":16,\"hidden_size\":8,\"eos_token_id\":3}" },
        .{ .name = "generation_config.json", .bytes = "{\"temperature\":0.7,\"top_k\":20}" },
    };
    var stub: abi.Parsed = .{};
    try testing.expect(parseFiles(&files, true, &stub).isOk());
    defer freeParsed(&stub);
    try testing.expect(std.math.isNan(stub.shell.gen.temperature));
    try testing.expectEqual(@as(u32, 3), stub.shell.eos[0]);
    var full: abi.Parsed = .{};
    try testing.expect(parseFiles(&files, false, &full).isOk());
    defer freeParsed(&full);
    try testing.expectEqual(@as(f32, 0.7), full.shell.gen.temperature);
    try testing.expectEqual(@as(i32, 20), full.shell.gen.top_k);
    try testing.expectEqual(@as(u32, 8), of(&full).hidden_size);
}

test "_template config: out-of-range dimensions and broken JSON are refused by name, leaving nothing to free" {
    inline for (.{ "{\"vocab_size\":0,\"hidden_size\":8}", "{\"vocab_size\":16,\"hidden_size\":99999999}", "{\"vocab_size\":16", "[]" }) |json| {
        var p: abi.Parsed = .{};
        const v = parseFiles(&.{.{ .name = "config.json", .bytes = json }}, true, &p);
        try testing.expect(!v.isOk() and v.why.len > 0 and p.private == null);
    }
}
