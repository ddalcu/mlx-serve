//! Tencent Hunyuan 4-flash (`hy_v4_flash`): claim, config parse, and bill. MLX-free.

const std = @import("std");
const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const generation = @import("../../sdk/config/generation.zig");
const json = @import("../../sdk/config/json.zig");

pub const plugin = sdk.Plugin{
    .name = "hy_v4_flash",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{"tencent/Hy4-flash"},
};

pub const model_types = [_][]const u8{"hy_v4_flash"};
pub const claims = sdk.claimsByType(&model_types);

/// `<｜hy_end:opensource｜>`: the family's terminator, merged into whatever the config declares.
pub const hy_eos: u32 = 120025;

pub const Hy4Spec = struct {
    file: []const u8,
    bits: u32,
    group_size: u32,
    head_dim: u32,
    rows: u64,
    layers: []const u32 = &.{},
    table_rows: []const u64 = &.{},

    pub fn rowOffset(self: *const Hy4Spec, k: usize) u64 {
        var off: u64 = 0;
        for (self.table_rows[0..@min(k, self.table_rows.len)]) |n| off += n;
        return off;
    }
};

pub const Config = struct {
    hidden: usize = 3072,
    layers: usize = 42,
    vocab: usize = 120832,
    hc_mult: usize = 4,
    hc_magnitude: f32 = 2.0,
    hc_eps: f32 = 1e-6,
    eps: f32 = 1e-5,
    q_lora: usize = 1024,
    kv_lora: usize = 512,
    qk_nope: usize = 192,
    rope_dim: usize = 64,
    v_dim: usize = 256,
    indexed_heads: usize = 32,
    swa_heads: usize = 64,
    swa_qk_nope: usize = 192,
    swa_v_dim: usize = 256,
    idx_heads: usize = 16,
    idx_head_dim: usize = 128,
    group_topk: usize = 512,
    candidate_topk: usize = 2048,
    ratio: usize = 4,
    sliding: usize = 512,
    engram_fp8: bool = false,
    engram_head_dim: usize = 96,
    head_dim: usize = 96,
    latent: usize = 1536,
    moe_inter: usize = 1536,
    top_k: usize = 16,
    route_scale: f32 = 3.998,
    swiglu_limit: f32 = 10.0,
    quant_group_size: u32 = 64,
    engram_table: ?Hy4Spec = null,
    model_dir: []const u8 = "",
    lookup_size: usize = 120818,
    pad: usize = 120002,
    max_ngram: usize = 3,
    indexed_theta: f64 = 10_000_000,
    swa_theta: f64 = 10_000,
    mrope: [3]usize = .{ 12, 10, 10 },

    indexed: []const bool = &.{},
    sparse: []const bool = &.{},
    engram_layers: []const usize = &.{},
    engram_sizes: []const []const usize = &.{},
    hash_mult: []const i64 = &.{},

    pub fn isIndexed(self: Config, layer: usize) bool {
        if (layer < self.indexed.len) return self.indexed[layer];
        return (layer % 2 == 0);
    }

    pub fn isDense(self: Config, layer: usize) bool {
        if (layer < self.sparse.len) return !self.sparse[layer];
        return (layer == 0);
    }

    pub fn pad_id(self: Config) usize {
        return self.pad;
    }

    pub fn deinit(self: *Config, a: std.mem.Allocator) void {
        if (self.indexed.len > 0) a.free(self.indexed);
        if (self.sparse.len > 0) a.free(self.sparse);
        if (self.engram_layers.len > 0) a.free(self.engram_layers);
        for (self.engram_sizes) |s| a.free(s);
        if (self.engram_sizes.len > 0) a.free(self.engram_sizes);
        if (self.hash_mult.len > 0) a.free(self.hash_mult);
        if (self.engram_table) |sp| {
            a.free(sp.file);
            if (sp.layers.len > 0) a.free(sp.layers);
            if (sp.table_rows.len > 0) a.free(sp.table_rows);
        }
        if (self.model_dir.len > 0) a.free(self.model_dir);
    }
};

const gpa = std.heap.c_allocator;

fn unsupported(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.unsupported, "hy_v4_flash: " ++ why);
}

fn invalid(comptime why: []const u8) abi.Verdict {
    return abi.Verdict.refuse(.failed, "hy_v4_flash: " ++ why);
}

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    if (span != null) return unsupported("a layer span (the cluster split)");
    const json_slice = config_json.slice();

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const root_val = std.json.parseFromSlice(std.json.Value, arena.allocator(), json_slice, .{}) catch return invalid("config.json is not valid JSON");
    if (root_val.value != .object) return invalid("config.json is not a JSON object");
    const root_obj = root_val.value.object;
    const type_str = if (root_obj.get("model_type")) |mt| (if (mt == .string) mt.string else return unsupported("invalid model_type")) else return unsupported("no model_type");
    if (!std.mem.eql(u8, type_str, "hy_v4_flash")) return unsupported("another model_type");

    var cfg = (if (root_obj.get("text_config") != null or root_obj.get("num_hidden_layers") != null)
        parseRealConfig(gpa, json_slice)
    else
        parseConfigJson(gpa, json_slice)) catch return invalid("failed to parse hy_v4_flash config");
    errdefer cfg.deinit(gpa);

    var p: abi.Parsed = .{
        .caps = .{
            .owns_decode_state = true,
            .prefix_cache = false,
            .moe = true,
        },
    };
    const s = &p.shell;
    s.* = .{
        .vocab_size = @intCast(cfg.vocab),
        .hidden_size = @intCast(cfg.hidden),
        .intermediate_size = @intCast(cfg.moe_inter),
        .num_hidden_layers = @intCast(cfg.layers),
        .num_attention_heads = @intCast(cfg.indexed_heads),
        .num_key_value_heads = @intCast(cfg.indexed_heads),
        .head_dim = @intCast(cfg.head_dim),
        .num_experts_per_tok = @intCast(cfg.top_k),
        .moe_intermediate_size = @intCast(cfg.moe_inter),
        .max_position_embeddings = 32768,
        .context_cap = 32768,
        .act_dtype = .bf16,
        .model_type = abi.Str.of("hy_v4_flash"),
    };

    if (root_obj.get("bos_token_id")) |v| if (v == .integer and v.integer >= 0) {
        s.bos = @intCast(v.integer);
    };
    json.eos(root_val.value.object, s) catch return invalid("eos_token_id");
    generation.mergeEos(s, hy_eos);
    if (!stub) generation.apply(dir, s);

    const cfg_out = gpa.create(Config) catch return invalid("out of memory");
    cfg_out.* = cfg;
    p.private = cfg_out;
    out.* = p;
    return .ok;
}

pub fn freeParsed(p: *abi.Parsed) callconv(.c) void {
    if (p.private) |ptr| {
        const c: *Config = @ptrCast(@alignCast(ptr));
        c.deinit(gpa);
        gpa.destroy(c);
    }
    p.private = null;
}

pub fn of(p: *const abi.Parsed) *const Config {
    return @ptrCast(@alignCast(p.private.?));
}

pub fn bill(_: *const abi.Parsed, _: *const abi.Shape, out: *abi.Bill) callconv(.c) void {
    out.* = .{};
}

pub fn parseHy4Spec(a: std.mem.Allocator, cfg_json: []const u8) !?Hy4Spec {
    const parsed = std.json.parseFromSlice(std.json.Value, a, cfg_json, .{}) catch return error.EngramTableHeader;
    defer parsed.deinit();
    if (parsed.value != .object) return null;
    const v = parsed.value.object.get("engram_table") orelse return null;
    if (v != .object) return error.EngramTableHeader;
    const o = v.object;
    const file = o.get("file") orelse return error.EngramTableMissing;
    if (file != .string or file.string.len == 0 or std.mem.indexOfAny(u8, file.string, "/\\") != null or
        std.mem.indexOf(u8, file.string, "..") != null) return error.EngramTableMissing;
    const num = struct {
        fn readU32(obj: std.json.ObjectMap, key: []const u8) !u32 {
            const x = obj.get(key) orelse return error.EngramTableHeader;
            if (x != .integer or x.integer <= 0 or x.integer > std.math.maxInt(u32)) return error.EngramTableHeader;
            return @intCast(x.integer);
        }
        fn readU64(obj: std.json.ObjectMap, key: []const u8) !u64 {
            const x = obj.get(key) orelse return error.EngramTableHeader;
            if (x != .integer or x.integer <= 0) return error.EngramTableHeader;
            return @intCast(x.integer);
        }
    };
    var layers: []u32 = &.{};
    if (o.get("layers")) |lv| {
        if (lv != .array) return error.EngramTableHeader;
        layers = try a.alloc(u32, lv.array.items.len);
        for (lv.array.items, 0..) |item, i| {
            if (item != .integer or item.integer < 0) return error.EngramTableHeader;
            layers[i] = @intCast(item.integer);
        }
    }
    var table_rows: []u64 = &.{};
    if (o.get("table_rows")) |rv| {
        if (rv != .array) return error.EngramTableHeader;
        table_rows = try a.alloc(u64, rv.array.items.len);
        for (rv.array.items, 0..) |item, i| {
            if (item != .integer or item.integer <= 0) return error.EngramTableHeader;
            table_rows[i] = @intCast(item.integer);
        }
    }
    return Hy4Spec{
        .file = try a.dupe(u8, file.string),
        .bits = try num.readU32(o, "bits"),
        .group_size = try num.readU32(o, "group_size"),
        .head_dim = try num.readU32(o, "head_dim"),
        .rows = try num.readU64(o, "rows"),
        .layers = layers,
        .table_rows = table_rows,
    };
}

pub fn parseConfigJson(a: std.mem.Allocator, json_str: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json_str, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    var cfg = Config{};
    cfg.hidden = @intCast(try intField(root, "hidden"));
    cfg.layers = @intCast(try intField(root, "layers"));
    cfg.vocab = @intCast(try intField(root, "vocab"));
    cfg.hc_mult = @intCast(try intField(root, "hc_mult"));
    cfg.hc_magnitude = @floatCast(try floatField(root, "hc_magnitude"));
    cfg.eps = @floatCast(try floatField(root, "eps"));
    cfg.q_lora = @intCast(try intField(root, "q_lora"));
    cfg.kv_lora = @intCast(try intField(root, "kv_lora"));
    cfg.qk_nope = @intCast(try intField(root, "qk_nope"));
    cfg.rope_dim = @intCast(try intField(root, "rope_dim"));
    cfg.v_dim = @intCast(try intField(root, "v_dim"));
    cfg.swa_heads = @intCast(try intField(root, "swa_heads"));
    cfg.indexed_heads = @intCast(try intField(root, "indexed_heads"));
    cfg.swa_v_dim = @intCast(try intField(root, "swa_v_dim"));
    cfg.swa_qk_nope = @intCast(try intField(root, "swa_qk_nope"));
    cfg.idx_heads = @intCast(try intField(root, "idx_heads"));
    cfg.idx_head_dim = @intCast(try intField(root, "idx_head_dim"));
    cfg.group_topk = @intCast(try intField(root, "group_topk"));
    cfg.candidate_topk = @intCast(try intField(root, "candidate_topk"));
    cfg.ratio = @intCast(try intField(root, "ratio"));
    cfg.sliding = @intCast(try intField(root, "sliding"));
    cfg.head_dim = @intCast(try intField(root, "head_dim"));
    if (root.get("engram_fp8")) |v| if (v == .bool) { cfg.engram_fp8 = v.bool; };
    cfg.latent = @intCast(try intField(root, "latent"));
    cfg.moe_inter = @intCast(try intField(root, "moe_inter"));
    cfg.top_k = @intCast(try intField(root, "top_k"));
    cfg.route_scale = @floatCast(try floatField(root, "route_scale"));
    cfg.swiglu_limit = @floatCast(try floatField(root, "swiglu_limit"));
    cfg.lookup_size = @intCast(try intField(root, "lookup_size"));
    cfg.pad = @intCast(try intField(root, "pad"));
    cfg.max_ngram = @intCast(try intField(root, "max_ngram"));
    cfg.indexed_theta = try floatField(root, "indexed_theta");
    cfg.swa_theta = try floatField(root, "swa_theta");
    if (root.get("mrope")) |mr| {
        for (0..3) |i| cfg.mrope[i] = @intCast(mr.array.items[i].integer);
    }

    const ix = try a.alloc(bool, cfg.layers);
    const sp = try a.alloc(bool, cfg.layers);
    const lts = root.get("layer_types").?.array;
    const mlt = root.get("mlp_types").?.array;
    for (0..cfg.layers) |i| {
        ix[i] = std.mem.eql(u8, lts.items[i].string, "indexed_attention");
        sp[i] = std.mem.eql(u8, mlt.items[i].string, "sparse");
    }
    cfg.indexed = ix;
    cfg.sparse = sp;

    const els = root.get("engram_layers").?.array;
    const layers = try a.alloc(usize, els.items.len);
    for (els.items, 0..) |v, i| layers[i] = @intCast(v.integer);
    cfg.engram_layers = layers;

    const es = root.get("engram_sizes").?.array;
    var flat = try a.alloc(usize, 0);
    for (es.items) |order| {
        const next = try a.alloc(usize, flat.len + order.array.items.len);
        @memcpy(next[0..flat.len], flat);
        for (order.array.items, 0..) |v, h| next[flat.len + h] = @intCast(v.integer);
        a.free(flat);
        flat = next;
    }
    const one_layer = try a.alloc([]const usize, 1);
    one_layer[0] = flat;
    cfg.engram_sizes = one_layer;
    if (root.get("hash_mult")) |hm| {
        const mult = try a.alloc(i64, hm.array.items.len);
        for (hm.array.items, 0..) |v, i| mult[i] = v.integer;
        cfg.hash_mult = mult;
    }
    cfg.engram_table = try parseHy4Spec(a, json_str);
    return cfg;
}

pub fn parseRealConfig(a: std.mem.Allocator, json_str: []const u8) !Config {
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json_str, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    const o = if (root.get("text_config")) |tc| tc.object else root;
    var cfg = Config{};
    cfg.hidden = @intCast(try hfInt(o, "hidden_size", 3072));
    cfg.layers = @intCast(try hfInt(o, "num_hidden_layers", 42));
    cfg.vocab = @intCast(try hfInt(o, "vocab_size", 120832));
    cfg.eps = @floatCast(try hfFloat(o, "rms_norm_eps", 1e-5));
    cfg.hc_eps = @floatCast(try hfFloat(o, "hc_eps", 1e-6));
    cfg.hc_magnitude = @floatCast(try hfFloat(o, "hc_magnitude", 2.0));
    cfg.hc_mult = @intCast(try hfInt(o, "hc_mult", 4));
    cfg.q_lora = @intCast(try hfInt(o, "q_lora_rank", 1024));
    cfg.kv_lora = @intCast(try hfInt(o, "kv_lora_rank", 512));
    cfg.qk_nope = @intCast(try hfInt(o, "qk_nope_head_dim", 192));
    cfg.rope_dim = @intCast(try hfInt(o, "qk_rope_head_dim", 64));
    cfg.v_dim = @intCast(try hfInt(o, "v_head_dim", 256));
    cfg.indexed_heads = @intCast(try hfInt(o, "num_attention_heads", 32));
    cfg.swa_heads = @intCast(try hfInt(o, "swa_num_attention_heads", 64));
    cfg.swa_qk_nope = @intCast(try hfInt(o, "swa_qk_nope_head_dim", 192));
    cfg.swa_v_dim = @intCast(try hfInt(o, "swa_v_head_dim", 256));
    cfg.idx_heads = @intCast(try hfInt(o, "compressed_indexer_num_attention_heads", 16));
    cfg.idx_head_dim = @intCast(try hfInt(o, "compressed_indexer_head_dim", 128));
    cfg.group_topk = @intCast(try hfInt(o, "compressed_indexer_group_topk", 512));
    cfg.candidate_topk = cfg.group_topk * @as(usize, @intCast(try hfInt(o, "compressed_indexer_compress_ratio", 4)));
    cfg.ratio = @intCast(try hfInt(o, "compressed_indexer_compress_ratio", 4));
    cfg.sliding = @intCast(try hfInt(o, "sliding_window", 512));
    cfg.max_ngram = @intCast(try hfInt(o, "engram_max_ngram_size", 3));
    cfg.head_dim = @intCast(@divTrunc(try hfInt(o, "engram_n_embed_per_ngram", 768), try hfInt(o, "engram_n_head_per_ngram", 8)));
    cfg.lookup_size = @intCast(try hfInt(o, "engram_lookup_table_size", 120818));
    cfg.pad = @intCast(try hfInt(o, "engram_pad_id", 120002));
    if (o.get("engram_fp8_embedding")) |v| if (v == .bool) { cfg.engram_fp8 = v.bool; };
    cfg.latent = @intCast(try hfInt(o, "routed_expert_hidden_size", 1536));
    cfg.moe_inter = @intCast(try hfInt(o, "moe_intermediate_size", 1536));
    cfg.top_k = @intCast(try hfInt(o, "num_experts_per_tok", 16));
    cfg.route_scale = @floatCast(try hfFloat(o, "routed_scaling_factor", 3.998));
    cfg.swiglu_limit = @floatCast(try hfFloat(o, "swiglu_limit", 10.0));

    if (o.get("rope_parameters")) |rp| {
        if (rp == .object) {
            const rpi = rp.object.get("deepseek_sparse_attention") orelse rp.object.get("indexed_attention");
            const rps = rp.object.get("sliding_attention");
            if (rpi != null and rpi.? == .object) {
                cfg.indexed_theta = try hfFloat(rpi.?.object, "rope_theta", 1e7);
                if (rpi.?.object.get("mrope_section")) |ms| if (ms == .array and ms.array.items.len >= 3) {
                    for (0..3) |i| cfg.mrope[i] = @intCast(ms.array.items[i].integer);
                };
            }
            if (rps != null and rps.? == .object) {
                cfg.swa_theta = try hfFloat(rps.?.object, "rope_theta", 1e4);
            }
        }
    }

    if (o.get("layer_types")) |lts| {
        const mlt = o.get("mlp_layer_types");
        const ix = try a.alloc(bool, cfg.layers);
        const sp = try a.alloc(bool, cfg.layers);
        for (0..cfg.layers) |i| {
            if (i < lts.array.items.len) {
                const s = lts.array.items[i].string;
                ix[i] = std.mem.eql(u8, s, "deepseek_sparse_attention") or std.mem.eql(u8, s, "indexed_attention");
            } else ix[i] = true;
            if (mlt != null and i < mlt.?.array.items.len) {
                sp[i] = std.mem.eql(u8, mlt.?.array.items[i].string, "sparse");
            } else sp[i] = true;
        }
        cfg.indexed = ix;
        cfg.sparse = sp;
    }

    if (o.get("engram_layer_ids")) |el| {
        const ids = try a.alloc(usize, el.array.items.len);
        for (el.array.items, 0..) |v, i| ids[i] = @intCast(v.integer - 1);
        cfg.engram_layers = ids;
        if (o.get("engram_layer_head_vocab_sizes")) |hv| {
            const sizes = try a.alloc([]const usize, el.array.items.len);
            for (el.array.items, 0..) |v, k| {
                var kbuf: [32]u8 = undefined;
                const key = try std.fmt.bufPrint(&kbuf, "{d}", .{v.integer});
                if (hv.object.get(key)) |per_order| {
                    const flat = try a.alloc(usize, (cfg.max_ngram - 1) * per_order.array.items[0].array.items.len);
                    for (0..cfg.max_ngram - 1) |ord| {
                        const heads = per_order.array.items[ord].array;
                        for (heads.items, 0..) |m, h| flat[ord * heads.items.len + h] = @intCast(m.integer);
                    }
                    sizes[k] = flat;
                } else {
                    sizes[k] = &.{};
                }
            }
            cfg.engram_sizes = sizes;
        }
        cfg.hash_mult = &.{};
    }
    if (root.get("quantization")) |qv| {
        if (qv == .object) if (qv.object.get("group_size")) |g| {
            if (g == .integer and g.integer > 0 and g.integer <= std.math.maxInt(u32)) cfg.quant_group_size = @intCast(g.integer);
        };
    }
    cfg.engram_table = try parseHy4Spec(a, json_str);
    return cfg;
}

fn hfInt(obj: std.json.ObjectMap, name: []const u8, default: u64) !i64 {
    const v = obj.get(name) orelse return @intCast(default);
    return switch (v) {
        .integer => |n| n,
        else => @intCast(default),
    };
}

fn hfFloat(obj: std.json.ObjectMap, name: []const u8, default: f64) !f64 {
    const v = obj.get(name) orelse return default;
    return switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        else => default,
    };
}

fn intField(obj: std.json.ObjectMap, name: []const u8) !i64 {
    const v = obj.get(name) orelse return error.MissingConfigField;
    return switch (v) {
        .integer => |n| n,
        else => error.MissingConfigField,
    };
}

fn floatField(obj: std.json.ObjectMap, name: []const u8) !f64 {
    const v = obj.get(name) orelse return error.MissingConfigField;
    return switch (v) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        else => error.MissingConfigField,
    };
}
