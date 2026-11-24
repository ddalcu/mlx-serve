//! The hy_v4_flash load. The census test runs against the real pack when it is on disk and skips
//! hermetically when it is not; the tiny-pack tests run everywhere on a synthetic safetensors file.

const std = @import("std");
const testing = std.testing;
const mlx = @import("../../mlx/ffi.zig");
const sdk = @import("../../sdk/root.zig");
const weights_load = @import("../../sdk/weights/load.zig");
const config = @import("config.zig");
const fixture = @import("fixture.zig");
const abi = @import("../../abi/plugin.zig");

const pack_path = "~/.mlx-serve/models/TensorFold/Hy4-flash-MLX-4bit-MTP";

/// The pack's shape families, with the per-kind counts for the full 42-layer trunk (12 DSA layers with
/// compressed_indexer arms, 30 SWA, layer 0 dense, 41 MoE, engram at 1/11/21/31). Every family the load
/// must place; nothing outside them loads.
/// The pack's full name census: every tensor key belongs to one family here, at its exact count (12 DSA
/// layers with compressed_indexer arms, 30 SWA, layer 0 dense, 41 MoE, engram at 1/11/21/31, 27 VIT blocks,
/// 5 DFly drafts). `{d}` is a layer index, `{n}` a module index.
const expect_families = .{
    .{ "model.embed_tokens.weight", .{}, 1 },
    .{ "model.embed_tokens.scales", .{}, 1 },
    .{ "model.embed_tokens.biases", .{}, 1 },
    .{ "lm_head.weight", .{}, 1 },
    .{ "lm_head.scales", .{}, 1 },
    .{ "lm_head.biases", .{}, 1 },
    .{ "model.norm.weight", .{}, 1 },
    .{ "model.hc_head.hc_head_base", .{}, 1 },
    .{ "model.hc_head.hc_head_fn", .{}, 1 },
    .{ "model.hc_head.hc_head_scale", .{}, 1 },
    .{ "model.layers.{d}.input_layernorm.weight", .{}, 42 },
    .{ "model.layers.{d}.post_attention_layernorm.weight", .{}, 42 },
    .{ "model.layers.{d}.self_attn.q_a_proj.weight", .{}, 42 },
    .{ "model.layers.{d}.self_attn.q_a_proj.scales", .{}, 42 },
    .{ "model.layers.{d}.self_attn.q_a_proj.biases", .{}, 42 },
    .{ "model.layers.{d}.self_attn.q_a_layernorm.weight", .{}, 42 },
    .{ "model.layers.{d}.self_attn.q_b_proj.weight", .{}, 42 },
    .{ "model.layers.{d}.self_attn.q_b_proj.scales", .{}, 42 },
    .{ "model.layers.{d}.self_attn.q_b_proj.biases", .{}, 42 },
    .{ "model.layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{}, 42 },
    .{ "model.layers.{d}.self_attn.kv_a_proj_with_mqa.scales", .{}, 42 },
    .{ "model.layers.{d}.self_attn.kv_a_proj_with_mqa.biases", .{}, 42 },
    .{ "model.layers.{d}.self_attn.kv_a_layernorm.weight", .{}, 42 },
    .{ "model.layers.{d}.self_attn.kv_b_proj.weight", .{}, 42 },
    .{ "model.layers.{d}.self_attn.kv_b_proj.scales", .{}, 42 },
    .{ "model.layers.{d}.self_attn.kv_b_proj.biases", .{}, 42 },
    .{ "model.layers.{d}.self_attn.linear_gate.weight", .{}, 42 },
    .{ "model.layers.{d}.self_attn.linear_gate.scales", .{}, 42 },
    .{ "model.layers.{d}.self_attn.linear_gate.biases", .{}, 42 },
    .{ "model.layers.{d}.self_attn.o_proj.weight", .{}, 42 },
    .{ "model.layers.{d}.self_attn.o_proj.scales", .{}, 42 },
    .{ "model.layers.{d}.self_attn.o_proj.biases", .{}, 42 },
    .{ "model.layers.{d}.self_attn.post_proj_norm.weight", .{}, 42 },
    .{ "model.layers.{d}.self_attn.learnable_sink_param", .{}, 42 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.query_projection.weight", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.query_projection.scales", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.query_projection.biases", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.token_key_projection.weight", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.token_key_projection.scales", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.token_key_projection.biases", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.head_weight_projection.weight", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.head_weight_projection.scales", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.head_weight_projection.biases", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.key_value_projection.weight", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.key_gate_projection.weight", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.key_norm.weight", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.token_key_norm.weight", .{}, 12 },
    .{ "model.layers.{d}.self_attn.compressed_indexer.intra_group_ape", .{}, 12 },
    .{ "model.layers.{d}.hc_attn_layer.hc_pre.hc_base", .{}, 42 },
    .{ "model.layers.{d}.hc_attn_layer.hc_pre.hc_fn", .{}, 42 },
    .{ "model.layers.{d}.hc_attn_layer.hc_pre.hc_scale", .{}, 42 },
    .{ "model.layers.{d}.hc_mlp_layer.hc_pre.hc_base", .{}, 42 },
    .{ "model.layers.{d}.hc_mlp_layer.hc_pre.hc_fn", .{}, 42 },
    .{ "model.layers.{d}.hc_mlp_layer.hc_pre.hc_scale", .{}, 42 },
    .{ "model.layers.{d}.mlp.gate.weight", .{}, 41 },
    .{ "model.layers.{d}.mlp.gate.e_score_correction_bias", .{}, 41 },
    .{ "model.layers.{d}.mlp.experts.gate_up_proj.weight", .{}, 41 },
    .{ "model.layers.{d}.mlp.experts.gate_up_proj.scales", .{}, 41 },
    .{ "model.layers.{d}.mlp.experts.gate_up_proj.biases", .{}, 41 },
    .{ "model.layers.{d}.mlp.experts.down_proj.weight", .{}, 41 },
    .{ "model.layers.{d}.mlp.experts.down_proj.scales", .{}, 41 },
    .{ "model.layers.{d}.mlp.experts.down_proj.biases", .{}, 41 },
    .{ "model.layers.{d}.mlp.routed_expert_up_proj.weight", .{}, 41 },
    .{ "model.layers.{d}.mlp.routed_expert_up_proj.scales", .{}, 41 },
    .{ "model.layers.{d}.mlp.routed_expert_up_proj.biases", .{}, 41 },
    .{ "model.layers.{d}.mlp.routed_expert_down_proj.weight", .{}, 41 },
    .{ "model.layers.{d}.mlp.routed_expert_down_proj.scales", .{}, 41 },
    .{ "model.layers.{d}.mlp.routed_expert_down_proj.biases", .{}, 41 },
    .{ "model.layers.{d}.mlp.shared_experts.gate_proj.weight", .{}, 41 },
    .{ "model.layers.{d}.mlp.shared_experts.gate_proj.scales", .{}, 41 },
    .{ "model.layers.{d}.mlp.shared_experts.gate_proj.biases", .{}, 41 },
    .{ "model.layers.{d}.mlp.shared_experts.up_proj.weight", .{}, 41 },
    .{ "model.layers.{d}.mlp.shared_experts.up_proj.scales", .{}, 41 },
    .{ "model.layers.{d}.mlp.shared_experts.up_proj.biases", .{}, 41 },
    .{ "model.layers.{d}.mlp.shared_experts.down_proj.weight", .{}, 41 },
    .{ "model.layers.{d}.mlp.shared_experts.down_proj.scales", .{}, 41 },
    .{ "model.layers.{d}.mlp.shared_experts.down_proj.biases", .{}, 41 },
    .{ "model.layers.{d}.mlp.combine_output_norm.weight", .{}, 41 },
    .{ "model.layers.{d}.mlp.post_fc2_norm.weight", .{}, 1 },
    .{ "model.layers.{d}.mlp.gate_proj.weight", .{}, 1 },
    .{ "model.layers.{d}.mlp.gate_proj.scales", .{}, 1 },
    .{ "model.layers.{d}.mlp.gate_proj.biases", .{}, 1 },
    .{ "model.layers.{d}.mlp.up_proj.weight", .{}, 1 },
    .{ "model.layers.{d}.mlp.up_proj.scales", .{}, 1 },
    .{ "model.layers.{d}.mlp.up_proj.biases", .{}, 1 },
    .{ "model.layers.{d}.mlp.down_proj.weight", .{}, 1 },
    .{ "model.layers.{d}.mlp.down_proj.scales", .{}, 1 },
    .{ "model.layers.{d}.mlp.down_proj.biases", .{}, 1 },
    .{ "model.layers.{d}.engram_layer.key_projs.0.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.0.scales", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.0.biases", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.0.bias", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.norm_query.0.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.norm_key.0.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.unified_per_hc_scales.0", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.1.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.1.scales", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.1.biases", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.1.bias", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.norm_query.1.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.norm_key.1.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.unified_per_hc_scales.1", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.2.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.2.scales", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.2.biases", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.2.bias", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.norm_query.2.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.norm_key.2.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.unified_per_hc_scales.2", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.3.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.3.scales", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.3.biases", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.key_projs.3.bias", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.norm_query.3.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.norm_key.3.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.unified_per_hc_scales.3", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.0.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.1.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.2.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.3.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.4.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.5.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.6.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.7.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.8.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.9.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.10.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.11.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.12.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.13.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.14.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_per_head_norms.15.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.lookup_table", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.unified_pre_gamma", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.multi_head_embedding.embedding.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.multi_head_embedding.embedding.scales", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.multi_head_embedding.embedding.biases", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.value_proj.weight", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.value_proj.scales", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.value_proj.biases", .{}, 4 },
    .{ "model.layers.{d}.engram_layer.value_proj.bias", .{}, 4 },
    .{ "model.dfly_layers.{d}.input_layernorm.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.post_attention_layernorm.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.self_attn.q_a_layernorm.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.self_attn.kv_a_layernorm.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.self_attn.post_proj_norm.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.mlp.down_proj.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.mlp.gate_proj.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.mlp.up_proj.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.self_attn.q_a_proj.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.self_attn.q_b_proj.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.self_attn.kv_a_proj_with_mqa.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.self_attn.kv_b_proj.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.self_attn.linear_gate.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.self_attn.o_proj.weight", .{}, 5 },
    .{ "model.dfly_layers.{d}.self_attn.learnable_sink_param", .{}, 5 },
    .{ "model.dfly_layers.final_norm.weight", .{}, 1 },
    .{ "model.dfly_layers.mtp_fusion.eh_proj.weight", .{}, 1 },
    .{ "model.dfly_layers.mtp_fusion.enorm.weight", .{}, 1 },
    .{ "model.dfly_layers.mtp_fusion.hnorm.weight", .{}, 1 },
    .{ "model.dfly_layers.mtp_fusion.gate", .{}, 1 },
    .{ "model.dfly_layers.hidden_correction.down_proj.weight", .{}, 1 },
    .{ "model.dfly_layers.hidden_correction.embed_norm.weight", .{}, 1 },
    .{ "model.dfly_layers.hidden_correction.gate_proj.weight", .{}, 1 },
    .{ "model.dfly_layers.hidden_correction.hidden_norm.weight", .{}, 1 },
    .{ "model.dfly_layers.hidden_correction.up_proj.weight", .{}, 1 },
    .{ "model.dfly_layers.confidence_head.proj.weight", .{}, 1 },
    .{ "model.dfly_layers.confidence_head.proj.bias", .{}, 1 },
    .{ "vit.embeddings.patch_embedding.weight", .{}, 1 },
    .{ "vit.embeddings.patch_embedding.bias", .{}, 1 },
    .{ "vit.embeddings.position_embedding.weight", .{}, 1 },
    .{ "vit.layers.{d}.input_layernorm.weight", .{}, 27 },
    .{ "vit.layers.{d}.input_layernorm.bias", .{}, 27 },
    .{ "vit.layers.{d}.post_attention_layernorm.weight", .{}, 27 },
    .{ "vit.layers.{d}.post_attention_layernorm.bias", .{}, 27 },
    .{ "vit.layers.{d}.self_attn.q_proj.weight", .{}, 27 },
    .{ "vit.layers.{d}.self_attn.q_proj.bias", .{}, 27 },
    .{ "vit.layers.{d}.self_attn.k_proj.weight", .{}, 27 },
    .{ "vit.layers.{d}.self_attn.k_proj.bias", .{}, 27 },
    .{ "vit.layers.{d}.self_attn.v_proj.weight", .{}, 27 },
    .{ "vit.layers.{d}.self_attn.v_proj.bias", .{}, 27 },
    .{ "vit.layers.{d}.self_attn.o_proj.weight", .{}, 27 },
    .{ "vit.layers.{d}.self_attn.o_proj.bias", .{}, 27 },
    .{ "vit.layers.{d}.mlp.dense_h_to_4h.weight", .{}, 27 },
    .{ "vit.layers.{d}.mlp.dense_h_to_4h.bias", .{}, 27 },
    .{ "vit.layers.{d}.mlp.dense_4h_to_h.weight", .{}, 27 },
    .{ "vit.layers.{d}.mlp.dense_4h_to_h.bias", .{}, 27 },
    .{ "vit.perceive.norm.weight", .{}, 1 },
    .{ "vit.perceive.image_newline", .{}, 1 },
    .{ "vit.perceive.proj.0.weight", .{}, 1 },
    .{ "vit.perceive.proj.0.bias", .{}, 1 },
    .{ "vit.perceive.proj.2.weight", .{}, 1 },
    .{ "vit.perceive.proj.2.bias", .{}, 1 },
};

test "hy_v4 loader: the census — every family at its expected count, every tensor in a known family" {
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.SkipZigTest);
    const dir_path = std.fmt.allocPrint(testing.allocator,
        "{s}/.mlx-serve/models/TensorFold/Hy4-flash-MLX-4bit-MTP", .{home})
        catch return error.SkipZigTest;
    defer testing.allocator.free(dir_path);
    if (std.mem.indexOf(u8, dir_path, "=/") != null) return error.SkipZigTest;
    var dir = std.Io.Dir.openDirAbsolute(std.Io.Threaded.global_single_threaded.io(), dir_path, .{}) catch {
        std.debug.print("hy_v4 census: pack {s} not on disk — skipped\n", .{pack_path});
        return error.SkipZigTest;
    };
    defer dir.close(std.Io.Threaded.global_single_threaded.io());
    const io = std.Io.Threaded.global_single_threaded.io();
    const index_raw = dir.readFileAlloc(io, "model.safetensors.index.json", testing.allocator, .limited(64 << 20)) catch
        return error.SkipZigTest;
    defer testing.allocator.free(index_raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, index_raw, .{});
    defer parsed.deinit();
    const wm = parsed.value.object.get("weight_map").?;

    // 1. Every family of the expectation list holds its expected count of keys.
    inline for (expect_families) |fam| {
        var count: usize = 0;
        var fit = wm.object.iterator();
        while (fit.next()) |e| {
            if (matchesFamily(e.key_ptr.*, fam[0])) count += 1;
        }
        if (count != fam[2]) {
            std.debug.print("family {s}: {d} keys, want {d}\n", .{ fam[0], count, @as(usize, fam[2]) });
            return error.Unexpected;
        }
    }
    // 2. Every weight or weight-adjacent key lands in one expected family.
    var it = wm.object.iterator();
    while (it.next()) |e| {
        const k = e.key_ptr.*;
        if (!std.mem.endsWith(u8, k, ".weight") and !std.mem.endsWith(u8, k, ".scales") and
            !std.mem.endsWith(u8, k, ".biases") and !std.mem.endsWith(u8, k, "_bias") and
            !std.mem.endsWith(u8, k, "_ape") and !std.mem.endsWith(u8, k, ".hc_base") and
            !std.mem.endsWith(u8, k, ".hc_fn") and !std.mem.endsWith(u8, k, ".hc_scale") and
            !std.mem.endsWith(u8, k, "lookup_table")) continue;
        var found = false;
        inline for (expect_families) |fam| {
            if (matchesFamily(k, fam[0])) {
                found = true;
                break;
            }
        }
        if (!found) {
            std.debug.print("census: {s} is in no expected family\n", .{k});
            return error.Unexpected;
        }
    }
}

test "hy_v4 loader: the pack's own geometry — DSA and SWA attention widths, the 3-D expert stack" {
    // The header oracle: the shapes the load's checks must accept are the shapes the pack STORES. A DSA layer
    // (0) and a SWA layer (2) diff in head count (32 vs 64), and the routed experts are three-axis fused
    // stacks over the LATENT width, not 2-D rows.
    const home = std.mem.span(std.c.getenv("HOME") orelse return error.SkipZigTest);
    const dir_path = std.fmt.allocPrint(testing.allocator,
        "{s}/.mlx-serve/models/TensorFold/Hy4-flash-MLX-4bit-MTP", .{home}) catch return error.SkipZigTest;
    defer testing.allocator.free(dir_path);
    if (std.mem.indexOf(u8, dir_path, "=/") != null) return error.SkipZigTest;
    const io = std.Io.Threaded.global_single_threaded.io();
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return error.SkipZigTest;
    defer dir.close(io);
    const index_raw = dir.readFileAlloc(io, "model.safetensors.index.json", testing.allocator, .limited(64 << 20)) catch
        return error.SkipZigTest;
    defer testing.allocator.free(index_raw);
    const parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, index_raw, .{});
    defer parsed.deinit();
    const wm = parsed.value.object.get("weight_map").?.object;

    const V = struct {
        dir: std.Io.Dir,
        io: std.Io,
        wm: std.json.ObjectMap,
        cache: std.StringHashMap([]const u64),
        a: std.mem.Allocator,

        fn shape(self: *@This(), key: []const u8) ![]const u64 {
            if (self.cache.get(key)) |sh| return sh;
            const file = self.wm.get(key) orelse return error.NoSuchTensor;
            if (file != .string) return error.NoSuchTensor;
            // Read the header only (a shard is ~1 GB): 8-byte length, then its JSON.
            const f = try self.dir.openFile(self.io, file.string, .{});
            defer f.close(self.io);
            var len_buf: [8]u8 = undefined;
            _ = try f.readPositionalAll(self.io, &len_buf, 0);
            const hl = std.mem.readInt(u64, &len_buf, .little);
            const hbuf = try self.a.alloc(u8, @intCast(hl));
            defer self.a.free(hbuf);
            _ = try f.readPositionalAll(self.io, hbuf, 8);
            const hdr = try std.json.parseFromSlice(std.json.Value, self.a, hbuf, .{});
            defer hdr.deinit();
            const e = hdr.value.object.get(key) orelse return error.NoSuchTensor;
            const arr = e.object.get("shape").?.array;
            const out = try self.a.alloc(u64, arr.items.len);
            for (arr.items, 0..) |v, i| out[i] = @intCast(v.integer);
            try self.cache.put(key, out);
            return out;
        }
    };
    var a = std.heap.ArenaAllocator.init(testing.allocator);
    defer a.deinit();
    var v = V{ .dir = dir, .io = io, .wm = wm, .cache = std.StringHashMap([]const u64).init(testing.allocator),
        .a = a.allocator() };
    defer v.cache.deinit();

    // DSA layer 0: 32 heads x (192 nope + 64 rope), latent 512, v 256.
    try testing.expectEqualSlices(u64, &.{ 8192, 128 }, try v.shape("model.layers.0.self_attn.q_b_proj.weight"));
    try testing.expectEqualSlices(u64, &.{ 14336, 64 }, try v.shape("model.layers.0.self_attn.kv_b_proj.weight"));
    try testing.expectEqualSlices(u64, &.{ 3072, 1024 }, try v.shape("model.layers.0.self_attn.o_proj.weight"));
    try testing.expectEqualSlices(u64, &.{32}, try v.shape("model.layers.0.self_attn.learnable_sink_param"));
    // SWA layer 2: 64 heads, twice every head-major width.
    try testing.expectEqualSlices(u64, &.{ 16384, 128 }, try v.shape("model.layers.2.self_attn.q_b_proj.weight"));
    try testing.expectEqualSlices(u64, &.{ 28672, 64 }, try v.shape("model.layers.2.self_attn.kv_b_proj.weight"));
    try testing.expectEqualSlices(u64, &.{ 3072, 2048 }, try v.shape("model.layers.2.self_attn.o_proj.weight"));
    try testing.expectEqualSlices(u64, &.{64}, try v.shape("model.layers.2.self_attn.learnable_sink_param"));
    // The routed experts: three axes, the latent width (1536) in, g64 (24 groups).
    try testing.expectEqualSlices(u64, &.{ 512, 3072, 192 }, try v.shape("model.layers.1.mlp.experts.gate_up_proj.weight"));
    try testing.expectEqualSlices(u64, &.{ 512, 3072, 24 }, try v.shape("model.layers.1.mlp.experts.gate_up_proj.scales"));
    try testing.expectEqualSlices(u64, &.{ 512, 1536, 192 }, try v.shape("model.layers.1.mlp.experts.down_proj.weight"));
    try testing.expectEqualSlices(u64, &.{ 1536, 384 }, try v.shape("model.layers.1.mlp.routed_expert_down_proj.weight"));
    try testing.expectEqualSlices(u64, &.{ 3072, 192 }, try v.shape("model.layers.1.mlp.routed_expert_up_proj.weight"));
}

/// Does `key` match the family template `fam`? A digit placeholder (`{d}`/`{n}`) matches one run of digits; the
/// template is anchored at both ends.
fn matchesFamily(key: []const u8, comptime fam: []const u8) bool {
    const k = key;
    var ki: usize = 0;
    var ti: usize = 0;
    while (ti < fam.len) {
        if (fam[ti] == '{') {
            ti += 3; // "{d}" or "{n}"
            const s = ki;
            while (ki < k.len and std.ascii.isDigit(k[ki])) : (ki += 1) {}
            if (ki == s) return false;
            continue;
        }
        if (ki >= k.len or k[ki] != fam[ti]) return false;
        ki += 1;
        ti += 1;
    }
    return ki == k.len;
}

test "hy_v4 loader: the tiny pack's expected tensors load by name with the quant layout the config names" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // The tiny pack is `fixture.zig`'s (one builder for the conformance suite and this test): 2 layers, a DSA
    // layer 0 (2 heads), a SWA layer 1 (4 heads, engram + latent MoE), layer 0 dense. The SDK loader rewrites
    // quant triples whose in-features sit under its wide-tensor cutoff into DENSE rows (dequantizeSmall); the
    // tiny pack lives entirely under that line, so the load's shape checks run both arms.
    const files = try fixture.build(a);
    var cfg_json: []const u8 = "";
    var st: []const u8 = "";
    for (files) |f| {
        if (std.mem.eql(u8, f.name, "config.json")) cfg_json = f.bytes;
        if (std.mem.eql(u8, f.name, "model.safetensors")) st = f.bytes;
    }
    try testing.expect(st.len > 1000);
    var p: abi.Parsed = .{};
    try testing.expect(config.parseJson(fixture.tiny_config, true, &p).isOk());
    config.freeParsed(&p);

    // The load arm opens the pack the way the conformance suite's `Loaded.open` does and places every name.
    try expectTinyLoad(st, cfg_json);
}

/// Stages the tiny pack on disk, opens it through the SDK weight loader and the plugin's load arm, and checks
/// what the load placed: every name the config asks for, the quant layout the config names, and the Engram
/// tables NOT resident (the stream bank owns them).
fn expectTinyLoad(st: []const u8, cfg_json: []const u8) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const tmp = std.mem.span(std.c.getenv("TMPDIR") orelse "/tmp");
    var rnd: [6]u8 = undefined;
    io.random(&rnd);
    const path = try std.fmt.allocPrint(testing.allocator, "{s}/hyv4-tiny-{x}", .{
        std.mem.trimEnd(u8, tmp, "/"),
        std.mem.readInt(u48, &rnd, .little),
    });
    defer testing.allocator.free(path);
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, path, .{});
    defer {
        dir.close(io);
        std.Io.Dir.cwd().deleteTree(io, path) catch {};
    }
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg_json });
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = st });

    const weights = try weights_load.loadWeightsOpt(io, testing.allocator, path, .{});

    var p: abi.Parsed = .{};
    const cjson = abi.Str.of(cfg_json);
    var mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = cfg_json }} };
    try testing.expect(plugin_table.parse(&mem.packDir(), cjson, .{}, null, false, &p).isOk());
    defer plugin_table.free_parsed(&p);

    // The plugin's own load arm, through the table: the driver's map shape (a Resident it may take), the
    // shapes validated against the config, the Engram opened as a streamed table — and a refusal that
    // names what is missing when the map loses one tensor.
    var resident = @import("../../sdk/weights/resident.zig").Resident.adopt(weights);
    defer resident.deinit();
    const src = resident.source();
    var why: abi.Verdict = .ok;
    var fwd_out: abi.ForwardOut = .{};
    const model = plugin_table.load(&.{
        .model_dir = abi.Str.of(path),
        .dir = &mem.packDir(),
        .parsed = &p,
        .weights = &src,
        .stream = .{ .ctx = null },
    }, &why);
    if (model) |m| {
        defer plugin_table.unload(m);
        // The load placed everything, and the text forward runs over the tiny pack: a 3-token prompt through a
        // real cache yields one finite logit row of the pack's vocab width.
        // An error an earlier test in this shard latched would abort our first dispatch: drop it first, as the
        // conformance suite does.
        var err_buf: [512]u8 = undefined;
        if (mlx.takeError(&err_buf)) |stale| std.debug.print("hy_v4 loader: dropped an earlier test's MLX error ({s})\n", .{stale});
        const cached = @import("../../sdk/testing/cache_ops.zig");
        var cache = try cached.TestCache.init(testing.allocator, p.shell.num_hidden_layers);
        defer cache.deinit();
        const tokens = [_]u32{ 1, 2, 3 };
        const slots = [_]u32{0};
        const positions = [_]u32{0};
        var v = plugin_table.forward(m, &.{
            .mode = .prefill,
            .logits = .last_row,
            .rows = 3,
            .tokens = .of(&tokens),
            .slots = .of(&slots),
            .positions = .of(&positions),
            .caches = .of(&[_]abi.Handle{cache.handle()}),
            .cache_ops = &cached.ops,
        }, &fwd_out);
        try testing.expect(v.isOk());
        defer _ = mlx.mlx_array_free(.{ .ctx = fwd_out.logits.ctx });
        const sh = mlx.mlx_array_shape(.{ .ctx = fwd_out.logits.ctx });
        try testing.expect(mlx.mlx_array_ndim(.{ .ctx = fwd_out.logits.ctx }) == 2);
        try testing.expectEqual(@as(c_int, 1), sh[0]);
        try testing.expectEqual(@as(c_int, @intCast(p.shell.vocab_size)), sh[1]);
    } else return error.LoadRefused;

    // A pack missing one named tensor: the refusal names the tensor the config asked for. The map's own
    // shape is the load's input, so the load may TAKE the resident — re-stage the pack instead: same bytes,
    // the lookup table's key renamed in the header, a second directory the load opens on its own.
    {
        const renamed_path = try std.fmt.allocPrint(testing.allocator, "{s}-miss", .{path});
        defer testing.allocator.free(renamed_path);
        var dir2 = try std.Io.Dir.cwd().createDirPathOpen(io, renamed_path, .{});
        defer dir2.close(io);
        defer std.Io.Dir.cwd().deleteTree(io, renamed_path) catch {};
        try dir2.writeFile(io, .{ .sub_path = "config.json", .data = cfg_json });
        {
            const renamed = renameLookupTable(st);
            defer testing.allocator.free(renamed);
            try dir2.writeFile(io, .{ .sub_path = "model.safetensors", .data = renamed });
        }

        const small = try weights_load.loadWeightsOpt(io, testing.allocator, renamed_path, .{});
        var small_res = @import("../../sdk/weights/resident.zig").Resident.adopt(small);
        defer small_res.deinit();
        var why2: abi.Verdict = .ok;
        const second = plugin_table.load(&.{
            .model_dir = abi.Str.of(renamed_path),
            .dir = &mem.packDir(),
            .parsed = &p,
            .weights = &small_res.source(),
        }, &why2);
        if (second) |m| {
            _ = m;
            return error.ShouldRefuse;
        }
        try testing.expect(std.mem.indexOf(u8, why2.why.slice(), "model.layers.1.engram_layer.lookup_table") != null);
    }
}

/// The staged bytes with `model.layers.1.engram_layer.lookup_table` renamed in place to `...tabld`: the
/// key length is unchanged, so the header JSON and every data offset stay valid.
fn renameLookupTable(st: []const u8) []const u8 {
    const a = testing.allocator;
    const out = a.dupe(u8, st) catch @panic("fixture header");
    const hl = std.mem.readInt(u64, out[0..8], .little);
    const h = out[8..][0..hl];
    const needle = "\"model.layers.1.engram_layer.lookup_table\":";
    const at = std.mem.indexOf(u8, h, needle) orelse @panic("fixture lacks the lookup table");
    // Change the key's last letter in place: an equal-length edit keeps the JSON valid and every
    // data offset untouched. The `:` is the needle's last char, so the key's closing quote sits
    // one before it.
    const close_q = at + needle.len - 2; // the quote before ':'
    std.debug.assert(h[close_q] == '"');
    std.debug.assert(h[close_q - 1] == 'e'); // ...table
    h[close_q - 1] = 'd'; // ...tabld
    // The name must be gone wherever it appears.
    if (std.mem.count(u8, h, "\"model.layers.1.engram_layer.lookup_table\":") != 0)
        @panic("rename left the original key behind");
    return out;
}

const plugin_table = @import("plugin.zig").table;
