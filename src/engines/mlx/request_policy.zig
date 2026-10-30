//! The engine's half of the request policy: what a loaded model can draft with (`host/api/request_options.zig` reads
//! it through `mlx.facts`).

const std = @import("std");
const env = @import("../../env.zig");
const generate_mod = @import("generate.zig");
const transformer_mod = @import("transformer.zig");
const model_mod = @import("../../sdk/config/model.zig");
const dsv41_mod = @import("../../plugins/deepseek_v41/deepseek_v41.zig");
const v41_plugin = @import("../../plugins/deepseek_v41/plugin.zig");
const plugin_config = @import("plugin/config.zig");
const plugin_driver = @import("plugin/driver.zig");
const LoadedModel = @import("../../host/registry/registry.zig").LoadedModel;
const request_options = @import("../../host/api/request_options.zig");
const mlxPropsSettings = @import("../../host/api/props.zig").mlxPropsSettings;
const settingsPropsJson = @import("../../host/api/props.zig").settingsPropsJson;
const Transformer = transformer_mod.Transformer;
const testing = std.testing;

/// Does this model serve DeepSeek-V4 or V4.1 with DSpark draft stages loaded?
pub fn dsv4DraftStages(lm: *const LoadedModel) bool {
    if (lm.config) |cfg| if (cfg.mtp_override == false) return false;
    if (!generate_mod.Generator.dsparkEnabled()) return false;
    const x = lm.transformer orelse return false;
    return x.dsparkStages() > 0;
}

// ── tests ──

const mtpCapable = request_options.mtpCapable;
const defaultEnableMtp = request_options.defaultEnableMtp;

test "native speculation: resolved opt-out gates capability and API defaults" {
    const saved = env.override(.MLX_SERVE_DSV4_DSPARK, null);
    defer saved.restore();
    // A served V4.1 whose module loaded three DSpark stages.
    var native: dsv41_mod.Dsv41Model = undefined;
    native.n_mtp = 3;
    var served: v41_plugin.Model = undefined;
    served.mdl = &native;
    var binding: plugin_config.Binding = .{ .table = &v41_plugin.table };
    binding.parsed.caps.owns_decode_state = true;
    var bound: plugin_driver.Bound = .{ .table = &v41_plugin.table, .model = &served };
    var cfg = model_mod.ModelConfig{ .plugin = &binding };
    var xfm: Transformer = undefined;
    xfm.config = cfg;
    xfm.mlx_stream_ext = null;
    xfm.plugin = &bound;
    xfm.dense_attn_proj = false;
    var lm: LoadedModel = undefined;
    lm.config = &cfg;
    lm.transformer = &xfm;
    lm.mtp = null;
    lm.engine = null;
    lm.dflash = null;
    lm.drafter = null;
    lm.mtp_depth = 6;
    for ([_]?[:0]const u8{ null, "1", "0" }) |raw| {
        if (raw) |value| env.set(.MLX_SERVE_DSV4_DSPARK, value) else env.unset(.MLX_SERVE_DSV4_DSPARK);
        const env_enabled = if (raw) |value| value[0] != '0' else true;
        for ([_]?bool{ null, true, false }) |setting| {
            cfg.mtp_override = setting;
            xfm.config.mtp_override = setting;
            const enabled = setting != false and env_enabled;
            try testing.expectEqual(enabled, mtpCapable(&lm));
            try testing.expectEqual(enabled, defaultEnableMtp(false, dsv4DraftStages(&lm)));
            const frag = try settingsPropsJson(testing.allocator, mlxPropsSettings(&lm));
            defer testing.allocator.free(frag);
            var parsed = try std.json.parseFromSlice(std.json.Value, testing.allocator, frag[",\"settings\":".len..], .{});
            defer parsed.deinit();
            const mtp = parsed.value.object.get("mtp").?.object;
            try testing.expect(mtp.get("loaded").?.bool);
            try testing.expectEqual(enabled, mtp.get("default_on").?.bool);
        }
    }
}
