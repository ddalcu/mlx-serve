//! Qwen3-ASR packs (`qwen3_asr`, mlx-audio's layout, the forced aligner's too): the claim, and the parse of the dense
//! qwen3 trunk nested under `thinker_config` through the qwen3 plugin's parser. MLX-free.

const abi = @import("../../abi/plugin.zig");
const sdk = @import("../../sdk/root.zig");
const q3 = @import("../qwen3/config.zig");

pub const plugin = @import("../../sdk/plugin.zig").Plugin{
    .name = "qwen3_asr",
    .kind = .model,
    .claims = &model_types,
    .gate_models = &.{ "mlx-community/Qwen3-ASR-0.6B-8bit", "mlx-community/Qwen3-ForcedAligner-0.6B-8bit" },
};

/// The aligner's pack says `qwen3_asr` at its root; discovery names it by its thinker's type.
pub const model_types = [_][]const u8{ "qwen3_asr", "qwen3_forced_aligner" };

pub const claims = sdk.claimsByType(&model_types);

pub const freeParsed = q3.freeParsed;

pub fn parse(dir: *const abi.PackDir, config_json: abi.Str, settings_json: abi.Str, span: ?*const abi.LayerSpan, stub: bool, out: *abi.Parsed) callconv(.c) abi.Verdict {
    _ = settings_json;
    return q3.parseAs(true, dir, config_json, span, stub, out);
}

// ── tests ──

const std = @import("std");
const testing = std.testing;

/// mlx-community/Qwen3-ASR-0.6B-8bit, abridged.
pub const asr_json =
    \\{"model_type":"qwen3_asr","quantization":{"group_size":64,"bits":8,"mode":"affine"},
    \\ "thinker_config":{"model_type":"qwen3_asr","audio_token_id":151676,"audio_start_token_id":151669,
    \\  "audio_end_token_id":151670,
    \\  "audio_config":{"d_model":896,"encoder_layers":18,"encoder_attention_heads":14,"encoder_ffn_dim":3584,
    \\   "num_mel_bins":128,"downsample_hidden_size":480,"n_window":50,"n_window_infer":800,"output_dim":1024,
    \\   "max_source_positions":1500},
    \\  "text_config":{"model_type":"qwen3","hidden_size":1024,"intermediate_size":3072,"num_hidden_layers":28,
    \\   "num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"rms_norm_eps":1e-06,"rope_theta":1000000,
    \\   "tie_word_embeddings":true,"vocab_size":151936,"hidden_act":"silu",
    \\   "rope_scaling":{"interleaved":true,"mrope_interleaved":true,"mrope_section":[24,20,20],"rope_type":"default"}}}}
;

test "qwen3_asr config: the thinker's qwen3 trunk, the outer quantization, audio rows at audio_token_id" {
    const mem: sdk.fixture.MemDir = .{ .files = &.{.{ .name = "config.json", .bytes = asr_json }} };
    const dir = mem.packDir();
    var p: abi.Parsed = .{};
    try testing.expect(parse(&dir, abi.Str.of(asr_json), .{}, null, true, &p).isOk());
    defer freeParsed(&p);
    try testing.expectEqualStrings("qwen3", p.shell.model_type.slice());
    try testing.expectEqual(@as(u32, 28), p.shell.num_hidden_layers);
    try testing.expectEqual(@as(u8, 8), p.shell.quant.bits);
    try testing.expectEqual(@as(u32, 151676), p.placeholder_ids.audio);
    try testing.expect(q3.of(&p).tie_word_embeddings);
    // The plain qwen3 parse never claims it.
    var p2: abi.Parsed = .{};
    try testing.expectEqual(abi.Verdict.Code.unsupported, q3.parse(&dir, abi.Str.of(asr_json), .{}, null, true, &p2).code);
}
