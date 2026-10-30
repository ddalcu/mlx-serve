//! Stamps per-model settings onto a config at every load site.

const std = @import("std");
const model_mod = @import("../../sdk/config/model.zig");
const chat_mod = @import("../chat/chat.zig");
const model_settings = @import("model_settings.zig");
const ModelConfig = model_mod.ModelConfig;
const ChatConfig = chat_mod.ChatConfig;
const testing = std.testing;

/// Both load construction sites (here and main.zig's startup load) stamp the
/// per-model settings onto the config the bills and defaults read.
/// The kwargs strings MOVE to the freshly loaded `chat_config` (same allocator).
/// `mtp_flag` false (`--no-mtp`) stamps the head off unless the model overrides it.
pub fn applyModelSettings(config: *ModelConfig, chat_config: *ChatConfig, o: *model_settings.Override, mtp_flag: bool) void {
    config.ctx_override = o.ctx_size orelse 0;
    config.kv_quant_override = o.kv_quant;
    config.mtp_override = o.mtp orelse if (mtp_flag) null else false;
    config.mtp_acceptance_override = o.mtp_acceptance;
    config.mtp_greedy_tail_override = o.mtp_greedy_tail;
    config.int8_prefill_override = o.int8_prefill;
    config.lm_head_mxfp8_override = o.lm_head_mxfp8;
    config.ane_override = o.aneChoice();
    config.stream_experts_override = o.stream_experts;
    config.drafter_override = o.drafter;
    o.drafter = null;
    chat_config.chat_template_kwargs = o.chat_template_kwargs;
    o.chat_template_kwargs = null;
}

// ── tests ──

test "applyModelSettings: model MTP setting outranks the launch default" {
    var cc = ChatConfig{ .chat_template = "", .bos_token = null, .eos_token = null, .add_bos_token = false, .allocator = testing.allocator };
    for ([_]struct { flag: bool, setting: ?bool, want: ?bool }{
        .{ .flag = false, .setting = null, .want = false },
        .{ .flag = false, .setting = true, .want = true },
        .{ .flag = false, .setting = false, .want = false },
        .{ .flag = true, .setting = null, .want = null },
        .{ .flag = true, .setting = false, .want = false },
        .{ .flag = true, .setting = true, .want = true },
    }) |c| {
        var cfg = ModelConfig{};
        var o = model_settings.Override{ .mtp = c.setting };
        applyModelSettings(&cfg, &cc, &o, c.flag);
        try testing.expectEqual(c.want, cfg.mtp_override);
    }
}
