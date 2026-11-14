//! The Hugging Face hub a download reads: `HF_ENDPOINT`, else huggingface.co.
const std = @import("std");
const env = @import("../../env.zig");
const testing = std.testing;

const default_hub = "https://huggingface.co";

/// The hub base URL from an `HF_ENDPOINT` value: a mirror (hf-mirror.com) keeps the hub's
/// `api/models/...` and `resolve/main/...` layout. Unset or blank = huggingface.co.
pub fn base(raw: ?[]const u8) error{InvalidHfEndpoint}![]const u8 {
    const v = std.mem.trim(u8, raw orelse return default_hub, " \t\r\n");
    if (v.len == 0) return default_hub;
    const trimmed = std.mem.trimEnd(u8, v, "/");
    for ([_][]const u8{ "https://", "http://" }) |scheme| {
        if (std.ascii.startsWithIgnoreCase(trimmed, scheme) and trimmed.len > scheme.len) return trimmed;
    }
    return error.InvalidHfEndpoint;
}

pub fn envBase() error{InvalidHfEndpoint}![]const u8 {
    return base(env.span(.HF_ENDPOINT));
}

test "hf_hub.base: HF_ENDPOINT picks the hub, unset or blank is huggingface.co, a value without a scheme is refused" {
    try testing.expectEqualStrings("https://huggingface.co", try base(null));
    try testing.expectEqualStrings("https://huggingface.co", try base(" \n"));
    try testing.expectEqualStrings("https://hf-mirror.com", try base("https://hf-mirror.com"));
    try testing.expectEqualStrings("https://hf-mirror.com", try base(" https://hf-mirror.com//\n"));
    try testing.expectEqualStrings("http://127.0.0.1:8080/hub", try base("http://127.0.0.1:8080/hub/"));
    try testing.expectEqualStrings("HTTPS://hf-mirror.com", try base("HTTPS://hf-mirror.com"));
    try testing.expectError(error.InvalidHfEndpoint, base("hf-mirror.com"));
    try testing.expectError(error.InvalidHfEndpoint, base("https://"));
    try testing.expectError(error.InvalidHfEndpoint, base("ftp://hf-mirror.com"));
}
