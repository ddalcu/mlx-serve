//! Where Hugging Face downloads come from: `HF_ENDPOINT` (a mirror such as https://hf-mirror.com), else
//! huggingface.co. The variable is the one `huggingface_hub` and the mirrors' own docs use.

const std = @import("std");
const env = @import("../../env.zig");

pub const default = "https://huggingface.co";

/// The hub's base URL, no trailing slash.
pub fn base() []const u8 {
    return baseFrom(if (env.get(.HF_ENDPOINT)) |p| std.mem.span(p) else null);
}

/// `raw` as a base URL. A value that is not an http(s) URL with a host is ignored: `HF_TOKEN` rides every request
/// to the base, so it must name a hub and nothing else.
pub fn baseFrom(raw: ?[]const u8) []const u8 {
    const url = std.mem.trimEnd(u8, std.mem.trim(u8, raw orelse return default, " \t\r\n"), "/");
    inline for (.{ "https://", "http://" }) |scheme| {
        if (std.mem.startsWith(u8, url, scheme)) {
            const host = url[scheme.len..];
            if (host.len > 0 and std.mem.indexOfAny(u8, host, " \t\r\n\"'<>") == null and host[0] != '/') return url;
        }
    }
    return default;
}

test "hub base: HF_ENDPOINT names the hub, without its trailing slash; anything that is not an http(s) URL is ignored" {
    const t = std.testing;
    try t.expectEqualStrings(default, baseFrom(null));
    try t.expectEqualStrings("https://hf-mirror.com", baseFrom("https://hf-mirror.com"));
    try t.expectEqualStrings("https://hf-mirror.com", baseFrom(" https://hf-mirror.com/ \n"));
    try t.expectEqualStrings("http://127.0.0.1:8123/hub", baseFrom("http://127.0.0.1:8123/hub//"));
    for ([_][]const u8{ "", "  ", "hf-mirror.com", "ftp://hf-mirror.com", "https://", "https:///x", "file:///etc", "https://a b" }) |bad|
        try t.expectEqualStrings(default, baseFrom(bad));
}
