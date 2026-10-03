//! ZCode 3.14's versioned Personal Provider Config contract. No model-name
//! matching: every chat row receives the serving endpoint's advertised budget.
const std = @import("std");

fn quoted(a: std.mem.Allocator, value: []const u8) ![]u8 {
    return std.json.Stringify.valueAlloc(a, value, .{});
}

pub fn configJson(a: std.mem.Allocator, base_url: []const u8, model: []const u8, entries: anytype) ![]u8 {
    var out = std.ArrayList(u8).empty;
    errdefer out.deinit(a);
    const endpoint = try std.fmt.allocPrint(a, "{s}/v1", .{base_url});
    defer a.free(endpoint);
    const url = try quoted(a, endpoint);
    defer a.free(url);
    const selected = try quoted(a, model);
    defer a.free(selected);
    try out.print(a,
        \\{{"schemaVersion":1,"config":{{
        \\"providerOrder":["mlx"],
        \\"defaultModelSelection":{{"providerId":"mlx","modelId":{s},"options":{{"reasoningLevel":"medium"}}}},
        \\"providerConfigRules":{{"providerRules":[{{"providerId":"mlx","providerName":"MLX Serve / Sushi","enabled":true,"config":{{
        \\"group":"standard-personal","access":{{"type":"api-key","apiKey":"mlx-serve"}},
        \\"api":{{"type":"openai-chat-completions","baseUrl":{s}}},"personalModelIds":[
    , .{ selected, url });
    for (entries, 0..) |e, i| {
        const id = try quoted(a, e.id);
        defer a.free(id);
        try out.print(a, "{s}{s}", .{ if (i == 0) "" else ",", id });
    }
    try out.appendSlice(a, "]}}]},\"modelConfigRules\":{\"manualProviderModelRules\":[],\"providerModelRules\":[");
    for (entries, 0..) |e, i| {
        const id = try quoted(a, e.id);
        defer a.free(id);
        try out.print(a,
            \\{s}{{"providerId":"mlx","modelId":{s},"config":{{"enabled":true,
            \\"properties":{{"contextWindow":{d},"requiresMfjsToolSchema":false,
            \\"inputFormat":{{"supportsText":true,"supportsImage":{s},"supportsVideo":false,"supportsAudio":false,"supportsPdf":false}},
            \\"outputFormat":{{"supportsText":true}},"supportsToolCall":true,"supportsJsonSchemaOutput":false,
            \\"supportsNativeWebSearch":false,"supportsMidConversationSystem":false}},
            \\"optionSpecs":{{"reasoningLevel":{{"values":["none","low","medium","high"],"map":"{{\"reasoning_effort\": reasoningLevel}}"}},
            \\"maxOutputTokens":{{"max":{d},"map":"{{\"max_tokens\": maxOutputTokens}}"}}}}}}}}
        , .{ if (i == 0) "" else ",", id, e.budget.context, if (e.vision) "true" else "false", e.budget.output });
    }
    try out.appendSlice(a, "]}}}\n");
    return out.toOwnedSlice(a);
}
