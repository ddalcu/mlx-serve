//! English number words for TTS text frontends: Kokoro's plain form and inflect's `number_to_words` (Ming-3.0-tts).

const std = @import("std");

pub const ONES = [_][]const u8{ "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen" };
const TENS = [_][]const u8{ "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety" };
const SCALES = [_][]const u8{ "", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion", "sextillion", "septillion", "octillion", "nonillion", "decillion" };

pub const Style = enum {
    /// "one thousand two hundred five", "twenty one". Past 999 trillion, the count of trillions is read as a number.
    plain,
    /// inflect `number_to_words(n, andword="")`: "one thousand, two hundred five", "twenty-one". Past the
    /// decillions, digit by digit.
    inflect,
    /// inflect `number_to_words(n)`: "two thousand and five", "one hundred and five".
    inflect_and,
};

/// Appends the words for `digits` (ASCII digits; leading zeros are ignored).
pub fn append(a: std.mem.Allocator, out: *std.ArrayList(u8), digits: []const u8, style: Style) !void {
    var d = digits;
    while (d.len > 1 and d[0] == '0') d = d[1..];
    if (style == .plain and d.len > 15) {
        try append(a, out, d[0 .. d.len - 12], style);
        try out.appendSlice(a, " trillion");
        const rest = std.mem.trimStart(u8, d[d.len - 12 ..], "0");
        if (rest.len > 0) {
            try out.append(a, ' ');
            try append(a, out, rest, style);
        }
        return;
    }
    if (d.len > 3 * SCALES.len) {
        for (d, 0..) |c, i| {
            if (i > 0) try out.append(a, ' ');
            try out.appendSlice(a, ONES[c - '0']);
        }
        return;
    }
    if (d.len == 1 and d[0] == '0') return out.appendSlice(a, "zero");
    var groups: [SCALES.len]u32 = @splat(0);
    var ng: usize = 0;
    var end = d.len;
    while (end > 0) : (ng += 1) {
        const start = if (end >= 3) end - 3 else 0;
        groups[ng] = std.fmt.parseInt(u32, d[start..end], 10) catch 0;
        end = start;
    }
    var first = true;
    var gi = ng;
    while (gi > 0) {
        gi -= 1;
        const v = groups[gi];
        if (v == 0) continue;
        // inflect joins the last group under 100 with "and" (or a space), the others with ", ".
        if (!first) try out.appendSlice(a, if (style == .plain) " " else if (gi == 0 and v < 100) andSep(style) else ", ");
        try under1000(a, out, v, style);
        if (gi > 0) {
            try out.append(a, ' ');
            try out.appendSlice(a, SCALES[gi]);
        }
        first = false;
    }
}

fn andSep(style: Style) []const u8 {
    return if (style == .inflect_and) " and " else " ";
}

fn under100(a: std.mem.Allocator, out: *std.ArrayList(u8), n: u32, style: Style) !void {
    if (n < 20) return out.appendSlice(a, ONES[n]);
    try out.appendSlice(a, TENS[n / 10]);
    if (n % 10 != 0) {
        try out.append(a, if (style == .plain) ' ' else '-');
        try out.appendSlice(a, ONES[n % 10]);
    }
}

fn under1000(a: std.mem.Allocator, out: *std.ArrayList(u8), n: u32, style: Style) !void {
    if (n >= 100) {
        try out.appendSlice(a, ONES[n / 100]);
        try out.appendSlice(a, " hundred");
        if (n % 100 != 0) {
            try out.appendSlice(a, andSep(style));
            try under100(a, out, n % 100, style);
        }
    } else try under100(a, out, n, style);
}

test "number_words: plain, inflect and inflect with and" {
    const a = std.testing.allocator;
    const cases = [_]struct { []const u8, Style, []const u8 }{
        .{ "1234", .inflect, "one thousand, two hundred thirty-four" },
        .{ "2024", .inflect_and, "two thousand and twenty-four" },
        .{ "1001000", .inflect_and, "one million, one thousand" },
        .{ "123456789012", .inflect_and, "one hundred and twenty-three billion, four hundred and fifty-six million, seven hundred and eighty-nine thousand and twelve" },
        .{ "0", .inflect_and, "zero" },
        .{ "1234", .plain, "one thousand two hundred thirty four" },
        .{ "1000005", .plain, "one million five" },
        .{ "0", .plain, "zero" },
        .{ "18446744073709551615", .plain, "eighteen million four hundred forty six thousand seven hundred forty four trillion seventy three billion seven hundred nine million five hundred fifty one thousand six hundred fifteen" },
        .{ "1000000000000000", .plain, "one thousand trillion" },
    };
    for (cases) |c| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(a);
        try append(a, &out, c[0], c[1]);
        try std.testing.expectEqualStrings(c[2], out.items);
    }
}
