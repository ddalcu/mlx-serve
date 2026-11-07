//! French text normalization for MULTILINGUAL with language fr. Upstream
//! passes French text to the model untouched, so digits are left to the
//! model, which reads "60" as "sixante" and "14h30" or "12,50 €" unreliably.
//! This spells numbers out and flattens French typography before the text is
//! cut into fragments. It is a deliberate departure from upstream.
//!
//! Spelling: traditional (pre-1990) rules, because they are what most French
//! text, and so most of the model's training text, uses: hyphens only between
//! tens and units below 100 ("vingt-deux", "soixante et onze", "deux mille
//! vingt-quatre"), "cents"/"quatre-vingts" plural only when final or before
//! million/milliard, "mille" invariable. France forms only (soixante-dix,
//! quatre-vingts, quatre-vingt-dix), not septante/huitante/nonante.
//!
//! Handled: cardinals up to 999 999 999 999, thousands separators (space,
//! NBSP, narrow NBSP, dot) in groups of three, negative numbers, decimals with
//! a comma ("3,5" → "trois virgule cinq"), ordinals (1er, 1re, 1ère, 2e, 2ème,
//! 2nd, Roman "XXe"), percentages, euros ("12,50 €" → "douze euros
//! cinquante"), times ("14h30" → "quatorze heures trente"). Typography: NBSP
//! and narrow NBSP → space, « » ‹ › and “ ” → removed, ’ → ', … → "...",
//! em/en dash between spaces → comma.

const std = @import("std");

const units = [_][]const u8{ "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize" };
const tens = [_][]const u8{ "", "dix", "vingt", "trente", "quarante", "cinquante", "soixante" };

/// 0..99. `final`: nothing follows inside the number (for "quatre-vingts").
fn below100(out: *std.ArrayList(u8), a: std.mem.Allocator, n: u64, final: bool) !void {
    if (n <= 16) return out.appendSlice(a, units[n]);
    if (n < 20) return out.print(a, "dix-{s}", .{units[n - 10]});
    if (n < 70) {
        const t = n / 10;
        const u = n % 10;
        try out.appendSlice(a, tens[t]);
        if (u == 1) try out.appendSlice(a, " et un") else if (u > 0) try out.print(a, "-{s}", .{units[u]});
        return;
    }
    if (n < 80) {
        // 70–79: soixante-dix, soixante et onze, soixante-douze …
        try out.appendSlice(a, "soixante");
        if (n == 71) return out.appendSlice(a, " et onze");
        try out.append(a, '-');
        return below100(out, a, n - 60, final);
    }
    // 80–99: quatre-vingts, quatre-vingt-un, quatre-vingt-dix, quatre-vingt-onze …
    if (n == 80) return out.appendSlice(a, if (final) "quatre-vingts" else "quatre-vingt");
    try out.appendSlice(a, "quatre-vingt-");
    return below100(out, a, n - 80, final);
}

/// 0..999. `final` as above; `before_noun`: million/milliard follows (cents
/// and quatre-vingts keep their s there, unlike before mille).
fn below1000(out: *std.ArrayList(u8), a: std.mem.Allocator, n: u64, final: bool) !void {
    const h = n / 100;
    const r = n % 100;
    if (h > 0) {
        if (h > 1) {
            try below100(out, a, h, false);
            try out.append(a, ' ');
        }
        try out.appendSlice(a, if (h > 1 and r == 0 and final) "cents" else "cent");
        if (r == 0) return;
        try out.append(a, ' ');
    }
    try below100(out, a, r, final);
}

/// Cardinal, traditional spelling. `final`: the word ends the number phrase.
pub fn cardinal(out: *std.ArrayList(u8), a: std.mem.Allocator, n: u64) !void {
    if (n == 0) return out.appendSlice(a, "zéro");
    const scales = [_]struct { v: u64, one: []const u8, many: []const u8, noun: bool }{
        .{ .v = 1_000_000_000, .one = "un milliard", .many = "milliards", .noun = true },
        .{ .v = 1_000_000, .one = "un million", .many = "millions", .noun = true },
        .{ .v = 1_000, .one = "mille", .many = "mille", .noun = false },
    };
    var rest = n;
    var first = true;
    for (scales) |sc| {
        const q = rest / sc.v;
        rest %= sc.v;
        if (q == 0) continue;
        if (!first) try out.append(a, ' ');
        first = false;
        if (q == 1) {
            try out.appendSlice(a, sc.one);
        } else {
            // Before million/milliard (nouns) "cents"/"quatre-vingts" stay plural.
            try below1000(out, a, q, sc.noun);
            try out.print(a, " {s}", .{sc.many});
        }
    }
    if (rest > 0) {
        if (!first) try out.append(a, ' ');
        try below1000(out, a, rest, true);
    }
}

/// Ordinal: premier/première, otherwise cardinal + "ième".
pub fn ordinal(out: *std.ArrayList(u8), a: std.mem.Allocator, n: u64, feminine: bool) !void {
    if (n == 1) return out.appendSlice(a, if (feminine) "première" else "premier");
    var c: std.ArrayList(u8) = .empty;
    defer c.deinit(a);
    try cardinal(&c, a, n);
    var w = c.items;
    // quatre-vingts → quatre-vingtième, deux cents → deux centième.
    if (std.mem.endsWith(u8, w, "s") and (std.mem.endsWith(u8, w, "vingts") or std.mem.endsWith(u8, w, "cents"))) w = w[0 .. w.len - 1];
    if (std.mem.endsWith(u8, w, "cinq")) {
        try out.appendSlice(a, w);
        return out.appendSlice(a, "uième");
    }
    if (std.mem.endsWith(u8, w, "neuf")) {
        try out.appendSlice(a, w[0 .. w.len - 1]);
        return out.appendSlice(a, "vième");
    }
    if (std.mem.endsWith(u8, w, "e")) w = w[0 .. w.len - 1]; // quatre, onze, douze, trente, mille …
    try out.appendSlice(a, w);
    try out.appendSlice(a, "ième");
}

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

/// Thousands separators accepted inside a number: space, NBSP, narrow NBSP, dot.
fn sepLen(s: []const u8, i: usize) usize {
    if (i >= s.len) return 0;
    if (s[i] == ' ' or s[i] == '.') return 1;
    if (std.mem.startsWith(u8, s[i..], "\u{00A0}")) return 2;
    if (std.mem.startsWith(u8, s[i..], "\u{202F}")) return 3;
    return 0;
}

/// Digits starting at `i`, with thousands groups ("1 234 567", "1.234").
/// Returns the value and the end index.
fn scanInt(s: []const u8, start: usize) ?struct { v: u64, end: usize } {
    var i = start;
    var v: u64 = 0;
    var nd: usize = 0;
    while (i < s.len and isDigit(s[i])) : (i += 1) {
        v = v *% 10 +% (s[i] - '0');
        nd += 1;
    }
    if (nd == 0 or nd > 12) return null;
    if (nd <= 3) {
        // Thousands groups: separator + exactly three digits not followed by a digit.
        while (true) {
            const sl = sepLen(s, i);
            if (sl == 0) break;
            const g = i + sl;
            if (g + 3 > s.len or !isDigit(s[g]) or !isDigit(s[g + 1]) or !isDigit(s[g + 2])) break;
            if (g + 3 < s.len and isDigit(s[g + 3])) break;
            // A dot followed by three digits after a 1–3 digit group is a separator ("1.234");
            // "3.5" stays a dot (not French decimal) and is left alone below.
            v = v * 1000 + @as(u64, s[g] - '0') * 100 + @as(u64, s[g + 1] - '0') * 10 + (s[g + 2] - '0');
            i = g + 3;
        }
    }
    return .{ .v = v, .end = i };
}

fn startsWithAt(s: []const u8, i: usize, p: []const u8) bool {
    return i <= s.len and std.mem.startsWith(u8, s[i..], p);
}

fn isWordByte(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c >= 0x80;
}

/// Skip one space-like separator (space, NBSP, narrow NBSP) at i.
fn skipSpace(s: []const u8, i: usize) usize {
    if (i < s.len and s[i] == ' ') return i + 1;
    if (startsWithAt(s, i, "\u{00A0}")) return i + 2;
    if (startsWithAt(s, i, "\u{202F}")) return i + 3;
    return i;
}

/// Spell out numbers and flatten typography. Caller frees.
pub fn normalize(a: std.mem.Allocator, s: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < s.len) {
        const c = s[i];
        const prev_word = i > 0 and isWordByte(s[i - 1]);
        // Roman century/ordinal: XXe, XIXe, Ier, IIIe (uppercase, own word).
        if (!prev_word and (c == 'I' or c == 'V' or c == 'X' or c == 'L' or c == 'C')) {
            var j = i;
            while (j < s.len and (s[j] == 'I' or s[j] == 'V' or s[j] == 'X' or s[j] == 'L' or s[j] == 'C')) j += 1;
            if (romanValue(s[i..j])) |v| {
                const suf = ordinalSuffix(s, j);
                // One-letter forms are words too often ("Le", "Ce", "Ve"): only "Ier"/"Ire",
                // or any form followed by "siècle".
                const plausible = j - i >= 2 or (s[i] == 'I' and suf.len >= 2) or startsWithAt(s, skipSpace(s, j + suf.len), "siècle");
                if (plausible and suf.len > 0 and (j + suf.len >= s.len or !isWordByte(s[j + suf.len]))) {
                    try ordinal(&out, a, v, suf.feminine);
                    i = j + suf.len;
                    continue;
                }
            }
        }
        if (isDigit(c) and !prev_word) {
            const neg = out.items.len > 0 and out.items[out.items.len - 1] == '-' and (out.items.len == 1 or out.items[out.items.len - 2] == ' ');
            if (scanInt(s, i)) |n| {
                var j = n.end;
                // Time: 14h30, 14 h 30, 14h.
                const hj = skipSpace(s, j);
                if (hj < s.len and s[hj] == 'h' and n.v < 24 and (hj + 1 >= s.len or !std.ascii.isAlphabetic(s[hj + 1]) or isDigit(s[skipSpace(s, hj + 1)]))) {
                    var k = skipSpace(s, hj + 1);
                    var mins: ?u64 = null;
                    if (k + 1 < s.len and isDigit(s[k]) and isDigit(s[k + 1]) and (k + 2 >= s.len or !isDigit(s[k + 2]))) {
                        mins = (s[k] - '0') * 10 + (s[k + 1] - '0');
                        k += 2;
                    } else k = hj + 1;
                    if (mins == null or mins.? < 60) {
                        try feminineCardinal(&out, a, n.v);
                        try out.appendSlice(a, if (n.v > 1) " heures" else " heure");
                        if (mins) |m| if (m > 0) {
                            try out.append(a, ' ');
                            try feminineCardinal(&out, a, m);
                        };
                        i = k;
                        continue;
                    }
                }
                // Ordinal: 1er, 1re, 1ère, 2e, 2ème, 2nd(e).
                const suf = ordinalSuffix(s, j);
                if (suf.len > 0 and (j + suf.len >= s.len or !isWordByte(s[j + suf.len]))) {
                    try ordinal(&out, a, n.v, suf.feminine);
                    i = j + suf.len;
                    continue;
                }
                // Decimal comma: 3,5 / 12,50.
                var frac: ?[]const u8 = null;
                if (j + 1 < s.len and s[j] == ',' and isDigit(s[j + 1])) {
                    var k = j + 1;
                    while (k < s.len and isDigit(s[k])) k += 1;
                    frac = s[j + 1 .. k];
                    j = k;
                }
                if (neg) out.items.len -= 1; // the '-' we copied
                if (neg) try out.appendSlice(a, "moins ");
                // Euros after the number: "12,50 €", "12 euros".
                const ej = skipSpace(s, j);
                if (startsWithAt(s, ej, "€")) {
                    try euros(&out, a, n.v, frac);
                    i = ej + "€".len;
                    continue;
                }
                // Percent: "25 %", "25%".
                if (startsWithAt(s, ej, "%")) {
                    try cardinal(&out, a, n.v);
                    if (frac) |f| try decimals(&out, a, f);
                    try out.appendSlice(a, " pour cent");
                    i = ej + 1;
                    continue;
                }
                try cardinal(&out, a, n.v);
                if (frac) |f| try decimals(&out, a, f);
                i = j;
                continue;
            }
        }
        // Typography.
        if (startsWithAt(s, i, "\u{00A0}")) {
            try out.append(a, ' ');
            i += 2;
            continue;
        }
        if (startsWithAt(s, i, "\u{202F}") or startsWithAt(s, i, "\u{2009}")) {
            try out.append(a, ' ');
            i += 3;
            continue;
        }
        inline for (.{ "«", "»", "‹", "›", "“", "”", "„" }) |q| {
            if (startsWithAt(s, i, q)) {
                i += q.len;
                // Drop the quote and the space that French typography puts inside it.
                if (i < s.len and s[i] == ' ' and out.items.len > 0 and out.items[out.items.len - 1] == ' ') i += 1;
                break;
            }
        } else {
            if (startsWithAt(s, i, "’") or startsWithAt(s, i, "‘")) {
                try out.append(a, '\'');
                i += 3;
                continue;
            }
            if (startsWithAt(s, i, "…")) {
                try out.appendSlice(a, "...");
                i += 3;
                continue;
            }
            if ((startsWithAt(s, i, "—") or startsWithAt(s, i, "–")) and out.items.len > 0 and out.items[out.items.len - 1] == ' ') {
                out.items[out.items.len - 1] = ',';
                try out.append(a, ' ');
                i += 3;
                if (i < s.len and s[i] == ' ') i += 1;
                continue;
            }
            try out.append(a, c);
            i += 1;
            continue;
        }
    }
    // Collapse the double spaces quote removal can leave, and the space it
    // leaves before a comma or period.
    var w: usize = 0;
    for (out.items) |ch| {
        if (ch == ' ' and w > 0 and out.items[w - 1] == ' ') continue;
        if ((ch == ',' or ch == '.') and w > 0 and out.items[w - 1] == ' ') w -= 1;
        out.items[w] = ch;
        w += 1;
    }
    out.items.len = w;
    return out.toOwnedSlice(a);
}

/// "un" → "une" before a feminine noun (heure): une heure, vingt et une heures.
fn feminineCardinal(out: *std.ArrayList(u8), a: std.mem.Allocator, n: u64) !void {
    const start = out.items.len;
    try cardinal(out, a, n);
    if (std.mem.endsWith(u8, out.items[start..], "un")) try out.append(a, 'e');
}

/// Digits after the decimal comma, read as a number after leading zeros
/// ("3,05" → "trois virgule zéro cinq").
fn decimals(out: *std.ArrayList(u8), a: std.mem.Allocator, f: []const u8) !void {
    try out.appendSlice(a, " virgule");
    var k: usize = 0;
    while (k + 1 < f.len and f[k] == '0') : (k += 1) try out.appendSlice(a, " zéro");
    if (f.len - k > 12) return; // absurdly long: leave it
    try out.append(a, ' ');
    try cardinal(out, a, std.fmt.parseInt(u64, f[k..], 10) catch 0);
}

/// 12,50 € → "douze euros cinquante"; 1 € → "un euro"; 0,50 € → "cinquante centimes".
fn euros(out: *std.ArrayList(u8), a: std.mem.Allocator, whole: u64, frac: ?[]const u8) !void {
    var cents: u64 = 0;
    if (frac) |f| {
        cents = std.fmt.parseInt(u64, f[0..@min(2, f.len)], 10) catch 0;
        if (f.len == 1) cents *= 10;
    }
    if (whole == 0 and cents > 0) {
        try cardinal(out, a, cents);
        return out.appendSlice(a, if (cents > 1) " centimes" else " centime");
    }
    try cardinal(out, a, whole);
    // "un million d'euros"
    if (whole >= 1_000_000 and whole % 1_000_000 == 0) try out.appendSlice(a, " d'euros") else try out.appendSlice(a, if (whole > 1) " euros" else " euro");
    if (cents > 0) {
        try out.append(a, ' ');
        try cardinal(out, a, cents);
    }
}

const Suffix = struct { len: usize, feminine: bool };

fn ordinalSuffix(s: []const u8, j: usize) Suffix {
    const opts = [_]struct { []const u8, bool }{
        .{ "ère", true },
        .{ "ème", false },
        .{ "nde", true },
        .{ "er", false },
        .{ "re", true },
        .{ "nd", false },
        .{ "e", false },
    };
    for (opts) |o| if (startsWithAt(s, j, o[0])) return .{ .len = o[0].len, .feminine = o[1] };
    return .{ .len = 0, .feminine = false };
}

fn romanValue(r: []const u8) ?u64 {
    if (r.len == 0 or r.len > 8) return null;
    var total: u64 = 0;
    var i: usize = 0;
    while (i < r.len) : (i += 1) {
        const v = romanDigit(r[i]);
        const next = if (i + 1 < r.len) romanDigit(r[i + 1]) else 0;
        if (v < next) total -%= v else total += v;
    }
    if (total == 0 or total > 399) return null;
    return total;
}

fn romanDigit(c: u8) u64 {
    return switch (c) {
        'I' => 1,
        'V' => 5,
        'X' => 10,
        'L' => 50,
        'C' => 100,
        else => 0,
    };
}

test "French cardinals, traditional spelling" {
    const a = std.testing.allocator;
    const cases = [_]struct { u64, []const u8 }{
        .{ 0, "zéro" },
        .{ 1, "un" },
        .{ 16, "seize" },
        .{ 17, "dix-sept" },
        .{ 21, "vingt et un" },
        .{ 22, "vingt-deux" },
        .{ 31, "trente et un" },
        .{ 60, "soixante" },
        .{ 61, "soixante et un" },
        .{ 70, "soixante-dix" },
        .{ 71, "soixante et onze" },
        .{ 77, "soixante-dix-sept" },
        .{ 80, "quatre-vingts" },
        .{ 81, "quatre-vingt-un" },
        .{ 90, "quatre-vingt-dix" },
        .{ 91, "quatre-vingt-onze" },
        .{ 99, "quatre-vingt-dix-neuf" },
        .{ 100, "cent" },
        .{ 101, "cent un" },
        .{ 200, "deux cents" },
        .{ 201, "deux cent un" },
        .{ 280, "deux cent quatre-vingts" },
        .{ 1000, "mille" },
        .{ 1001, "mille un" },
        .{ 2000, "deux mille" },
        .{ 2024, "deux mille vingt-quatre" },
        .{ 1990, "mille neuf cent quatre-vingt-dix" },
        .{ 80000, "quatre-vingt mille" },
        .{ 200000, "deux cent mille" },
        .{ 1_000_000, "un million" },
        .{ 2_000_000, "deux millions" },
        .{ 200_000_000, "deux cents millions" },
        .{ 80_000_000, "quatre-vingts millions" },
        .{ 999_999_999, "neuf cent quatre-vingt-dix-neuf millions neuf cent quatre-vingt-dix-neuf mille neuf cent quatre-vingt-dix-neuf" },
        .{ 1_000_000_000, "un milliard" },
    };
    for (cases) |c| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(a);
        try cardinal(&out, a, c[0]);
        try std.testing.expectEqualStrings(c[1], out.items);
    }
}

test "French text normalization" {
    const a = std.testing.allocator;
    const cases = [_][2][]const u8{
        .{ "Il a 60 ans.", "Il a soixante ans." },
        .{ "J'ai 3 pommes et 71 poires.", "J'ai trois pommes et soixante et onze poires." },
        .{ "En 2024, pas en 1999.", "En deux mille vingt-quatre, pas en mille neuf cent quatre-vingt-dix-neuf." },
        .{ "Il fait 3,5 degrés, ou -2.", "Il fait trois virgule cinq degrés, ou moins deux." },
        .{ "Ça coûte 12,50 € et 1 €.", "Ça coûte douze euros cinquante et un euro." },
        .{ "Soit 0,50 €.", "Soit cinquante centimes." },
        .{ "Une baisse de 25 %.", "Une baisse de vingt-cinq pour cent." },
        .{ "Rendez-vous à 14h30 ou 21h, ou 1h05.", "Rendez-vous à quatorze heures trente ou vingt et une heures, ou une heure cinq." },
        .{ "Le 1er mai, la 1re fois, la 2e, le 21e.", "Le premier mai, la première fois, la deuxième, le vingt et unième." },
        .{ "Au XXe siècle et au XIXe.", "Au vingtième siècle et au dix-neuvième." },
        .{ "Le 5e et le 9e.", "Le cinquième et le neuvième." },
        .{ "1 234 567 habitants, 12\u{202F}000 ou 3.000.", "un million deux cent trente-quatre mille cinq cent soixante-sept habitants, douze mille ou trois mille." },
        .{ "Vraiment\u{00A0}? Oui\u{202F}! «\u{00A0}C’est génial\u{00A0}», dit-il…", "Vraiment ? Oui ! C'est génial, dit-il..." },
        .{ "Le chat et Ce soir, Ve République, Louis XIV, le Ier.", "Le chat et Ce soir, Ve République, Louis XIV, le premier." },
        .{ "Le V siècle et le Xe siècle.", "Le V siècle et le dixième siècle." },
        .{ "Attends — on verra.", "Attends, on verra." },
        .{ "M. Dupont et Mme Lefèvre.", "M. Dupont et Mme Lefèvre." },
        .{ "Version v2 et MP3.", "Version v2 et MP3." },
    };
    for (cases) |c| {
        const got = try normalize(a, c[0]);
        defer a.free(got);
        try std.testing.expectEqualStrings(c[1], got);
    }
}
