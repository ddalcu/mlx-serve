//! Pre-tokenizers: isolated runs and the gpt2 / llama3 / cased split grammars.

const std = @import("std");

const PretokStyle = @import("tokenizer.zig").PretokStyle;
const IsolatedRuns = @import("tokenizer.zig").IsolatedRuns;

/// HF applies the isolating Splits first and the main regex to each piece alone, so no
/// pre-token crosses a run's edge: `"x  1"` keeps `"  "`, which Llama-3's combined regex splits.
pub fn isolatedRunsPreTokenize(allocator: std.mem.Allocator, text: []const u8, style: PretokStyle, digit_group: u8, runs: IsolatedRuns, words: *std.ArrayList([]const u8)) !void {
    if (!runs.digits and !runs.cjk) return styledPreTokenize(allocator, text, style, digit_group, words);
    var start: usize = 0;
    var prev: u2 = 0;
    var i: usize = 0;
    while (decodeCodepoint(text, i)) |c| : (i += c.len) {
        const cjk = (c.cp >= 0x4E00 and c.cp <= 0x9FA5) or (c.cp >= 0x3040 and c.cp <= 0x30FF);
        const kind: u2 = if (runs.digits and isDigit(c.cp)) 1 else if (runs.cjk and cjk) 2 else 0;
        if (kind != prev and i > start) {
            try styledPreTokenize(allocator, text[start..i], style, digit_group, words);
            start = i;
        }
        prev = kind;
    }
    if (start < text.len) try styledPreTokenize(allocator, text[start..], style, digit_group, words);
}

fn styledPreTokenize(allocator: std.mem.Allocator, text: []const u8, style: PretokStyle, digit_group: u8, words: *std.ArrayList([]const u8)) !void {
    return switch (style) {
        .gpt2 => gpt2PreTokenize(allocator, text, digit_group, words),
        .llama3 => llama3PreTokenize(allocator, text, false, words),
        .cased => llama3PreTokenize(allocator, text, true, words),
    };
}

/// GPT-2 pre-tokenization: splits text following the Qwen / Llama-3 / GPT-2
/// pre-tokenizer regex as a hand-rolled state machine. Each iteration picks
/// the FIRST matching pattern from the alternation, in declared order.
///
/// Reference (from `tokenizer.json` `pre_tokenizer.pretokenizers[0].pattern`):
///
///     (?i:'s|'t|'re|'ve|'m|'ll|'d)
///   | [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
///   | \p{N}
///   |  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
///   | \s*[\r\n]+
///   | \s+(?!\S)
///   | \s+
///
/// Critical priority rules vs. naïve "consume whitespace, then letters":
///   1. ` letter+` is one pre-token (pattern 2 with optional leading non-LN char).
///   2. ` punct+` is one pre-token (pattern 4 with optional leading space).
///   3. Multi-space `    word`: pattern 6 `\s+(?!\S)` matches all-but-last
///      whitespace (`   `), then pattern 2 picks up the last space + letters
///      as one combined ` word` pre-token. Getting this wrong adds extra
///      whitespace pre-tokens that the BPE stage cannot merge across, and
///      causes the model to see a perturbed prior on every subsequent word.
///   4. Digits are SINGLE-codepoint pre-tokens (pattern 3 = `\p{N}`, not
///      `\p{N}+`). `100` → three separate `1`, `0`, `0` pre-tokens.
/// `digit_group`: how many consecutive digits form one pre-token. 1 = the
/// Qwen/GPT-2 `\p{N}` rule; 3 = the DeepSeek-V4/Llama-3 `\p{N}{1,3}` rule
/// (greedy left-to-right groups). Parsed per model from tokenizer.json —
/// feeding a {1,3}-trained model per-digit pre-tokens puts every number
/// off-distribution (the echo-precision slip class: `1o` for `10`, split
/// digits, o-for-0 near-ties).
pub fn gpt2PreTokenize(allocator: std.mem.Allocator, text: []const u8, digit_group: u8, words: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    while (i < text.len) {
        const start = i;

        // ── Pattern 1: contraction `(?i:'s|'t|'re|'ve|'m|'ll|'d)` ──
        if (text[i] == '\'' and i + 1 < text.len) {
            const next = std.ascii.toLower(text[i + 1]);
            if (next == 's' or next == 't' or next == 'm' or next == 'd') {
                i += 2;
                try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                continue;
            }
            if (i + 2 < text.len) {
                const next2 = std.ascii.toLower(text[i + 2]);
                if ((next == 'r' and next2 == 'e') or
                    (next == 'v' and next2 == 'e') or
                    (next == 'l' and next2 == 'l'))
                {
                    i += 3;
                    try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                    continue;
                }
            }
        }

        // ── Pattern 2: `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+` ──
        // Optional 1 char that's NOT \r, NOT \n, NOT letter, NOT digit (so it
        // CAN be whitespace or punct), followed by 1+ letters/marks.
        if (matchOptionalNonLnnAndLetters(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Pattern 3: `\p{N}` or `\p{N}{1,3}` — up to digit_group digits ──
        if (decodeCodepoint(text, i)) |cp_info| {
            if (isDigit(cp_info.cp)) {
                i += cp_info.len;
                var taken: u8 = 1;
                while (taken < digit_group) : (taken += 1) {
                    const next = decodeCodepoint(text, i) orelse break;
                    if (!isDigit(next.cp)) break;
                    i += next.len;
                }
                try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                continue;
            }
        }

        // ── Pattern 4: ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*` ──
        // Optional 1 space + 1+ chars that are NOT whitespace, NOT letter,
        // NOT mark, NOT digit (i.e. punctuation/symbols), then optional \r\n.
        if (matchOptionalSpaceAndPunct(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Pattern 5: `\s*[\r\n]+` — whitespace ending in newline run ──
        if (matchWhitespaceWithNewline(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Pattern 6: `\s+(?!\S)` — whitespace not followed by non-ws ──
        // Greedy match with backtrack: shortens by 1 if the next char is \S
        // so the trailing space gets handed to pattern 2/4 on the next pass.
        if (matchTrailingWhitespace(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Pattern 7: `\s+` — fallback whitespace ──
        if (matchWhitespaceRun(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // Fallback: single byte (unreachable in well-formed UTF-8 input).
        i += 1;
        try words.append(allocator, try allocator.dupe(u8, text[start..i]));
    }
}

/// Llama-3-style pre-tokenizer (Muse-Glimmer). Branch order mirrors the
/// tokenizer.json Split regex exactly:
///   1. [^\r\n\p{L}\p{N}]?[UPPER]*[LOWER]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
///   2. [^\r\n\p{L}\p{N}]?[UPPER]+[LOWER]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
///   3. \p{N}{1,3}
///   4.  ?[^\s\p{L}\p{N}]+[\r\n/]*
///   5. \s*[\r\n]+   6. \s+(?!\S)   7. \s+
/// UPPER = \p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}, LOWER = \p{Ll}\p{Lm}\p{Lo}\p{M} —
/// caseless letters and marks belong to BOTH classes. Branches 1+2 collapse
/// into one deterministic scan (greedy upper-run then greedy lower-run, ≥1
/// letter total): for disjoint ASCII classes this reproduces the regex's
/// backtracking exactly, and for the overlap classes every reachable match
/// END coincides. Known gap: non-ASCII CASED letters (Cyrillic/Greek Lu/Ll)
/// are treated caseless, so a mid-word case transition there doesn't split —
/// same text, off-reference boundary; revisit with real Lu/Ll tables if a
/// checkpoint's traffic warrants it.
/// `strict_punct` (`.cased`): branch 4 is ` ?[\p{P}\p{S}]+[\r\n]*`, and a `\p{C}` run no branch takes
/// is one piece, as HF keeps an unmatched span.
pub fn llama3PreTokenize(allocator: std.mem.Allocator, text: []const u8, strict_punct: bool, words: *std.ArrayList([]const u8)) !void {
    var i: usize = 0;
    var gap_end: ?usize = null;
    while (i < text.len) {
        const start = i;

        // ── Branches 1+2: optional non-LNN char + cased word + contraction ──
        if (llama3MatchWord(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Branch 3: `\p{N}{1,3}` ──
        if (decodeCodepoint(text, i)) |cp_info| {
            if (isDigit(cp_info.cp)) {
                i += cp_info.len;
                var taken: u8 = 1;
                while (taken < 3) : (taken += 1) {
                    const next = decodeCodepoint(text, i) orelse break;
                    if (!isDigit(next.cp)) break;
                    i += next.len;
                }
                try words.append(allocator, try allocator.dupe(u8, text[start..i]));
                continue;
            }
        }

        // ── Branch 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*` (marks ride with punct,
        // and `/` joins the newline tail) ──
        if (llama3MatchPunct(text, i, strict_punct)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        // ── Branches 5-7: identical whitespace grammar to gpt2 ──
        if (matchWhitespaceWithNewline(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }
        if (matchTrailingWhitespace(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }
        if (matchWhitespaceRun(text, i)) |new_i| {
            i = new_i;
            try words.append(allocator, try allocator.dupe(u8, text[start..i]));
            continue;
        }

        if (strict_punct) {
            // HF keeps the run of positions no branch matches as one piece: extend the last one.
            i += if (decodeCodepoint(text, i)) |c| c.len else 1;
            var span_start = start;
            if (gap_end == start) {
                const prev = words.pop().?;
                span_start = start - prev.len;
                allocator.free(prev);
            }
            gap_end = i;
            try words.append(allocator, try allocator.dupe(u8, text[span_start..i]));
            continue;
        }
        i += 1;
        try words.append(allocator, try allocator.dupe(u8, text[start..i]));
    }
}

/// `[^\s\p{L}\p{N}]`, or `[\p{P}\p{S}]` when strict (which leaves out marks and `\p{C}`).
fn isPunctClass(cp: u21, strict: bool) bool {
    if (isWhitespaceCp(cp) or isLetter(cp) or isDigit(cp)) return false;
    return !strict or !(isMark(cp) or isOtherCp(cp));
}

/// Approximate `\p{C}`: controls, the common format characters, private use.
fn isOtherCp(cp: u21) bool {
    if (cp < 0x20 or (cp >= 0x7F and cp <= 0x9F)) return true;
    return switch (cp) {
        0xAD, 0x061C, 0x06DD, 0x070F, 0x180E, 0xFEFF => true,
        0x0600...0x0605, 0x200B...0x200F, 0x202A...0x202E, 0x2060...0x2064, 0x2066...0x206F, 0xFFF9...0xFFFB => true,
        0xE000...0xF8FF, 0xF0000...0x10FFFF => true,
        else => false,
    };
}

/// UPPER class: \p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M} — ASCII exact, non-ASCII
/// letters/marks caseless (in both classes; see llama3PreTokenize doc).
fn llama3UpperClass(cp: u21) bool {
    if (cp < 128) return cp >= 'A' and cp <= 'Z';
    return isLetterOrMark(cp);
}

/// LOWER class: \p{Ll}\p{Lm}\p{Lo}\p{M}.
fn llama3LowerClass(cp: u21) bool {
    if (cp < 128) return cp >= 'a' and cp <= 'z';
    return isLetterOrMark(cp);
}

/// Branches 1+2 of the llama3 grammar. Returns the match end or null.
fn llama3MatchWord(text: []const u8, start: usize) ?usize {
    var i = start;
    const first = decodeCodepoint(text, i) orelse return null;
    // Optional single char that's NOT \r/\n/letter/digit (space and marks OK).
    if (!isLetter(first.cp) and !isDigit(first.cp) and first.cp != '\r' and first.cp != '\n') {
        // Consume it only if letters actually follow (regex optionality).
        const after = i + first.len;
        const next = decodeCodepoint(text, after) orelse return null;
        if (!llama3UpperClass(next.cp) and !llama3LowerClass(next.cp)) return null;
        i = after;
    }
    const word_start = i;
    // Greedy upper-run, then greedy lower-run; ≥1 letter total.
    while (decodeCodepoint(text, i)) |c| {
        if (!llama3UpperClass(c.cp)) break;
        i += c.len;
    }
    while (decodeCodepoint(text, i)) |c| {
        if (!llama3LowerClass(c.cp)) break;
        i += c.len;
    }
    if (i == word_start) return null;
    // Optional (?i) contraction suffix.
    if (i < text.len and text[i] == '\'' and i + 1 < text.len) {
        const n1 = std.ascii.toLower(text[i + 1]);
        if (n1 == 's' or n1 == 't' or n1 == 'm' or n1 == 'd') {
            i += 2;
        } else if (i + 2 < text.len) {
            const n2 = std.ascii.toLower(text[i + 2]);
            if ((n1 == 'r' and n2 == 'e') or (n1 == 'v' and n2 == 'e') or (n1 == 'l' and n2 == 'l')) {
                i += 3;
            }
        }
    }
    return i;
}

/// Branch 4: ` ?[^\s\p{L}\p{N}]+[\r\n/]*`. Unlike the gpt2 pattern, marks
/// belong to the punct class and `/` joins the trailing run.
fn llama3MatchPunct(text: []const u8, start: usize, strict: bool) ?usize {
    if (start >= text.len) return null;
    var p_start: usize = start;
    if (text[start] == ' ') p_start = start + 1;

    if (p_start >= text.len) return null;
    const first_cp = decodeCodepoint(text, p_start) orelse return null;
    if (!isPunctClass(first_cp.cp, strict)) return null;

    var i: usize = p_start + first_cp.len;
    while (i < text.len) {
        const c = decodeCodepoint(text, i) orelse break;
        if (!isPunctClass(c.cp, strict)) break;
        i += c.len;
    }
    while (i < text.len and (text[i] == '\r' or text[i] == '\n' or (!strict and text[i] == '/'))) i += 1;
    return i;
}

/// Pattern 2: `[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+`. Returns end position of
/// match, or null if no letters at the right place.
fn matchOptionalNonLnnAndLetters(text: []const u8, start: usize) ?usize {
    if (start >= text.len) return null;
    const cp_start = decodeCodepoint(text, start) orelse return null;

    // Try with the optional non-LNN char consumed.
    if (!isLetter(cp_start.cp) and !isDigit(cp_start.cp) and
        cp_start.cp != '\r' and cp_start.cp != '\n')
    {
        const after_opt = start + cp_start.len;
        if (after_opt < text.len) {
            const next_cp = decodeCodepoint(text, after_opt);
            if (next_cp != null and isLetterOrMark(next_cp.?.cp)) {
                var i: usize = after_opt + next_cp.?.len;
                while (i < text.len) {
                    const c = decodeCodepoint(text, i) orelse break;
                    if (!isLetterOrMark(c.cp)) break;
                    i += c.len;
                }
                return i;
            }
        }
    }

    // Try with 0-length optional: text[start] must itself be a letter/mark.
    if (isLetterOrMark(cp_start.cp)) {
        var i: usize = start + cp_start.len;
        while (i < text.len) {
            const c = decodeCodepoint(text, i) orelse break;
            if (!isLetterOrMark(c.cp)) break;
            i += c.len;
        }
        return i;
    }

    return null;
}

/// Pattern 4: ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`. Optional ASCII space then
/// 1+ punct/symbol codepoints, then optional \r\n run. Returns end position
/// or null. The optional space MUST be exactly the byte ' ' (0x20), not
/// any other whitespace — matches Qwen's tokenizer.json regex literal.
fn matchOptionalSpaceAndPunct(text: []const u8, start: usize) ?usize {
    if (start >= text.len) return null;
    var p_start: usize = start;
    if (text[start] == ' ') p_start = start + 1;

    if (p_start >= text.len) return null;
    const first_cp = decodeCodepoint(text, p_start) orelse return null;
    // Must be NOT whitespace, NOT letter, NOT mark, NOT digit.
    if (isWhitespaceCp(first_cp.cp) or isLetter(first_cp.cp) or
        isMark(first_cp.cp) or isDigit(first_cp.cp)) return null;

    var i: usize = p_start + first_cp.len;
    while (i < text.len) {
        const c = decodeCodepoint(text, i) orelse break;
        if (isWhitespaceCp(c.cp) or isLetter(c.cp) or isMark(c.cp) or isDigit(c.cp)) break;
        i += c.len;
    }
    // Optional trailing \r\n.
    while (i < text.len and (text[i] == '\r' or text[i] == '\n')) i += 1;
    return i;
}

/// Pattern 5: `\s*[\r\n]+`. The greedy `\s*` backtracks to the whitespace
/// run's LAST \r or \n, so the match ends right after it ("\n    \n" is one
/// pre-token). Returns null if the run holds no \r or \n.
fn matchWhitespaceWithNewline(text: []const u8, start: usize) ?usize {
    var i: usize = start;
    var end: ?usize = null;
    while (whitespaceLen(text, i)) |n| {
        if (text[i] == '\r' or text[i] == '\n') end = i + 1;
        i += n;
    }
    return end;
}

/// Pattern 6: `\s+(?!\S)`. Greedy match of all whitespace codepoints, then
/// shortens by one if the next char is \S so the trailing space can be picked
/// up by pattern 2/4 on the next iteration. Returns null if there's only
/// one whitespace char and the next is \S (lookahead can't be satisfied).
fn matchTrailingWhitespace(text: []const u8, start: usize) ?usize {
    var end: usize = start;
    var last: usize = start;
    while (whitespaceLen(text, end)) |n| {
        last = end;
        end += n;
    }
    if (end == start) return null;
    // text[start..end] is the maximal whitespace run starting at start.
    if (end == text.len) return end; // end of input — lookahead trivially OK
    // text[end] is non-whitespace (\S). Backtrack one whitespace char so
    // the position-after-match lands on a whitespace char (lookahead OK).
    if (last > start) return last;
    return null;
}

/// Pattern 7: `\s+`. Returns the end of the whitespace run, or null.
fn matchWhitespaceRun(text: []const u8, start: usize) ?usize {
    var end: usize = start;
    while (whitespaceLen(text, end)) |n| end += n;
    return if (end > start) end else null;
}

/// `\s` in HF `tokenizers`' regex engine (Oniguruma) is Unicode White_Space:
/// the full-width U+3000 and NBSP U+00A0 are whitespace, not punctuation.
pub fn isWhitespaceCp(cp: u21) bool {
    return switch (cp) {
        0x09...0x0D, 0x20, 0x85, 0xA0, 0x1680, 0x2000...0x200A, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000 => true,
        else => false,
    };
}

/// Byte length of the `\s` codepoint at `pos`, or null.
fn whitespaceLen(text: []const u8, pos: usize) ?usize {
    const c = decodeCodepoint(text, pos) orelse return null;
    return if (isWhitespaceCp(c.cp)) c.len else null;
}

fn isLetterOrMark(cp: u21) bool {
    return isLetter(cp) or isMark(cp);
}

/// `\p{M}` (Mn, Mc, Me).
fn isMark(cp: u21) bool {
    return cp >= 0x300 and inRanges(&MARK_RANGES, cp);
}

const CpInfo = struct { cp: u21, len: usize };

fn decodeCodepoint(text: []const u8, pos: usize) ?CpInfo {
    if (pos >= text.len) return null;
    const byte_len = std.unicode.utf8ByteSequenceLength(text[pos]) catch return CpInfo{ .cp = text[pos], .len = 1 };
    const end = @min(pos + byte_len, text.len);
    const cp = std.unicode.utf8Decode(text[pos..end]) catch return CpInfo{ .cp = text[pos], .len = 1 };
    return CpInfo{ .cp = cp, .len = end - pos };
}

/// CJK Unified Ideographs, U+4E00..U+9FFF: the `'\u4e00' <= c <= '\u9fff'` test of most text frontends.
pub fn isCjk(cp: u21) bool {
    return cp >= 0x4E00 and cp <= 0x9FFF;
}

/// `isCjk` plus Extension A, Extensions B-F and the compatibility ideographs: Qwen3-ASR's `is_cjk_char`.
pub fn isCjkExtended(cp: u21) bool {
    return switch (cp) {
        0x4E00...0x9FFF, 0x3400...0x4DBF, 0x20000...0x2A6DF, 0x2A700...0x2B73F, 0x2B740...0x2B81F, 0x2B820...0x2CEAF, 0xF900...0xFAFF => true,
        else => false,
    };
}

/// `\p{L}` (Lu, Ll, Lt, Lm, Lo).
pub fn isLetter(cp: u21) bool {
    if (cp < 0x80) return (cp >= 'A' and cp <= 'Z') or (cp >= 'a' and cp <= 'z');
    return inRanges(&LETTER_RANGES, cp);
}

/// `\p{N}` (Nd, Nl, No) as inclusive ranges, Unicode 15.1.0:
/// `[(a, b) for runs of unicodedata.category(c)[0] == 'N']`.
const NUMBER_RANGES = [_][2]u21{
    .{ 0x30, 0x39 }, .{ 0xB2, 0xB3 }, .{ 0xB9, 0xB9 }, .{ 0xBC, 0xBE }, .{ 0x660, 0x669 },
    .{ 0x6F0, 0x6F9 }, .{ 0x7C0, 0x7C9 }, .{ 0x966, 0x96F }, .{ 0x9E6, 0x9EF }, .{ 0x9F4, 0x9F9 },
    .{ 0xA66, 0xA6F }, .{ 0xAE6, 0xAEF }, .{ 0xB66, 0xB6F }, .{ 0xB72, 0xB77 }, .{ 0xBE6, 0xBF2 },
    .{ 0xC66, 0xC6F }, .{ 0xC78, 0xC7E }, .{ 0xCE6, 0xCEF }, .{ 0xD58, 0xD5E }, .{ 0xD66, 0xD78 },
    .{ 0xDE6, 0xDEF }, .{ 0xE50, 0xE59 }, .{ 0xED0, 0xED9 }, .{ 0xF20, 0xF33 }, .{ 0x1040, 0x1049 },
    .{ 0x1090, 0x1099 }, .{ 0x1369, 0x137C }, .{ 0x16EE, 0x16F0 }, .{ 0x17E0, 0x17E9 },
    .{ 0x17F0, 0x17F9 }, .{ 0x1810, 0x1819 }, .{ 0x1946, 0x194F }, .{ 0x19D0, 0x19DA },
    .{ 0x1A80, 0x1A89 }, .{ 0x1A90, 0x1A99 }, .{ 0x1B50, 0x1B59 }, .{ 0x1BB0, 0x1BB9 },
    .{ 0x1C40, 0x1C49 }, .{ 0x1C50, 0x1C59 }, .{ 0x2070, 0x2070 }, .{ 0x2074, 0x2079 },
    .{ 0x2080, 0x2089 }, .{ 0x2150, 0x2182 }, .{ 0x2185, 0x2189 }, .{ 0x2460, 0x249B },
    .{ 0x24EA, 0x24FF }, .{ 0x2776, 0x2793 }, .{ 0x2CFD, 0x2CFD }, .{ 0x3007, 0x3007 },
    .{ 0x3021, 0x3029 }, .{ 0x3038, 0x303A }, .{ 0x3192, 0x3195 }, .{ 0x3220, 0x3229 },
    .{ 0x3248, 0x324F }, .{ 0x3251, 0x325F }, .{ 0x3280, 0x3289 }, .{ 0x32B1, 0x32BF },
    .{ 0xA620, 0xA629 }, .{ 0xA6E6, 0xA6EF }, .{ 0xA830, 0xA835 }, .{ 0xA8D0, 0xA8D9 },
    .{ 0xA900, 0xA909 }, .{ 0xA9D0, 0xA9D9 }, .{ 0xA9F0, 0xA9F9 }, .{ 0xAA50, 0xAA59 },
    .{ 0xABF0, 0xABF9 }, .{ 0xFF10, 0xFF19 }, .{ 0x10107, 0x10133 }, .{ 0x10140, 0x10178 },
    .{ 0x1018A, 0x1018B }, .{ 0x102E1, 0x102FB }, .{ 0x10320, 0x10323 }, .{ 0x10341, 0x10341 },
    .{ 0x1034A, 0x1034A }, .{ 0x103D1, 0x103D5 }, .{ 0x104A0, 0x104A9 }, .{ 0x10858, 0x1085F },
    .{ 0x10879, 0x1087F }, .{ 0x108A7, 0x108AF }, .{ 0x108FB, 0x108FF }, .{ 0x10916, 0x1091B },
    .{ 0x109BC, 0x109BD }, .{ 0x109C0, 0x109CF }, .{ 0x109D2, 0x109FF }, .{ 0x10A40, 0x10A48 },
    .{ 0x10A7D, 0x10A7E }, .{ 0x10A9D, 0x10A9F }, .{ 0x10AEB, 0x10AEF }, .{ 0x10B58, 0x10B5F },
    .{ 0x10B78, 0x10B7F }, .{ 0x10BA9, 0x10BAF }, .{ 0x10CFA, 0x10CFF }, .{ 0x10D30, 0x10D39 },
    .{ 0x10E60, 0x10E7E }, .{ 0x10F1D, 0x10F26 }, .{ 0x10F51, 0x10F54 }, .{ 0x10FC5, 0x10FCB },
    .{ 0x11052, 0x1106F }, .{ 0x110F0, 0x110F9 }, .{ 0x11136, 0x1113F }, .{ 0x111D0, 0x111D9 },
    .{ 0x111E1, 0x111F4 }, .{ 0x112F0, 0x112F9 }, .{ 0x11450, 0x11459 }, .{ 0x114D0, 0x114D9 },
    .{ 0x11650, 0x11659 }, .{ 0x116C0, 0x116C9 }, .{ 0x11730, 0x1173B }, .{ 0x118E0, 0x118F2 },
    .{ 0x11950, 0x11959 }, .{ 0x11C50, 0x11C6C }, .{ 0x11D50, 0x11D59 }, .{ 0x11DA0, 0x11DA9 },
    .{ 0x11F50, 0x11F59 }, .{ 0x11FC0, 0x11FD4 }, .{ 0x12400, 0x1246E }, .{ 0x16A60, 0x16A69 },
    .{ 0x16AC0, 0x16AC9 }, .{ 0x16B50, 0x16B59 }, .{ 0x16B5B, 0x16B61 }, .{ 0x16E80, 0x16E96 },
    .{ 0x1D2C0, 0x1D2D3 }, .{ 0x1D2E0, 0x1D2F3 }, .{ 0x1D360, 0x1D378 }, .{ 0x1D7CE, 0x1D7FF },
    .{ 0x1E140, 0x1E149 }, .{ 0x1E2F0, 0x1E2F9 }, .{ 0x1E4F0, 0x1E4F9 }, .{ 0x1E8C7, 0x1E8CF },
    .{ 0x1E950, 0x1E959 }, .{ 0x1EC71, 0x1ECAB }, .{ 0x1ECAD, 0x1ECAF }, .{ 0x1ECB1, 0x1ECB4 },
    .{ 0x1ED01, 0x1ED2D }, .{ 0x1ED2F, 0x1ED3D }, .{ 0x1F100, 0x1F10C }, .{ 0x1FBF0, 0x1FBF9 },
};

/// `\p{N}`: every script's digits, superscripts, fractions and numerals, not just ASCII.
pub fn isDigit(cp: u21) bool {
    if (cp < 0x80) return cp >= '0' and cp <= '9';
    return inRanges(&NUMBER_RANGES, cp);
}

fn inRanges(ranges: []const [2]u21, cp: u21) bool {
    var lo: usize = 0;
    var hi: usize = ranges.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (cp < ranges[mid][0]) hi = mid else if (cp > ranges[mid][1]) lo = mid + 1 else return true;
    }
    return false;
}

/// `\p{L}` as inclusive ranges, Unicode 15.1.0: `[(a, b) for runs of unicodedata.category(c)[0] == 'L']`.
const LETTER_RANGES = [_][2]u21{
    .{ 0x41, 0x5A }, .{ 0x61, 0x7A }, .{ 0xAA, 0xAA }, .{ 0xB5, 0xB5 }, .{ 0xBA, 0xBA }, .{ 0xC0, 0xD6 },
    .{ 0xD8, 0xF6 }, .{ 0xF8, 0x2C1 }, .{ 0x2C6, 0x2D1 }, .{ 0x2E0, 0x2E4 }, .{ 0x2EC, 0x2EC }, .{ 0x2EE, 0x2EE },
    .{ 0x370, 0x374 }, .{ 0x376, 0x377 }, .{ 0x37A, 0x37D }, .{ 0x37F, 0x37F }, .{ 0x386, 0x386 }, .{ 0x388, 0x38A },
    .{ 0x38C, 0x38C }, .{ 0x38E, 0x3A1 }, .{ 0x3A3, 0x3F5 }, .{ 0x3F7, 0x481 }, .{ 0x48A, 0x52F }, .{ 0x531, 0x556 },
    .{ 0x559, 0x559 }, .{ 0x560, 0x588 }, .{ 0x5D0, 0x5EA }, .{ 0x5EF, 0x5F2 }, .{ 0x620, 0x64A }, .{ 0x66E, 0x66F },
    .{ 0x671, 0x6D3 }, .{ 0x6D5, 0x6D5 }, .{ 0x6E5, 0x6E6 }, .{ 0x6EE, 0x6EF }, .{ 0x6FA, 0x6FC }, .{ 0x6FF, 0x6FF },
    .{ 0x710, 0x710 }, .{ 0x712, 0x72F }, .{ 0x74D, 0x7A5 }, .{ 0x7B1, 0x7B1 }, .{ 0x7CA, 0x7EA }, .{ 0x7F4, 0x7F5 },
    .{ 0x7FA, 0x7FA }, .{ 0x800, 0x815 }, .{ 0x81A, 0x81A }, .{ 0x824, 0x824 }, .{ 0x828, 0x828 }, .{ 0x840, 0x858 },
    .{ 0x860, 0x86A }, .{ 0x870, 0x887 }, .{ 0x889, 0x88E }, .{ 0x8A0, 0x8C9 }, .{ 0x904, 0x939 }, .{ 0x93D, 0x93D },
    .{ 0x950, 0x950 }, .{ 0x958, 0x961 }, .{ 0x971, 0x980 }, .{ 0x985, 0x98C }, .{ 0x98F, 0x990 }, .{ 0x993, 0x9A8 },
    .{ 0x9AA, 0x9B0 }, .{ 0x9B2, 0x9B2 }, .{ 0x9B6, 0x9B9 }, .{ 0x9BD, 0x9BD }, .{ 0x9CE, 0x9CE }, .{ 0x9DC, 0x9DD },
    .{ 0x9DF, 0x9E1 }, .{ 0x9F0, 0x9F1 }, .{ 0x9FC, 0x9FC }, .{ 0xA05, 0xA0A }, .{ 0xA0F, 0xA10 }, .{ 0xA13, 0xA28 },
    .{ 0xA2A, 0xA30 }, .{ 0xA32, 0xA33 }, .{ 0xA35, 0xA36 }, .{ 0xA38, 0xA39 }, .{ 0xA59, 0xA5C }, .{ 0xA5E, 0xA5E },
    .{ 0xA72, 0xA74 }, .{ 0xA85, 0xA8D }, .{ 0xA8F, 0xA91 }, .{ 0xA93, 0xAA8 }, .{ 0xAAA, 0xAB0 }, .{ 0xAB2, 0xAB3 },
    .{ 0xAB5, 0xAB9 }, .{ 0xABD, 0xABD }, .{ 0xAD0, 0xAD0 }, .{ 0xAE0, 0xAE1 }, .{ 0xAF9, 0xAF9 }, .{ 0xB05, 0xB0C },
    .{ 0xB0F, 0xB10 }, .{ 0xB13, 0xB28 }, .{ 0xB2A, 0xB30 }, .{ 0xB32, 0xB33 }, .{ 0xB35, 0xB39 }, .{ 0xB3D, 0xB3D },
    .{ 0xB5C, 0xB5D }, .{ 0xB5F, 0xB61 }, .{ 0xB71, 0xB71 }, .{ 0xB83, 0xB83 }, .{ 0xB85, 0xB8A }, .{ 0xB8E, 0xB90 },
    .{ 0xB92, 0xB95 }, .{ 0xB99, 0xB9A }, .{ 0xB9C, 0xB9C }, .{ 0xB9E, 0xB9F }, .{ 0xBA3, 0xBA4 }, .{ 0xBA8, 0xBAA },
    .{ 0xBAE, 0xBB9 }, .{ 0xBD0, 0xBD0 }, .{ 0xC05, 0xC0C }, .{ 0xC0E, 0xC10 }, .{ 0xC12, 0xC28 }, .{ 0xC2A, 0xC39 },
    .{ 0xC3D, 0xC3D }, .{ 0xC58, 0xC5A }, .{ 0xC5D, 0xC5D }, .{ 0xC60, 0xC61 }, .{ 0xC80, 0xC80 }, .{ 0xC85, 0xC8C },
    .{ 0xC8E, 0xC90 }, .{ 0xC92, 0xCA8 }, .{ 0xCAA, 0xCB3 }, .{ 0xCB5, 0xCB9 }, .{ 0xCBD, 0xCBD }, .{ 0xCDD, 0xCDE },
    .{ 0xCE0, 0xCE1 }, .{ 0xCF1, 0xCF2 }, .{ 0xD04, 0xD0C }, .{ 0xD0E, 0xD10 }, .{ 0xD12, 0xD3A }, .{ 0xD3D, 0xD3D },
    .{ 0xD4E, 0xD4E }, .{ 0xD54, 0xD56 }, .{ 0xD5F, 0xD61 }, .{ 0xD7A, 0xD7F }, .{ 0xD85, 0xD96 }, .{ 0xD9A, 0xDB1 },
    .{ 0xDB3, 0xDBB }, .{ 0xDBD, 0xDBD }, .{ 0xDC0, 0xDC6 }, .{ 0xE01, 0xE30 }, .{ 0xE32, 0xE33 }, .{ 0xE40, 0xE46 },
    .{ 0xE81, 0xE82 }, .{ 0xE84, 0xE84 }, .{ 0xE86, 0xE8A }, .{ 0xE8C, 0xEA3 }, .{ 0xEA5, 0xEA5 }, .{ 0xEA7, 0xEB0 },
    .{ 0xEB2, 0xEB3 }, .{ 0xEBD, 0xEBD }, .{ 0xEC0, 0xEC4 }, .{ 0xEC6, 0xEC6 }, .{ 0xEDC, 0xEDF }, .{ 0xF00, 0xF00 },
    .{ 0xF40, 0xF47 }, .{ 0xF49, 0xF6C }, .{ 0xF88, 0xF8C }, .{ 0x1000, 0x102A }, .{ 0x103F, 0x103F },
    .{ 0x1050, 0x1055 }, .{ 0x105A, 0x105D }, .{ 0x1061, 0x1061 }, .{ 0x1065, 0x1066 }, .{ 0x106E, 0x1070 },
    .{ 0x1075, 0x1081 }, .{ 0x108E, 0x108E }, .{ 0x10A0, 0x10C5 }, .{ 0x10C7, 0x10C7 }, .{ 0x10CD, 0x10CD },
    .{ 0x10D0, 0x10FA }, .{ 0x10FC, 0x1248 }, .{ 0x124A, 0x124D }, .{ 0x1250, 0x1256 }, .{ 0x1258, 0x1258 },
    .{ 0x125A, 0x125D }, .{ 0x1260, 0x1288 }, .{ 0x128A, 0x128D }, .{ 0x1290, 0x12B0 }, .{ 0x12B2, 0x12B5 },
    .{ 0x12B8, 0x12BE }, .{ 0x12C0, 0x12C0 }, .{ 0x12C2, 0x12C5 }, .{ 0x12C8, 0x12D6 }, .{ 0x12D8, 0x1310 },
    .{ 0x1312, 0x1315 }, .{ 0x1318, 0x135A }, .{ 0x1380, 0x138F }, .{ 0x13A0, 0x13F5 }, .{ 0x13F8, 0x13FD },
    .{ 0x1401, 0x166C }, .{ 0x166F, 0x167F }, .{ 0x1681, 0x169A }, .{ 0x16A0, 0x16EA }, .{ 0x16F1, 0x16F8 },
    .{ 0x1700, 0x1711 }, .{ 0x171F, 0x1731 }, .{ 0x1740, 0x1751 }, .{ 0x1760, 0x176C }, .{ 0x176E, 0x1770 },
    .{ 0x1780, 0x17B3 }, .{ 0x17D7, 0x17D7 }, .{ 0x17DC, 0x17DC }, .{ 0x1820, 0x1878 }, .{ 0x1880, 0x1884 },
    .{ 0x1887, 0x18A8 }, .{ 0x18AA, 0x18AA }, .{ 0x18B0, 0x18F5 }, .{ 0x1900, 0x191E }, .{ 0x1950, 0x196D },
    .{ 0x1970, 0x1974 }, .{ 0x1980, 0x19AB }, .{ 0x19B0, 0x19C9 }, .{ 0x1A00, 0x1A16 }, .{ 0x1A20, 0x1A54 },
    .{ 0x1AA7, 0x1AA7 }, .{ 0x1B05, 0x1B33 }, .{ 0x1B45, 0x1B4C }, .{ 0x1B83, 0x1BA0 }, .{ 0x1BAE, 0x1BAF },
    .{ 0x1BBA, 0x1BE5 }, .{ 0x1C00, 0x1C23 }, .{ 0x1C4D, 0x1C4F }, .{ 0x1C5A, 0x1C7D }, .{ 0x1C80, 0x1C88 },
    .{ 0x1C90, 0x1CBA }, .{ 0x1CBD, 0x1CBF }, .{ 0x1CE9, 0x1CEC }, .{ 0x1CEE, 0x1CF3 }, .{ 0x1CF5, 0x1CF6 },
    .{ 0x1CFA, 0x1CFA }, .{ 0x1D00, 0x1DBF }, .{ 0x1E00, 0x1F15 }, .{ 0x1F18, 0x1F1D }, .{ 0x1F20, 0x1F45 },
    .{ 0x1F48, 0x1F4D }, .{ 0x1F50, 0x1F57 }, .{ 0x1F59, 0x1F59 }, .{ 0x1F5B, 0x1F5B }, .{ 0x1F5D, 0x1F5D },
    .{ 0x1F5F, 0x1F7D }, .{ 0x1F80, 0x1FB4 }, .{ 0x1FB6, 0x1FBC }, .{ 0x1FBE, 0x1FBE }, .{ 0x1FC2, 0x1FC4 },
    .{ 0x1FC6, 0x1FCC }, .{ 0x1FD0, 0x1FD3 }, .{ 0x1FD6, 0x1FDB }, .{ 0x1FE0, 0x1FEC }, .{ 0x1FF2, 0x1FF4 },
    .{ 0x1FF6, 0x1FFC }, .{ 0x2071, 0x2071 }, .{ 0x207F, 0x207F }, .{ 0x2090, 0x209C }, .{ 0x2102, 0x2102 },
    .{ 0x2107, 0x2107 }, .{ 0x210A, 0x2113 }, .{ 0x2115, 0x2115 }, .{ 0x2119, 0x211D }, .{ 0x2124, 0x2124 },
    .{ 0x2126, 0x2126 }, .{ 0x2128, 0x2128 }, .{ 0x212A, 0x212D }, .{ 0x212F, 0x2139 }, .{ 0x213C, 0x213F },
    .{ 0x2145, 0x2149 }, .{ 0x214E, 0x214E }, .{ 0x2183, 0x2184 }, .{ 0x2C00, 0x2CE4 }, .{ 0x2CEB, 0x2CEE },
    .{ 0x2CF2, 0x2CF3 }, .{ 0x2D00, 0x2D25 }, .{ 0x2D27, 0x2D27 }, .{ 0x2D2D, 0x2D2D }, .{ 0x2D30, 0x2D67 },
    .{ 0x2D6F, 0x2D6F }, .{ 0x2D80, 0x2D96 }, .{ 0x2DA0, 0x2DA6 }, .{ 0x2DA8, 0x2DAE }, .{ 0x2DB0, 0x2DB6 },
    .{ 0x2DB8, 0x2DBE }, .{ 0x2DC0, 0x2DC6 }, .{ 0x2DC8, 0x2DCE }, .{ 0x2DD0, 0x2DD6 }, .{ 0x2DD8, 0x2DDE },
    .{ 0x2E2F, 0x2E2F }, .{ 0x3005, 0x3006 }, .{ 0x3031, 0x3035 }, .{ 0x303B, 0x303C }, .{ 0x3041, 0x3096 },
    .{ 0x309D, 0x309F }, .{ 0x30A1, 0x30FA }, .{ 0x30FC, 0x30FF }, .{ 0x3105, 0x312F }, .{ 0x3131, 0x318E },
    .{ 0x31A0, 0x31BF }, .{ 0x31F0, 0x31FF }, .{ 0x3400, 0x4DBF }, .{ 0x4E00, 0xA48C }, .{ 0xA4D0, 0xA4FD },
    .{ 0xA500, 0xA60C }, .{ 0xA610, 0xA61F }, .{ 0xA62A, 0xA62B }, .{ 0xA640, 0xA66E }, .{ 0xA67F, 0xA69D },
    .{ 0xA6A0, 0xA6E5 }, .{ 0xA717, 0xA71F }, .{ 0xA722, 0xA788 }, .{ 0xA78B, 0xA7CA }, .{ 0xA7D0, 0xA7D1 },
    .{ 0xA7D3, 0xA7D3 }, .{ 0xA7D5, 0xA7D9 }, .{ 0xA7F2, 0xA801 }, .{ 0xA803, 0xA805 }, .{ 0xA807, 0xA80A },
    .{ 0xA80C, 0xA822 }, .{ 0xA840, 0xA873 }, .{ 0xA882, 0xA8B3 }, .{ 0xA8F2, 0xA8F7 }, .{ 0xA8FB, 0xA8FB },
    .{ 0xA8FD, 0xA8FE }, .{ 0xA90A, 0xA925 }, .{ 0xA930, 0xA946 }, .{ 0xA960, 0xA97C }, .{ 0xA984, 0xA9B2 },
    .{ 0xA9CF, 0xA9CF }, .{ 0xA9E0, 0xA9E4 }, .{ 0xA9E6, 0xA9EF }, .{ 0xA9FA, 0xA9FE }, .{ 0xAA00, 0xAA28 },
    .{ 0xAA40, 0xAA42 }, .{ 0xAA44, 0xAA4B }, .{ 0xAA60, 0xAA76 }, .{ 0xAA7A, 0xAA7A }, .{ 0xAA7E, 0xAAAF },
    .{ 0xAAB1, 0xAAB1 }, .{ 0xAAB5, 0xAAB6 }, .{ 0xAAB9, 0xAABD }, .{ 0xAAC0, 0xAAC0 }, .{ 0xAAC2, 0xAAC2 },
    .{ 0xAADB, 0xAADD }, .{ 0xAAE0, 0xAAEA }, .{ 0xAAF2, 0xAAF4 }, .{ 0xAB01, 0xAB06 }, .{ 0xAB09, 0xAB0E },
    .{ 0xAB11, 0xAB16 }, .{ 0xAB20, 0xAB26 }, .{ 0xAB28, 0xAB2E }, .{ 0xAB30, 0xAB5A }, .{ 0xAB5C, 0xAB69 },
    .{ 0xAB70, 0xABE2 }, .{ 0xAC00, 0xD7A3 }, .{ 0xD7B0, 0xD7C6 }, .{ 0xD7CB, 0xD7FB }, .{ 0xF900, 0xFA6D },
    .{ 0xFA70, 0xFAD9 }, .{ 0xFB00, 0xFB06 }, .{ 0xFB13, 0xFB17 }, .{ 0xFB1D, 0xFB1D }, .{ 0xFB1F, 0xFB28 },
    .{ 0xFB2A, 0xFB36 }, .{ 0xFB38, 0xFB3C }, .{ 0xFB3E, 0xFB3E }, .{ 0xFB40, 0xFB41 }, .{ 0xFB43, 0xFB44 },
    .{ 0xFB46, 0xFBB1 }, .{ 0xFBD3, 0xFD3D }, .{ 0xFD50, 0xFD8F }, .{ 0xFD92, 0xFDC7 }, .{ 0xFDF0, 0xFDFB },
    .{ 0xFE70, 0xFE74 }, .{ 0xFE76, 0xFEFC }, .{ 0xFF21, 0xFF3A }, .{ 0xFF41, 0xFF5A }, .{ 0xFF66, 0xFFBE },
    .{ 0xFFC2, 0xFFC7 }, .{ 0xFFCA, 0xFFCF }, .{ 0xFFD2, 0xFFD7 }, .{ 0xFFDA, 0xFFDC }, .{ 0x10000, 0x1000B },
    .{ 0x1000D, 0x10026 }, .{ 0x10028, 0x1003A }, .{ 0x1003C, 0x1003D }, .{ 0x1003F, 0x1004D }, .{ 0x10050, 0x1005D },
    .{ 0x10080, 0x100FA }, .{ 0x10280, 0x1029C }, .{ 0x102A0, 0x102D0 }, .{ 0x10300, 0x1031F }, .{ 0x1032D, 0x10340 },
    .{ 0x10342, 0x10349 }, .{ 0x10350, 0x10375 }, .{ 0x10380, 0x1039D }, .{ 0x103A0, 0x103C3 }, .{ 0x103C8, 0x103CF },
    .{ 0x10400, 0x1049D }, .{ 0x104B0, 0x104D3 }, .{ 0x104D8, 0x104FB }, .{ 0x10500, 0x10527 }, .{ 0x10530, 0x10563 },
    .{ 0x10570, 0x1057A }, .{ 0x1057C, 0x1058A }, .{ 0x1058C, 0x10592 }, .{ 0x10594, 0x10595 }, .{ 0x10597, 0x105A1 },
    .{ 0x105A3, 0x105B1 }, .{ 0x105B3, 0x105B9 }, .{ 0x105BB, 0x105BC }, .{ 0x10600, 0x10736 }, .{ 0x10740, 0x10755 },
    .{ 0x10760, 0x10767 }, .{ 0x10780, 0x10785 }, .{ 0x10787, 0x107B0 }, .{ 0x107B2, 0x107BA }, .{ 0x10800, 0x10805 },
    .{ 0x10808, 0x10808 }, .{ 0x1080A, 0x10835 }, .{ 0x10837, 0x10838 }, .{ 0x1083C, 0x1083C }, .{ 0x1083F, 0x10855 },
    .{ 0x10860, 0x10876 }, .{ 0x10880, 0x1089E }, .{ 0x108E0, 0x108F2 }, .{ 0x108F4, 0x108F5 }, .{ 0x10900, 0x10915 },
    .{ 0x10920, 0x10939 }, .{ 0x10980, 0x109B7 }, .{ 0x109BE, 0x109BF }, .{ 0x10A00, 0x10A00 }, .{ 0x10A10, 0x10A13 },
    .{ 0x10A15, 0x10A17 }, .{ 0x10A19, 0x10A35 }, .{ 0x10A60, 0x10A7C }, .{ 0x10A80, 0x10A9C }, .{ 0x10AC0, 0x10AC7 },
    .{ 0x10AC9, 0x10AE4 }, .{ 0x10B00, 0x10B35 }, .{ 0x10B40, 0x10B55 }, .{ 0x10B60, 0x10B72 }, .{ 0x10B80, 0x10B91 },
    .{ 0x10C00, 0x10C48 }, .{ 0x10C80, 0x10CB2 }, .{ 0x10CC0, 0x10CF2 }, .{ 0x10D00, 0x10D23 }, .{ 0x10E80, 0x10EA9 },
    .{ 0x10EB0, 0x10EB1 }, .{ 0x10F00, 0x10F1C }, .{ 0x10F27, 0x10F27 }, .{ 0x10F30, 0x10F45 }, .{ 0x10F70, 0x10F81 },
    .{ 0x10FB0, 0x10FC4 }, .{ 0x10FE0, 0x10FF6 }, .{ 0x11003, 0x11037 }, .{ 0x11071, 0x11072 }, .{ 0x11075, 0x11075 },
    .{ 0x11083, 0x110AF }, .{ 0x110D0, 0x110E8 }, .{ 0x11103, 0x11126 }, .{ 0x11144, 0x11144 }, .{ 0x11147, 0x11147 },
    .{ 0x11150, 0x11172 }, .{ 0x11176, 0x11176 }, .{ 0x11183, 0x111B2 }, .{ 0x111C1, 0x111C4 }, .{ 0x111DA, 0x111DA },
    .{ 0x111DC, 0x111DC }, .{ 0x11200, 0x11211 }, .{ 0x11213, 0x1122B }, .{ 0x1123F, 0x11240 }, .{ 0x11280, 0x11286 },
    .{ 0x11288, 0x11288 }, .{ 0x1128A, 0x1128D }, .{ 0x1128F, 0x1129D }, .{ 0x1129F, 0x112A8 }, .{ 0x112B0, 0x112DE },
    .{ 0x11305, 0x1130C }, .{ 0x1130F, 0x11310 }, .{ 0x11313, 0x11328 }, .{ 0x1132A, 0x11330 }, .{ 0x11332, 0x11333 },
    .{ 0x11335, 0x11339 }, .{ 0x1133D, 0x1133D }, .{ 0x11350, 0x11350 }, .{ 0x1135D, 0x11361 }, .{ 0x11400, 0x11434 },
    .{ 0x11447, 0x1144A }, .{ 0x1145F, 0x11461 }, .{ 0x11480, 0x114AF }, .{ 0x114C4, 0x114C5 }, .{ 0x114C7, 0x114C7 },
    .{ 0x11580, 0x115AE }, .{ 0x115D8, 0x115DB }, .{ 0x11600, 0x1162F }, .{ 0x11644, 0x11644 }, .{ 0x11680, 0x116AA },
    .{ 0x116B8, 0x116B8 }, .{ 0x11700, 0x1171A }, .{ 0x11740, 0x11746 }, .{ 0x11800, 0x1182B }, .{ 0x118A0, 0x118DF },
    .{ 0x118FF, 0x11906 }, .{ 0x11909, 0x11909 }, .{ 0x1190C, 0x11913 }, .{ 0x11915, 0x11916 }, .{ 0x11918, 0x1192F },
    .{ 0x1193F, 0x1193F }, .{ 0x11941, 0x11941 }, .{ 0x119A0, 0x119A7 }, .{ 0x119AA, 0x119D0 }, .{ 0x119E1, 0x119E1 },
    .{ 0x119E3, 0x119E3 }, .{ 0x11A00, 0x11A00 }, .{ 0x11A0B, 0x11A32 }, .{ 0x11A3A, 0x11A3A }, .{ 0x11A50, 0x11A50 },
    .{ 0x11A5C, 0x11A89 }, .{ 0x11A9D, 0x11A9D }, .{ 0x11AB0, 0x11AF8 }, .{ 0x11C00, 0x11C08 }, .{ 0x11C0A, 0x11C2E },
    .{ 0x11C40, 0x11C40 }, .{ 0x11C72, 0x11C8F }, .{ 0x11D00, 0x11D06 }, .{ 0x11D08, 0x11D09 }, .{ 0x11D0B, 0x11D30 },
    .{ 0x11D46, 0x11D46 }, .{ 0x11D60, 0x11D65 }, .{ 0x11D67, 0x11D68 }, .{ 0x11D6A, 0x11D89 }, .{ 0x11D98, 0x11D98 },
    .{ 0x11EE0, 0x11EF2 }, .{ 0x11F02, 0x11F02 }, .{ 0x11F04, 0x11F10 }, .{ 0x11F12, 0x11F33 }, .{ 0x11FB0, 0x11FB0 },
    .{ 0x12000, 0x12399 }, .{ 0x12480, 0x12543 }, .{ 0x12F90, 0x12FF0 }, .{ 0x13000, 0x1342F }, .{ 0x13441, 0x13446 },
    .{ 0x14400, 0x14646 }, .{ 0x16800, 0x16A38 }, .{ 0x16A40, 0x16A5E }, .{ 0x16A70, 0x16ABE }, .{ 0x16AD0, 0x16AED },
    .{ 0x16B00, 0x16B2F }, .{ 0x16B40, 0x16B43 }, .{ 0x16B63, 0x16B77 }, .{ 0x16B7D, 0x16B8F }, .{ 0x16E40, 0x16E7F },
    .{ 0x16F00, 0x16F4A }, .{ 0x16F50, 0x16F50 }, .{ 0x16F93, 0x16F9F }, .{ 0x16FE0, 0x16FE1 }, .{ 0x16FE3, 0x16FE3 },
    .{ 0x17000, 0x187F7 }, .{ 0x18800, 0x18CD5 }, .{ 0x18D00, 0x18D08 }, .{ 0x1AFF0, 0x1AFF3 }, .{ 0x1AFF5, 0x1AFFB },
    .{ 0x1AFFD, 0x1AFFE }, .{ 0x1B000, 0x1B122 }, .{ 0x1B132, 0x1B132 }, .{ 0x1B150, 0x1B152 }, .{ 0x1B155, 0x1B155 },
    .{ 0x1B164, 0x1B167 }, .{ 0x1B170, 0x1B2FB }, .{ 0x1BC00, 0x1BC6A }, .{ 0x1BC70, 0x1BC7C }, .{ 0x1BC80, 0x1BC88 },
    .{ 0x1BC90, 0x1BC99 }, .{ 0x1D400, 0x1D454 }, .{ 0x1D456, 0x1D49C }, .{ 0x1D49E, 0x1D49F }, .{ 0x1D4A2, 0x1D4A2 },
    .{ 0x1D4A5, 0x1D4A6 }, .{ 0x1D4A9, 0x1D4AC }, .{ 0x1D4AE, 0x1D4B9 }, .{ 0x1D4BB, 0x1D4BB }, .{ 0x1D4BD, 0x1D4C3 },
    .{ 0x1D4C5, 0x1D505 }, .{ 0x1D507, 0x1D50A }, .{ 0x1D50D, 0x1D514 }, .{ 0x1D516, 0x1D51C }, .{ 0x1D51E, 0x1D539 },
    .{ 0x1D53B, 0x1D53E }, .{ 0x1D540, 0x1D544 }, .{ 0x1D546, 0x1D546 }, .{ 0x1D54A, 0x1D550 }, .{ 0x1D552, 0x1D6A5 },
    .{ 0x1D6A8, 0x1D6C0 }, .{ 0x1D6C2, 0x1D6DA }, .{ 0x1D6DC, 0x1D6FA }, .{ 0x1D6FC, 0x1D714 }, .{ 0x1D716, 0x1D734 },
    .{ 0x1D736, 0x1D74E }, .{ 0x1D750, 0x1D76E }, .{ 0x1D770, 0x1D788 }, .{ 0x1D78A, 0x1D7A8 }, .{ 0x1D7AA, 0x1D7C2 },
    .{ 0x1D7C4, 0x1D7CB }, .{ 0x1DF00, 0x1DF1E }, .{ 0x1DF25, 0x1DF2A }, .{ 0x1E030, 0x1E06D }, .{ 0x1E100, 0x1E12C },
    .{ 0x1E137, 0x1E13D }, .{ 0x1E14E, 0x1E14E }, .{ 0x1E290, 0x1E2AD }, .{ 0x1E2C0, 0x1E2EB }, .{ 0x1E4D0, 0x1E4EB },
    .{ 0x1E7E0, 0x1E7E6 }, .{ 0x1E7E8, 0x1E7EB }, .{ 0x1E7ED, 0x1E7EE }, .{ 0x1E7F0, 0x1E7FE }, .{ 0x1E800, 0x1E8C4 },
    .{ 0x1E900, 0x1E943 }, .{ 0x1E94B, 0x1E94B }, .{ 0x1EE00, 0x1EE03 }, .{ 0x1EE05, 0x1EE1F }, .{ 0x1EE21, 0x1EE22 },
    .{ 0x1EE24, 0x1EE24 }, .{ 0x1EE27, 0x1EE27 }, .{ 0x1EE29, 0x1EE32 }, .{ 0x1EE34, 0x1EE37 }, .{ 0x1EE39, 0x1EE39 },
    .{ 0x1EE3B, 0x1EE3B }, .{ 0x1EE42, 0x1EE42 }, .{ 0x1EE47, 0x1EE47 }, .{ 0x1EE49, 0x1EE49 }, .{ 0x1EE4B, 0x1EE4B },
    .{ 0x1EE4D, 0x1EE4F }, .{ 0x1EE51, 0x1EE52 }, .{ 0x1EE54, 0x1EE54 }, .{ 0x1EE57, 0x1EE57 }, .{ 0x1EE59, 0x1EE59 },
    .{ 0x1EE5B, 0x1EE5B }, .{ 0x1EE5D, 0x1EE5D }, .{ 0x1EE5F, 0x1EE5F }, .{ 0x1EE61, 0x1EE62 }, .{ 0x1EE64, 0x1EE64 },
    .{ 0x1EE67, 0x1EE6A }, .{ 0x1EE6C, 0x1EE72 }, .{ 0x1EE74, 0x1EE77 }, .{ 0x1EE79, 0x1EE7C }, .{ 0x1EE7E, 0x1EE7E },
    .{ 0x1EE80, 0x1EE89 }, .{ 0x1EE8B, 0x1EE9B }, .{ 0x1EEA1, 0x1EEA3 }, .{ 0x1EEA5, 0x1EEA9 }, .{ 0x1EEAB, 0x1EEBB },
    .{ 0x20000, 0x2A6DF }, .{ 0x2A700, 0x2B739 }, .{ 0x2B740, 0x2B81D }, .{ 0x2B820, 0x2CEA1 }, .{ 0x2CEB0, 0x2EBE0 },
    .{ 0x2EBF0, 0x2EE5D }, .{ 0x2F800, 0x2FA1D }, .{ 0x30000, 0x3134A }, .{ 0x31350, 0x323AF },
};

/// `\p{M}` as inclusive ranges, Unicode 15.1.0: `[(a, b) for runs of unicodedata.category(c)[0] == 'M']`.
const MARK_RANGES = [_][2]u21{
    .{ 0x300, 0x36F }, .{ 0x483, 0x489 }, .{ 0x591, 0x5BD }, .{ 0x5BF, 0x5BF }, .{ 0x5C1, 0x5C2 }, .{ 0x5C4, 0x5C5 },
    .{ 0x5C7, 0x5C7 }, .{ 0x610, 0x61A }, .{ 0x64B, 0x65F }, .{ 0x670, 0x670 }, .{ 0x6D6, 0x6DC }, .{ 0x6DF, 0x6E4 },
    .{ 0x6E7, 0x6E8 }, .{ 0x6EA, 0x6ED }, .{ 0x711, 0x711 }, .{ 0x730, 0x74A }, .{ 0x7A6, 0x7B0 }, .{ 0x7EB, 0x7F3 },
    .{ 0x7FD, 0x7FD }, .{ 0x816, 0x819 }, .{ 0x81B, 0x823 }, .{ 0x825, 0x827 }, .{ 0x829, 0x82D }, .{ 0x859, 0x85B },
    .{ 0x898, 0x89F }, .{ 0x8CA, 0x8E1 }, .{ 0x8E3, 0x903 }, .{ 0x93A, 0x93C }, .{ 0x93E, 0x94F }, .{ 0x951, 0x957 },
    .{ 0x962, 0x963 }, .{ 0x981, 0x983 }, .{ 0x9BC, 0x9BC }, .{ 0x9BE, 0x9C4 }, .{ 0x9C7, 0x9C8 }, .{ 0x9CB, 0x9CD },
    .{ 0x9D7, 0x9D7 }, .{ 0x9E2, 0x9E3 }, .{ 0x9FE, 0x9FE }, .{ 0xA01, 0xA03 }, .{ 0xA3C, 0xA3C }, .{ 0xA3E, 0xA42 },
    .{ 0xA47, 0xA48 }, .{ 0xA4B, 0xA4D }, .{ 0xA51, 0xA51 }, .{ 0xA70, 0xA71 }, .{ 0xA75, 0xA75 }, .{ 0xA81, 0xA83 },
    .{ 0xABC, 0xABC }, .{ 0xABE, 0xAC5 }, .{ 0xAC7, 0xAC9 }, .{ 0xACB, 0xACD }, .{ 0xAE2, 0xAE3 }, .{ 0xAFA, 0xAFF },
    .{ 0xB01, 0xB03 }, .{ 0xB3C, 0xB3C }, .{ 0xB3E, 0xB44 }, .{ 0xB47, 0xB48 }, .{ 0xB4B, 0xB4D }, .{ 0xB55, 0xB57 },
    .{ 0xB62, 0xB63 }, .{ 0xB82, 0xB82 }, .{ 0xBBE, 0xBC2 }, .{ 0xBC6, 0xBC8 }, .{ 0xBCA, 0xBCD }, .{ 0xBD7, 0xBD7 },
    .{ 0xC00, 0xC04 }, .{ 0xC3C, 0xC3C }, .{ 0xC3E, 0xC44 }, .{ 0xC46, 0xC48 }, .{ 0xC4A, 0xC4D }, .{ 0xC55, 0xC56 },
    .{ 0xC62, 0xC63 }, .{ 0xC81, 0xC83 }, .{ 0xCBC, 0xCBC }, .{ 0xCBE, 0xCC4 }, .{ 0xCC6, 0xCC8 }, .{ 0xCCA, 0xCCD },
    .{ 0xCD5, 0xCD6 }, .{ 0xCE2, 0xCE3 }, .{ 0xCF3, 0xCF3 }, .{ 0xD00, 0xD03 }, .{ 0xD3B, 0xD3C }, .{ 0xD3E, 0xD44 },
    .{ 0xD46, 0xD48 }, .{ 0xD4A, 0xD4D }, .{ 0xD57, 0xD57 }, .{ 0xD62, 0xD63 }, .{ 0xD81, 0xD83 }, .{ 0xDCA, 0xDCA },
    .{ 0xDCF, 0xDD4 }, .{ 0xDD6, 0xDD6 }, .{ 0xDD8, 0xDDF }, .{ 0xDF2, 0xDF3 }, .{ 0xE31, 0xE31 }, .{ 0xE34, 0xE3A },
    .{ 0xE47, 0xE4E }, .{ 0xEB1, 0xEB1 }, .{ 0xEB4, 0xEBC }, .{ 0xEC8, 0xECE }, .{ 0xF18, 0xF19 }, .{ 0xF35, 0xF35 },
    .{ 0xF37, 0xF37 }, .{ 0xF39, 0xF39 }, .{ 0xF3E, 0xF3F }, .{ 0xF71, 0xF84 }, .{ 0xF86, 0xF87 }, .{ 0xF8D, 0xF97 },
    .{ 0xF99, 0xFBC }, .{ 0xFC6, 0xFC6 }, .{ 0x102B, 0x103E }, .{ 0x1056, 0x1059 }, .{ 0x105E, 0x1060 },
    .{ 0x1062, 0x1064 }, .{ 0x1067, 0x106D }, .{ 0x1071, 0x1074 }, .{ 0x1082, 0x108D }, .{ 0x108F, 0x108F },
    .{ 0x109A, 0x109D }, .{ 0x135D, 0x135F }, .{ 0x1712, 0x1715 }, .{ 0x1732, 0x1734 }, .{ 0x1752, 0x1753 },
    .{ 0x1772, 0x1773 }, .{ 0x17B4, 0x17D3 }, .{ 0x17DD, 0x17DD }, .{ 0x180B, 0x180D }, .{ 0x180F, 0x180F },
    .{ 0x1885, 0x1886 }, .{ 0x18A9, 0x18A9 }, .{ 0x1920, 0x192B }, .{ 0x1930, 0x193B }, .{ 0x1A17, 0x1A1B },
    .{ 0x1A55, 0x1A5E }, .{ 0x1A60, 0x1A7C }, .{ 0x1A7F, 0x1A7F }, .{ 0x1AB0, 0x1ACE }, .{ 0x1B00, 0x1B04 },
    .{ 0x1B34, 0x1B44 }, .{ 0x1B6B, 0x1B73 }, .{ 0x1B80, 0x1B82 }, .{ 0x1BA1, 0x1BAD }, .{ 0x1BE6, 0x1BF3 },
    .{ 0x1C24, 0x1C37 }, .{ 0x1CD0, 0x1CD2 }, .{ 0x1CD4, 0x1CE8 }, .{ 0x1CED, 0x1CED }, .{ 0x1CF4, 0x1CF4 },
    .{ 0x1CF7, 0x1CF9 }, .{ 0x1DC0, 0x1DFF }, .{ 0x20D0, 0x20F0 }, .{ 0x2CEF, 0x2CF1 }, .{ 0x2D7F, 0x2D7F },
    .{ 0x2DE0, 0x2DFF }, .{ 0x302A, 0x302F }, .{ 0x3099, 0x309A }, .{ 0xA66F, 0xA672 }, .{ 0xA674, 0xA67D },
    .{ 0xA69E, 0xA69F }, .{ 0xA6F0, 0xA6F1 }, .{ 0xA802, 0xA802 }, .{ 0xA806, 0xA806 }, .{ 0xA80B, 0xA80B },
    .{ 0xA823, 0xA827 }, .{ 0xA82C, 0xA82C }, .{ 0xA880, 0xA881 }, .{ 0xA8B4, 0xA8C5 }, .{ 0xA8E0, 0xA8F1 },
    .{ 0xA8FF, 0xA8FF }, .{ 0xA926, 0xA92D }, .{ 0xA947, 0xA953 }, .{ 0xA980, 0xA983 }, .{ 0xA9B3, 0xA9C0 },
    .{ 0xA9E5, 0xA9E5 }, .{ 0xAA29, 0xAA36 }, .{ 0xAA43, 0xAA43 }, .{ 0xAA4C, 0xAA4D }, .{ 0xAA7B, 0xAA7D },
    .{ 0xAAB0, 0xAAB0 }, .{ 0xAAB2, 0xAAB4 }, .{ 0xAAB7, 0xAAB8 }, .{ 0xAABE, 0xAABF }, .{ 0xAAC1, 0xAAC1 },
    .{ 0xAAEB, 0xAAEF }, .{ 0xAAF5, 0xAAF6 }, .{ 0xABE3, 0xABEA }, .{ 0xABEC, 0xABED }, .{ 0xFB1E, 0xFB1E },
    .{ 0xFE00, 0xFE0F }, .{ 0xFE20, 0xFE2F }, .{ 0x101FD, 0x101FD }, .{ 0x102E0, 0x102E0 }, .{ 0x10376, 0x1037A },
    .{ 0x10A01, 0x10A03 }, .{ 0x10A05, 0x10A06 }, .{ 0x10A0C, 0x10A0F }, .{ 0x10A38, 0x10A3A }, .{ 0x10A3F, 0x10A3F },
    .{ 0x10AE5, 0x10AE6 }, .{ 0x10D24, 0x10D27 }, .{ 0x10EAB, 0x10EAC }, .{ 0x10EFD, 0x10EFF }, .{ 0x10F46, 0x10F50 },
    .{ 0x10F82, 0x10F85 }, .{ 0x11000, 0x11002 }, .{ 0x11038, 0x11046 }, .{ 0x11070, 0x11070 }, .{ 0x11073, 0x11074 },
    .{ 0x1107F, 0x11082 }, .{ 0x110B0, 0x110BA }, .{ 0x110C2, 0x110C2 }, .{ 0x11100, 0x11102 }, .{ 0x11127, 0x11134 },
    .{ 0x11145, 0x11146 }, .{ 0x11173, 0x11173 }, .{ 0x11180, 0x11182 }, .{ 0x111B3, 0x111C0 }, .{ 0x111C9, 0x111CC },
    .{ 0x111CE, 0x111CF }, .{ 0x1122C, 0x11237 }, .{ 0x1123E, 0x1123E }, .{ 0x11241, 0x11241 }, .{ 0x112DF, 0x112EA },
    .{ 0x11300, 0x11303 }, .{ 0x1133B, 0x1133C }, .{ 0x1133E, 0x11344 }, .{ 0x11347, 0x11348 }, .{ 0x1134B, 0x1134D },
    .{ 0x11357, 0x11357 }, .{ 0x11362, 0x11363 }, .{ 0x11366, 0x1136C }, .{ 0x11370, 0x11374 }, .{ 0x11435, 0x11446 },
    .{ 0x1145E, 0x1145E }, .{ 0x114B0, 0x114C3 }, .{ 0x115AF, 0x115B5 }, .{ 0x115B8, 0x115C0 }, .{ 0x115DC, 0x115DD },
    .{ 0x11630, 0x11640 }, .{ 0x116AB, 0x116B7 }, .{ 0x1171D, 0x1172B }, .{ 0x1182C, 0x1183A }, .{ 0x11930, 0x11935 },
    .{ 0x11937, 0x11938 }, .{ 0x1193B, 0x1193E }, .{ 0x11940, 0x11940 }, .{ 0x11942, 0x11943 }, .{ 0x119D1, 0x119D7 },
    .{ 0x119DA, 0x119E0 }, .{ 0x119E4, 0x119E4 }, .{ 0x11A01, 0x11A0A }, .{ 0x11A33, 0x11A39 }, .{ 0x11A3B, 0x11A3E },
    .{ 0x11A47, 0x11A47 }, .{ 0x11A51, 0x11A5B }, .{ 0x11A8A, 0x11A99 }, .{ 0x11C2F, 0x11C36 }, .{ 0x11C38, 0x11C3F },
    .{ 0x11C92, 0x11CA7 }, .{ 0x11CA9, 0x11CB6 }, .{ 0x11D31, 0x11D36 }, .{ 0x11D3A, 0x11D3A }, .{ 0x11D3C, 0x11D3D },
    .{ 0x11D3F, 0x11D45 }, .{ 0x11D47, 0x11D47 }, .{ 0x11D8A, 0x11D8E }, .{ 0x11D90, 0x11D91 }, .{ 0x11D93, 0x11D97 },
    .{ 0x11EF3, 0x11EF6 }, .{ 0x11F00, 0x11F01 }, .{ 0x11F03, 0x11F03 }, .{ 0x11F34, 0x11F3A }, .{ 0x11F3E, 0x11F42 },
    .{ 0x13440, 0x13440 }, .{ 0x13447, 0x13455 }, .{ 0x16AF0, 0x16AF4 }, .{ 0x16B30, 0x16B36 }, .{ 0x16F4F, 0x16F4F },
    .{ 0x16F51, 0x16F87 }, .{ 0x16F8F, 0x16F92 }, .{ 0x16FE4, 0x16FE4 }, .{ 0x16FF0, 0x16FF1 }, .{ 0x1BC9D, 0x1BC9E },
    .{ 0x1CF00, 0x1CF2D }, .{ 0x1CF30, 0x1CF46 }, .{ 0x1D165, 0x1D169 }, .{ 0x1D16D, 0x1D172 }, .{ 0x1D17B, 0x1D182 },
    .{ 0x1D185, 0x1D18B }, .{ 0x1D1AA, 0x1D1AD }, .{ 0x1D242, 0x1D244 }, .{ 0x1DA00, 0x1DA36 }, .{ 0x1DA3B, 0x1DA6C },
    .{ 0x1DA75, 0x1DA75 }, .{ 0x1DA84, 0x1DA84 }, .{ 0x1DA9B, 0x1DA9F }, .{ 0x1DAA1, 0x1DAAF }, .{ 0x1E000, 0x1E006 },
    .{ 0x1E008, 0x1E018 }, .{ 0x1E01B, 0x1E021 }, .{ 0x1E023, 0x1E024 }, .{ 0x1E026, 0x1E02A }, .{ 0x1E08F, 0x1E08F },
    .{ 0x1E130, 0x1E136 }, .{ 0x1E2AE, 0x1E2AE }, .{ 0x1E2EC, 0x1E2EF }, .{ 0x1E4EC, 0x1E4EF }, .{ 0x1E8D0, 0x1E8D6 },
    .{ 0x1E944, 0x1E94A }, .{ 0xE0100, 0xE01EF },
};

/// Build the GPT-2 bytes_to_unicode mapping (256 entries).
pub fn buildBytesToUnicode() [256]u21 {
    var table: [256]u21 = undefined;
    var n: u21 = 256; // Next available unicode codepoint for unmapped bytes

    for (0..256) |b| {
        const byte: u8 = @intCast(b);
        // Printable ASCII range + some Latin-1 chars get identity mapping
        if ((byte >= '!' and byte <= '~') or
            (byte >= 0xA1 and byte <= 0xAC) or
            (byte >= 0xAE and byte <= 0xFF))
        {
            table[b] = @intCast(b);
        } else {
            // Non-printable bytes get mapped to codepoints starting at U+0100
            table[b] = n;
            n += 1;
        }
    }
    return table;
}
