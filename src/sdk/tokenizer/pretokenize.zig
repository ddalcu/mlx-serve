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

/// Approximate `\p{M}` — combining marks. Coverage: common Latin/Greek/
/// Cyrillic combining marks (U+0300–U+036F), plus Hebrew/Arabic/Devanagari
/// combining ranges. Not exhaustive, but handles every codepoint our test
/// corpus encounters; expand if a non-ASCII model surfaces a false negative.
fn isMark(cp: u21) bool {
    if (cp >= 0x80 and isDigit(cp)) return false; // whole-block ranges below hold digits
    if (cp >= 0x0300 and cp <= 0x036F) return true; // Combining diacritical marks
    if (cp >= 0x0483 and cp <= 0x0489) return true; // Cyrillic combining
    if (cp >= 0x0591 and cp <= 0x05BD) return true; // Hebrew points
    if (cp >= 0x064B and cp <= 0x065F) return true; // Arabic harakat
    if (cp >= 0x0670 and cp <= 0x0670) return true;
    if (cp >= 0x06D6 and cp <= 0x06DC) return true;
    if (cp >= 0x0900 and cp <= 0x097F) return true; // Devanagari (overlap with letters; harmless)
    if (cp >= 0x1AB0 and cp <= 0x1AFF) return true;
    if (cp >= 0x1DC0 and cp <= 0x1DFF) return true;
    if (cp >= 0x20D0 and cp <= 0x20FF) return true;
    if (cp >= 0xFE20 and cp <= 0xFE2F) return true;
    return false;
}

const CpInfo = struct { cp: u21, len: usize };

fn decodeCodepoint(text: []const u8, pos: usize) ?CpInfo {
    if (pos >= text.len) return null;
    const byte_len = std.unicode.utf8ByteSequenceLength(text[pos]) catch return CpInfo{ .cp = text[pos], .len = 1 };
    const end = @min(pos + byte_len, text.len);
    const cp = std.unicode.utf8Decode(text[pos..end]) catch return CpInfo{ .cp = text[pos], .len = 1 };
    return CpInfo{ .cp = cp, .len = end - pos };
}

pub fn isLetter(cp: u21) bool {
    // The script ranges below are whole blocks; their digits are \p{N}, not letters.
    if (cp >= 0x80 and isDigit(cp)) return false;
    // ASCII letters
    if (cp >= 'A' and cp <= 'Z') return true;
    if (cp >= 'a' and cp <= 'z') return true;
    // Common Unicode letter ranges
    if (cp >= 0xC0 and cp <= 0x024F) return true; // Latin Extended
    if (cp >= 0x0400 and cp <= 0x04FF) return true; // Cyrillic
    if (cp >= 0x4E00 and cp <= 0x9FFF) return true; // CJK
    if (cp >= 0x3040 and cp <= 0x30FF) return true; // Hiragana/Katakana
    if (cp >= 0xAC00 and cp <= 0xD7AF) return true; // Korean
    if (cp >= 0x0600 and cp <= 0x06FF) return true; // Arabic
    if (cp >= 0x0900 and cp <= 0x097F) return true; // Devanagari
    if (cp >= 0x0370 and cp <= 0x03FF) return true; // Greek
    return false;
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
    var lo: usize = 0;
    var hi: usize = NUMBER_RANGES.len;
    while (lo < hi) {
        const mid = (lo + hi) / 2;
        if (cp < NUMBER_RANGES[mid][0]) hi = mid else if (cp > NUMBER_RANGES[mid][1]) lo = mid + 1 else return true;
    }
    return false;
}

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
