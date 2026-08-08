import Foundation

/// Token classes a code block paints. Deliberately small: these are the
/// distinctions that make code readable at a glance, not a compiler's token
/// taxonomy. Anything unclassified is left in the block's default color, so the
/// tokenizer only ever needs to be RIGHT about what it does claim — a missed
/// keyword looks plain, a wrongly-claimed one looks broken.
enum SyntaxKind: String, Sendable {
    case keyword
    case type
    case function
    case property   // JSON/markup attribute or key name
    case string
    case number
    case comment
}

/// A classified run, in **UTF-16 units** so it converts to an `NSRange` with no
/// arithmetic. Character offsets would be short by one for every emoji earlier
/// in the block and silently paint the wrong text from there on.
struct SyntaxSpan: Equatable, Sendable {
    let start: Int
    let length: Int
    let kind: SyntaxKind
}

/// Languages we color. Each maps a family of fence aliases onto one lexer
/// configuration — models write `ts`, `tsx`, `typescript` and `javascript` for
/// the same thing, and the fence is the only hint we get.
enum SyntaxLanguage: String, Sendable, CaseIterable {
    case swift
    case javascript
    case python
    case json
    case shell
    case markup
    case cFamily
    case zig
    case rust
    case go

    /// Resolve a fence label. Returns nil for an absent or unrecognized label —
    /// an unfenced block renders plain, because guessing a language and
    /// coloring prose as code reads far worse than no color at all.
    init?(fence: String) {
        let key = fence.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !key.isEmpty else { return nil }
        switch key {
        case "swift": self = .swift
        case "js", "jsx", "javascript", "mjs", "cjs",
             "ts", "tsx", "typescript": self = .javascript
        case "py", "python", "python3": self = .python
        case "json", "jsonc", "json5": self = .json
        case "sh", "bash", "zsh", "shell", "console", "fish": self = .shell
        case "html", "xml", "svg", "vue", "svelte", "xhtml": self = .markup
        case "c", "h", "cpp", "c++", "cc", "hpp", "cxx",
             "objc", "objective-c", "m", "mm",
             "java", "kotlin", "kt", "cs", "csharp", "c#": self = .cFamily
        case "zig": self = .zig
        case "rust", "rs": self = .rust
        case "go", "golang": self = .go
        default: return nil
        }
    }

}

/// Single-pass lexer producing `SyntaxSpan`s for a code block.
///
/// It is a LEXER, not a parser: it understands comments, string literals,
/// numbers and identifiers, and classifies identifiers by lookup plus two cheap
/// positional rules (capitalized ⇒ type, followed by `(` ⇒ call). That is
/// enough to make code scannable and cannot go badly wrong on a language it was
/// not tuned for.
///
/// Order matters and is the whole correctness story: comments are recognized
/// before strings (an apostrophe in a comment must not open a literal) and
/// strings before everything else (a `//` inside a URL must not start a
/// comment). Both classes run to end-of-source when unterminated, because a
/// streaming reply shows half-written code constantly.
enum SyntaxHighlighter {

    static func spans(_ source: String, language: SyntaxLanguage) -> [SyntaxSpan] {
        let u = Array(source.utf16)
        guard !u.isEmpty else { return [] }
        switch language {
        case .markup: return markupSpans(u)
        case .json:   return jsonSpans(u)
        default:      return codeSpans(u, spec: Spec.of(language))
        }
    }

    // MARK: - Character classes
    //
    // Every syntactic character in every language we handle is ASCII, so
    // scanning UTF-16 units is exact. Units >= 0x80 count as identifier
    // characters so an accented name stays one token instead of fragmenting.

    private static func isDigit(_ c: UInt16) -> Bool { c >= 0x30 && c <= 0x39 }
    private static func isHexDigit(_ c: UInt16) -> Bool {
        isDigit(c) || (c >= 0x41 && c <= 0x46) || (c >= 0x61 && c <= 0x66)
    }
    private static func isIdentStart(_ c: UInt16) -> Bool {
        (c >= 0x41 && c <= 0x5A) || (c >= 0x61 && c <= 0x7A)
            || c == 0x5F /* _ */ || c == 0x24 /* $ */ || c == 0x40 /* @ */ || c >= 0x80
    }
    private static func isIdentCont(_ c: UInt16) -> Bool { isIdentStart(c) || isDigit(c) }
    private static func isUpper(_ c: UInt16) -> Bool { c >= 0x41 && c <= 0x5A }
    private static func isSpace(_ c: UInt16) -> Bool {
        c == 0x20 || c == 0x09 || c == 0x0A || c == 0x0D
    }

    /// Does `pattern` (ASCII) occur at `i`?
    private static func matches(_ u: [UInt16], _ i: Int, _ pattern: [UInt16]) -> Bool {
        guard i + pattern.count <= u.count else { return false }
        for (k, p) in pattern.enumerated() where u[i + k] != p { return false }
        return true
    }

    private static func ascii(_ s: String) -> [UInt16] { Array(s.utf16) }

    // MARK: - Generic code lexer

    private struct Spec {
        var lineComments: [[UInt16]] = []
        var blockComment: (open: [UInt16], close: [UInt16])?
        /// Delimiters for literals that stop at end-of-line.
        var stringDelims: Set<UInt16> = []
        /// Delimiters for literals that may span lines (``"""``, `'''`, backtick).
        var multilineStrings: [[UInt16]] = []
        var keywords: Set<String> = []
        var types: Set<String> = []
        /// Whether a capitalized identifier should read as a type. True for
        /// languages whose convention is universal enough to rely on; false for
        /// shell, where `PATH` is a variable and coloring it as a type is noise.
        var capitalizedIsType: Bool = true

        static func of(_ lang: SyntaxLanguage) -> Spec {
            var s = Spec()
            switch lang {
            case .swift:
                s.lineComments = [ascii("//")]
                s.blockComment = (ascii("/*"), ascii("*/"))
                s.stringDelims = [0x22]
                s.multilineStrings = [ascii("\"\"\"")]
                s.keywords = Keywords.swift
                s.types = Keywords.swiftTypes
            case .javascript:
                s.lineComments = [ascii("//")]
                s.blockComment = (ascii("/*"), ascii("*/"))
                s.stringDelims = [0x22, 0x27]
                s.multilineStrings = [ascii("`")]
                s.keywords = Keywords.javascript
                s.types = Keywords.javascriptTypes
            case .python:
                s.lineComments = [ascii("#")]
                s.stringDelims = [0x22, 0x27]
                s.multilineStrings = [ascii("\"\"\""), ascii("'''")]
                s.keywords = Keywords.python
                s.types = Keywords.pythonTypes
            case .shell:
                s.lineComments = [ascii("#")]
                s.stringDelims = [0x22, 0x27]
                s.keywords = Keywords.shell
                s.capitalizedIsType = false
            case .cFamily:
                s.lineComments = [ascii("//")]
                s.blockComment = (ascii("/*"), ascii("*/"))
                s.stringDelims = [0x22, 0x27]
                s.keywords = Keywords.cFamily
                s.types = Keywords.cFamilyTypes
            case .zig:
                s.lineComments = [ascii("//")]
                s.stringDelims = [0x22, 0x27]
                s.keywords = Keywords.zig
                s.types = Keywords.zigTypes
            case .rust:
                s.lineComments = [ascii("//")]
                s.blockComment = (ascii("/*"), ascii("*/"))
                s.stringDelims = [0x22]
                s.keywords = Keywords.rust
                s.types = Keywords.rustTypes
            case .go:
                s.lineComments = [ascii("//")]
                s.blockComment = (ascii("/*"), ascii("*/"))
                s.stringDelims = [0x22, 0x27]
                s.multilineStrings = [ascii("`")]
                s.keywords = Keywords.go
                s.types = Keywords.goTypes
            case .json, .markup:
                break   // handled by dedicated lexers
            }
            return s
        }
    }

    private static func codeSpans(_ u: [UInt16], spec: Spec) -> [SyntaxSpan] {
        var out: [SyntaxSpan] = []
        var i = 0
        let n = u.count

        while i < n {
            // 1. Block comment — before strings, so a quote inside it is text.
            if let bc = spec.blockComment, matches(u, i, bc.open) {
                var j = i + bc.open.count
                while j < n, !matches(u, j, bc.close) { j += 1 }
                // Unterminated (still streaming) → run to the end.
                let end = j < n ? j + bc.close.count : n
                out.append(SyntaxSpan(start: i, length: end - i, kind: .comment))
                i = end
                continue
            }

            // 2. Line comment.
            if let lc = spec.lineComments.first(where: { matches(u, i, $0) }) {
                var j = i + lc.count
                while j < n, u[j] != 0x0A { j += 1 }
                out.append(SyntaxSpan(start: i, length: j - i, kind: .comment))
                i = j
                continue
            }

            // 3. Multiline string — tried before the single-line delimiters
            //    because `"""` starts with a `"`.
            if let ml = spec.multilineStrings.first(where: { matches(u, i, $0) }) {
                let end = scanDelimited(u, from: i, delimiter: ml, stopAtNewline: false)
                out.append(SyntaxSpan(start: i, length: end - i, kind: .string))
                i = end
                continue
            }

            // 4. Single-line string.
            if spec.stringDelims.contains(u[i]) {
                let end = scanDelimited(u, from: i, delimiter: [u[i]], stopAtNewline: true)
                out.append(SyntaxSpan(start: i, length: end - i, kind: .string))
                i = end
                continue
            }

            // 5. Number.
            if isDigit(u[i]) {
                let end = scanNumber(u, from: i)
                out.append(SyntaxSpan(start: i, length: end - i, kind: .number))
                i = end
                continue
            }

            // 6. Identifier — greedy, so digits inside a name never read as a
            //    number and `functional` never matches the keyword `func`.
            if isIdentStart(u[i]) {
                var j = i + 1
                while j < n, isIdentCont(u[j]) { j += 1 }
                let word = String(decoding: u[i..<j], as: UTF16.self)
                if let kind = classify(word, at: j, in: u, spec: spec) {
                    out.append(SyntaxSpan(start: i, length: j - i, kind: kind))
                }
                i = j
                continue
            }

            i += 1
        }
        return out
    }

    /// Classify a bare identifier. `end` is the index just past it, used for the
    /// call-site rule. Returns nil when the word carries no color.
    private static func classify(_ word: String, at end: Int, in u: [UInt16], spec: Spec) -> SyntaxKind? {
        if spec.keywords.contains(word) { return .keyword }
        if spec.types.contains(word) { return .type }
        if spec.capitalizedIsType, let f = word.utf16.first, isUpper(f) { return .type }
        // Call site: the next non-space character is an opening paren.
        var k = end
        while k < u.count, isSpace(u[k]) { k += 1 }
        if k < u.count, u[k] == 0x28 /* ( */ { return .function }
        return nil
    }

    /// Scan a delimited literal starting at its opening delimiter. Honors
    /// backslash escapes, and stops at a newline for the single-line forms so a
    /// missing close quote can't swallow the rest of the block. Returns the
    /// index just past the literal.
    private static func scanDelimited(_ u: [UInt16], from start: Int,
                                      delimiter: [UInt16], stopAtNewline: Bool) -> Int {
        let n = u.count
        var j = start + delimiter.count
        while j < n {
            if u[j] == 0x5C /* \ */ { j += 2; continue }
            if stopAtNewline, u[j] == 0x0A { return j }
            if matches(u, j, delimiter) { return min(j + delimiter.count, n) }
            j += 1
        }
        return n
    }

    /// `42`, `0xFF`, `0b1010`, `1_000`, `3.14`, `1e-9`. The exponent is consumed
    /// here rather than left to the identifier scanner, which would otherwise
    /// see the `e` of `1e-9` as a separate name.
    private static func scanNumber(_ u: [UInt16], from start: Int) -> Int {
        let n = u.count
        var j = start
        if u[j] == 0x30 /* 0 */, j + 1 < n,
           u[j + 1] == 0x78 || u[j + 1] == 0x58    // x X
            || u[j + 1] == 0x62 || u[j + 1] == 0x42 // b B
            || u[j + 1] == 0x6F || u[j + 1] == 0x4F // o O
        {
            j += 2
            while j < n, isHexDigit(u[j]) || u[j] == 0x5F { j += 1 }
            return j
        }
        while j < n, isDigit(u[j]) || u[j] == 0x5F { j += 1 }
        // Fractional part — only when a digit follows, so `1..2` and a method
        // call on a literal stay intact.
        if j + 1 < n, u[j] == 0x2E /* . */, isDigit(u[j + 1]) {
            j += 1
            while j < n, isDigit(u[j]) || u[j] == 0x5F { j += 1 }
        }
        // Exponent.
        if j < n, u[j] == 0x65 || u[j] == 0x45 /* e E */ {
            var k = j + 1
            if k < n, u[k] == 0x2B || u[k] == 0x2D { k += 1 }   // + -
            if k < n, isDigit(u[k]) {
                j = k
                while j < n, isDigit(u[j]) { j += 1 }
            }
        }
        return j
    }

    // MARK: - JSON
    //
    // Its own pass because a JSON string's meaning depends on what FOLLOWS it:
    // the same token is a key before a colon and a value anywhere else, and
    // painting both alike loses the structure that makes JSON skimmable.

    private static func jsonSpans(_ u: [UInt16]) -> [SyntaxSpan] {
        var out: [SyntaxSpan] = []
        var i = 0
        let n = u.count
        while i < n {
            if u[i] == 0x22 /* " */ {
                let end = scanDelimited(u, from: i, delimiter: [0x22], stopAtNewline: false)
                var k = end
                while k < n, isSpace(u[k]) { k += 1 }
                let isKey = k < n && u[k] == 0x3A /* : */
                out.append(SyntaxSpan(start: i, length: end - i, kind: isKey ? .property : .string))
                i = end
                continue
            }
            if isDigit(u[i]) || (u[i] == 0x2D /* - */ && i + 1 < n && isDigit(u[i + 1])) {
                let numStart = u[i] == 0x2D ? i + 1 : i
                let end = scanNumber(u, from: numStart)
                out.append(SyntaxSpan(start: i, length: end - i, kind: .number))
                i = end
                continue
            }
            if isIdentStart(u[i]) {
                var j = i + 1
                while j < n, isIdentCont(u[j]) { j += 1 }
                let word = String(decoding: u[i..<j], as: UTF16.self)
                if word == "true" || word == "false" || word == "null" {
                    out.append(SyntaxSpan(start: i, length: j - i, kind: .keyword))
                }
                i = j
                continue
            }
            i += 1
        }
        return out
    }

    // MARK: - Markup
    //
    // Tag-structured rather than token-structured: names and attributes only
    // mean anything INSIDE a tag, and `<!-- -->` must win over tag scanning or a
    // commented-out `<div>` lights up as live markup.

    private static func markupSpans(_ u: [UInt16]) -> [SyntaxSpan] {
        var out: [SyntaxSpan] = []
        var i = 0
        let n = u.count
        let commentOpen = ascii("<!--"), commentClose = ascii("-->")

        while i < n {
            if matches(u, i, commentOpen) {
                var j = i + commentOpen.count
                while j < n, !matches(u, j, commentClose) { j += 1 }
                let end = j < n ? j + commentClose.count : n
                out.append(SyntaxSpan(start: i, length: end - i, kind: .comment))
                i = end
                continue
            }
            guard u[i] == 0x3C /* < */ else { i += 1; continue }

            // Tag name, after an optional `/` or `!`.
            var j = i + 1
            while j < n, u[j] == 0x2F || u[j] == 0x21 { j += 1 }
            let nameStart = j
            while j < n, isIdentCont(u[j]) || u[j] == 0x2D || u[j] == 0x3A { j += 1 }
            if j > nameStart {
                out.append(SyntaxSpan(start: nameStart, length: j - nameStart, kind: .keyword))
            }

            // Attributes until the tag closes.
            while j < n, u[j] != 0x3E /* > */ {
                if u[j] == 0x22 || u[j] == 0x27 {
                    let end = scanDelimited(u, from: j, delimiter: [u[j]], stopAtNewline: false)
                    out.append(SyntaxSpan(start: j, length: end - j, kind: .string))
                    j = end
                    continue
                }
                if isIdentStart(u[j]) {
                    let attrStart = j
                    while j < n, isIdentCont(u[j]) || u[j] == 0x2D || u[j] == 0x3A { j += 1 }
                    out.append(SyntaxSpan(start: attrStart, length: j - attrStart, kind: .property))
                    continue
                }
                j += 1
            }
            i = min(j + 1, n)
        }
        return out
    }
}

/// Keyword tables. Kept deliberately to the words that carry meaning when
/// skimming — control flow, declarations, and the primitive types — rather than
/// every reserved word in each grammar.
private enum Keywords {
    static let swift: Set<String> = [
        "associatedtype", "class", "deinit", "enum", "extension", "fileprivate", "func", "import",
        "init", "inout", "internal", "let", "open", "operator", "private", "protocol", "public",
        "rethrows", "static", "struct", "subscript", "typealias", "var", "actor", "async", "await",
        "break", "case", "continue", "default", "defer", "do", "else", "fallthrough", "for", "guard",
        "if", "in", "repeat", "return", "switch", "where", "while", "as", "catch", "false", "is",
        "nil", "super", "self", "Self", "throw", "throws", "true", "try", "some", "any", "lazy",
        "weak", "unowned", "override", "final", "mutating", "nonisolated", "convenience", "required",
    ]
    static let swiftTypes: Set<String> = [
        "Int", "Double", "Float", "String", "Bool", "Character", "Array", "Dictionary", "Set",
        "Optional", "Result", "Data", "Date", "URL", "UUID", "Void", "Any", "AnyObject",
    ]

    static let javascript: Set<String> = [
        "async", "await", "break", "case", "catch", "class", "const", "continue", "debugger",
        "default", "delete", "do", "else", "export", "extends", "finally", "for", "from", "function",
        "get", "if", "import", "in", "instanceof", "let", "new", "of", "return", "set", "static",
        "super", "switch", "this", "throw", "try", "typeof", "var", "void", "while", "yield",
        "true", "false", "null", "undefined", "interface", "type", "enum", "implements", "public",
        "private", "protected", "readonly", "declare", "namespace", "satisfies", "as",
    ]
    static let javascriptTypes: Set<String> = [
        "string", "number", "boolean", "object", "symbol", "bigint", "unknown", "never", "any",
    ]

    static let python: Set<String> = [
        "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del", "elif",
        "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
        "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
        "True", "False", "None", "self", "match", "case",
    ]
    static let pythonTypes: Set<String> = [
        "int", "float", "str", "bool", "bytes", "list", "dict", "set", "tuple", "frozenset",
    ]

    static let shell: Set<String> = [
        "if", "then", "else", "elif", "fi", "for", "while", "until", "do", "done", "case", "esac",
        "function", "in", "return", "exit", "local", "export", "readonly", "declare", "set", "unset",
        "source", "echo", "cd", "test", "trap", "shift", "eval", "exec",
    ]

    static let cFamily: Set<String> = [
        "auto", "break", "case", "class", "const", "continue", "default", "do", "else", "enum",
        "extern", "for", "goto", "if", "inline", "namespace", "new", "delete", "operator", "private",
        "protected", "public", "register", "return", "sizeof", "static", "struct", "switch",
        "template", "this", "typedef", "typename", "union", "using", "virtual", "volatile", "while",
        "nullptr", "true", "false", "final", "override", "package", "implements", "extends",
        "import", "throws", "try", "catch", "finally", "throw", "abstract", "interface", "synchronized",
    ]
    static let cFamilyTypes: Set<String> = [
        "bool", "char", "double", "float", "int", "long", "short", "signed", "unsigned", "void",
        "size_t", "ssize_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "int8_t", "int16_t", "int32_t", "int64_t", "string", "var", "val",
    ]

    static let zig: Set<String> = [
        "align", "allowzero", "and", "anyframe", "anytype", "asm", "async", "await", "break",
        "catch", "comptime", "const", "continue", "defer", "else", "enum", "errdefer", "error",
        "export", "extern", "fn", "for", "if", "inline", "noalias", "nosuspend", "or", "orelse",
        "packed", "pub", "resume", "return", "struct", "suspend", "switch", "test", "threadlocal",
        "try", "union", "unreachable", "usingnamespace", "var", "volatile", "while",
        "true", "false", "null", "undefined",
    ]
    static let zigTypes: Set<String> = [
        "bool", "void", "type", "noreturn", "anyerror", "comptime_int", "comptime_float", "usize",
        "isize", "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128",
        "f16", "f32", "f64", "f128", "c_int", "c_uint", "c_char",
    ]

    static let rust: Set<String> = [
        "as", "async", "await", "break", "const", "continue", "crate", "dyn", "else", "enum",
        "extern", "false", "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move",
        "mut", "pub", "ref", "return", "self", "Self", "static", "struct", "super", "trait", "true",
        "type", "unsafe", "use", "where", "while",
    ]
    static let rustTypes: Set<String> = [
        "bool", "char", "str", "String", "u8", "u16", "u32", "u64", "u128", "usize",
        "i8", "i16", "i32", "i64", "i128", "isize", "f32", "f64", "Vec", "Option", "Result", "Box",
    ]

    static let go: Set<String> = [
        "break", "case", "chan", "const", "continue", "default", "defer", "else", "fallthrough",
        "for", "func", "go", "goto", "if", "import", "interface", "map", "package", "range",
        "return", "select", "struct", "switch", "type", "var", "nil", "true", "false",
    ]
    static let goTypes: Set<String> = [
        "bool", "byte", "complex64", "complex128", "error", "float32", "float64", "int", "int8",
        "int16", "int32", "int64", "rune", "string", "uint", "uint8", "uint16", "uint32", "uint64",
        "uintptr", "any",
    ]
}
