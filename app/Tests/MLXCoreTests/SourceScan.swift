import Foundation

/// Shared machinery for the source-scan guards.
///
/// Every scan in this target has the same first problem: the thing it greps for
/// is also the thing the code COMMENTS about. `DialogDefaultActionTests` trips
/// over a comment in SandboxTerminalView that discusses `.alert(`; the Escape
/// guard counted a doc comment explaining why there are only two
/// `.keyboardShortcut(.cancelAction)` claims as a third claim. One stripper, so
/// the next scan starts from code.
enum SourceScan {

    /// Replace comments with spaces, leaving string literals — and every
    /// offset — intact.
    static func strippingComments(_ source: String) -> String {
        var out = ""
        out.reserveCapacity(source.count)
        let chars = Array(source)
        var i = 0
        var inString = false, inLine = false, inBlock = false
        while i < chars.count {
            let c = chars[i]
            let next: Character? = i + 1 < chars.count ? chars[i + 1] : nil
            if inLine {
                if c == "\n" { inLine = false; out.append(c) } else { out.append(" ") }
            } else if inBlock {
                if c == "*", next == "/" { inBlock = false; out += "  "; i += 2; continue }
                out.append(c == "\n" ? c : " ")
            } else if inString {
                out.append(c)
                if c == "\\", let n = next { out.append(n); i += 2; continue }
                if c == "\"" { inString = false }
            } else if c == "/", next == "/" {
                inLine = true; out += "  "; i += 2; continue
            } else if c == "/", next == "*" {
                inBlock = true; out += "  "; i += 2; continue
            } else {
                if c == "\"" { inString = true }
                out.append(c)
            }
            i += 1
        }
        return out
    }

    /// How many times `needle` occurs.
    static func count(_ needle: String, in source: String) -> Int {
        source.components(separatedBy: needle).count - 1
    }

    /// From `marker` to the end of the declaration it opens — the first line
    /// that closes at a method's own indentation.
    ///
    /// Use this rather than `prefix(n)` on the remainder: `strippingComments`
    /// preserves every OFFSET (it blanks comment characters rather than
    /// removing them), so a fixed character window shrinks as soon as someone
    /// writes a long explanation inside the function, and the scan starts
    /// failing for a reason that has nothing to do with what it checks.
    /// Returns the rest of the file when no close is found, which keeps a
    /// mis-anchored scan loud rather than vacuous.
    static func declarationBody(from marker: String, in source: String) -> String? {
        guard let start = source.range(of: marker) else { return nil }
        let rest = String(source[start.lowerBound...])
        guard let end = rest.range(of: "\n    }\n") else { return rest }
        return String(rest[..<end.upperBound])
    }

    /// A file under `app/Sources/MLXServe`, comments stripped. `file` is the
    /// path relative to that root.
    static func source(_ file: String, from testFilePath: String) -> String {
        let url = URL(fileURLWithPath: testFilePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/\(file)")
        return strippingComments((try? String(contentsOf: url, encoding: .utf8)) ?? "")
    }
}
