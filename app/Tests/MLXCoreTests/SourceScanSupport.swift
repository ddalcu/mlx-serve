import Foundation

/// Support for the source scans — the tests that read a view's own text to pin
/// something the type system can't (which builder draws a heading, that a
/// control is still reachable at all).
///
/// A scan asserts on a SPELLING, so any edit that rewrites the spelling without
/// changing the meaning breaks it. Localization is exactly that edit: copy held
/// in a `String` is localized where it is written, so `AgentSection("Prompt")`
/// becomes `AgentSection(String(localized: "Prompt"))` and every scan looking
/// for the first form fails while the code it describes is unchanged. Six
/// assertions in `AgentEditorLayoutTests` and three in `ChatWorkspaceTests`
/// went red that way on the zh-Hans pass, and re-spelling each needle would put
/// the same tripwire back for the next file anyone localizes.
///
/// So the wrapper is normalized OUT before a scan reads the file: a scan is
/// about which call draws the copy, never about how the literal reaches it.
enum SourceScan {

    /// The file, with `String(localized: "…")` unwrapped back to `"…"`.
    ///
    /// Nested calls are handled by working outwards from each match, so
    /// `String(localized: "Workspace: \(x ?? String(localized: "not set"))")`
    /// reduces to the plain interpolated literal.
    static func normalizingLocalization(_ source: String) -> String {
        let marker = "String(localized: "
        var text = source
        while let call = text.range(of: marker) {
            // Find the `)` that closes this call, skipping the ones inside its
            // own string literal and any nested parentheses.
            var depth = 1
            var i = call.upperBound
            var inString = false
            var close: String.Index?
            while i < text.endIndex {
                let c = text[i]
                if c == "\\", inString, text.index(after: i) < text.endIndex {
                    i = text.index(i, offsetBy: 2)
                    continue
                }
                if c == "\"" { inString.toggle() }
                if !inString {
                    if c == "(" { depth += 1 }
                    if c == ")" {
                        depth -= 1
                        if depth == 0 { close = i; break }
                    }
                }
                i = text.index(after: i)
            }
            guard let close else { break }   // unbalanced: leave it alone
            text.replaceSubrange(close ... close, with: "")
            text.replaceSubrange(call, with: "")
        }
        return text
    }

    /// A file under the app root, normalized as above.
    static func read(_ relativePath: String, from filePath: StaticString = #filePath) throws -> String {
        let url = URL(fileURLWithPath: "\(filePath)")
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent(relativePath)
        return normalizingLocalization(try String(contentsOf: url, encoding: .utf8))
    }
}
