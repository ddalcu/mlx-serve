import Foundation
import UniformTypeIdentifiers

/// A plain-text or source file attached to a chat: read off disk and sent in the user message as a fenced block
/// under its file name, so it works on every model (no tower needed).
enum TextAttachment {
    /// The largest file read. A book's worth of text; past it the folder index (attach a folder) is the tool.
    static let maxBytes = 2 * 1024 * 1024

    /// Source and config extensions the system has no text type for.
    private static let sourceExtensions: Set<String> = [
        "zig", "rs", "go", "kt", "kts", "ts", "tsx", "jsx", "mjs", "cjs", "vue", "svelte", "toml", "yml", "yaml", "ini", "cfg", "conf",
        "env", "lock", "gradle", "cmake", "mk", "proto", "graphql", "sql", "tf", "nix", "lua", "dart", "scala", "ex", "exs", "erl",
        "hs", "ml", "jl", "r", "metal", "glsl", "wgsl", "jinja", "j2", "tsv", "log", "markdown", "text",
    ]

    /// Whether a file with this extension is attached as text. RTF and PDF are text types whose bytes are not the
    /// text a person means to attach (PDF has its own extraction).
    /// The list wins over the system's type: `.ts` is TypeScript here, not a transport stream.
    static func isTextFile(ext: String) -> Bool {
        let e = ext.lowercased()
        if sourceExtensions.contains(e) { return true }
        return UTType(filenameExtension: e).map(isTextType) ?? false
    }

    /// A text type that is not also a picture (SVG) or one of the two whose bytes are not their text.
    static func isTextType(_ type: UTType) -> Bool {
        type.conforms(to: .text) && !type.conforms(to: .image) && !type.conforms(to: .rtf) && !type.conforms(to: .pdf)
    }

    /// What the attach panel offers: every text type, and the listed extensions as the types the system makes for them.
    static var contentTypes: [UTType] {
        [.text] + sourceExtensions.sorted().compactMap { UTType(filenameExtension: $0) }
    }

    /// The file's text. Nil when it is missing, empty, over `maxBytes`, or not text (a NUL byte, or bytes that are
    /// neither UTF-8 nor UTF-16 with a byte-order mark).
    static func read(_ url: URL) -> String? {
        guard let size = (try? url.resourceValues(forKeys: [.fileSizeKey]))?.fileSize, size > 0, size <= maxBytes,
              let data = try? Data(contentsOf: url) else { return nil }
        let utf16 = data.starts(with: [0xFF, 0xFE]) || data.starts(with: [0xFE, 0xFF])
        let text: String?
        if utf16 {
            text = String(data: data, encoding: .utf16)
        } else {
            text = data.contains(0) ? nil : String(data: data, encoding: .utf8)
        }
        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        return text
    }

    /// `[File: name]`, then the text inside a fence one backtick longer than the longest run in it.
    static func block(name: String, text: String) -> String {
        var longest = 0
        var run = 0
        for ch in text {
            run = ch == "`" ? run + 1 : 0
            longest = max(longest, run)
        }
        let fence = String(repeating: "`", count: max(3, longest + 1))
        return "[File: \(name)]\n\(fence)\n\(text)\(text.hasSuffix("\n") ? "" : "\n")\(fence)"
    }
}

/// One attachment the composer sends as text in the prompt instead of as media: a PDF's extracted text, or a text
/// file read off disk. `kind` picks the chip's label and how the model is told what it is.
struct PendingTextDoc: Equatable {
    enum Kind: Equatable { case pdf, text }

    let name: String
    let text: String
    let kind: Kind

    /// The documents as they precede the user's own words, in order.
    static func preamble(_ docs: [PendingTextDoc]) -> String {
        docs.map { $0.kind == .pdf ? "[PDF: \($0.name)]\n\($0.text)" : TextAttachment.block(name: $0.name, text: $0.text) }
            .joined(separator: "\n\n")
    }
}
