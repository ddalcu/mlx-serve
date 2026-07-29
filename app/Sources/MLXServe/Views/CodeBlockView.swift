import SwiftUI
import AppKit

/// Colors for a rendered code block.
///
/// Every value is a DYNAMIC `NSColor`, resolved per appearance, because a chat
/// transcript is read in both light and dark mode and a block hard-coded for one
/// is unreadable in the other. Hues follow the Xcode/VS Code convention most
/// people already read code in, so the mapping needs no learning.
enum CodeTheme {

    private static func dynamic(light: NSColor, dark: NSColor) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
        })
    }

    /// Block background. Deliberately a small step off the surrounding surface
    /// rather than pure black/white — the block should read as inset, not as a
    /// hole punched in the transcript.
    static let background = dynamic(
        light: NSColor(red: 0.96, green: 0.96, blue: 0.97, alpha: 1),
        dark: NSColor(red: 0.11, green: 0.11, blue: 0.13, alpha: 1))
    static let header = dynamic(
        light: NSColor(red: 0.92, green: 0.92, blue: 0.94, alpha: 1),
        dark: NSColor(red: 0.15, green: 0.15, blue: 0.17, alpha: 1))
    static let border = dynamic(
        light: NSColor(white: 0.0, alpha: 0.10),
        dark: NSColor(white: 1.0, alpha: 0.10))
    static let plainText = dynamic(
        light: NSColor(white: 0.15, alpha: 1),
        dark: NSColor(white: 0.90, alpha: 1))
    static let gutter = dynamic(
        light: NSColor(white: 0.60, alpha: 1),
        dark: NSColor(white: 0.42, alpha: 1))

    static func color(for kind: SyntaxKind?) -> Color {
        switch kind {
        case .none: return plainText
        case .keyword: return dynamic(
            light: NSColor(red: 0.61, green: 0.11, blue: 0.55, alpha: 1),
            dark: NSColor(red: 0.78, green: 0.57, blue: 0.92, alpha: 1))
        case .type: return dynamic(
            light: NSColor(red: 0.06, green: 0.48, blue: 0.42, alpha: 1),
            dark: NSColor(red: 0.31, green: 0.81, blue: 0.69, alpha: 1))
        case .function: return dynamic(
            light: NSColor(red: 0.16, green: 0.36, blue: 0.75, alpha: 1),
            dark: NSColor(red: 0.51, green: 0.67, blue: 1.00, alpha: 1))
        case .property: return dynamic(
            light: NSColor(red: 0.63, green: 0.35, blue: 0.00, alpha: 1),
            dark: NSColor(red: 1.00, green: 0.80, blue: 0.42, alpha: 1))
        case .string: return dynamic(
            light: NSColor(red: 0.12, green: 0.48, blue: 0.24, alpha: 1),
            dark: NSColor(red: 0.65, green: 0.84, blue: 0.65, alpha: 1))
        case .number: return dynamic(
            light: NSColor(red: 0.70, green: 0.28, blue: 0.00, alpha: 1),
            dark: NSColor(red: 0.97, green: 0.55, blue: 0.42, alpha: 1))
        case .comment: return dynamic(
            light: NSColor(white: 0.43, alpha: 1),
            dark: NSColor(white: 0.52, alpha: 1))
        }
    }
}

/// Layout constants for a code block, shared with `CodeBlockLayoutTests` so the
/// gutter can't silently stop fitting its own numbers.
enum CodeBlockLayout {
    static let fontSize: CGFloat = 12
    static let cornerRadius: CGFloat = 10
    static let lineSpacing: CGFloat = 2.5

    /// Width of the line-number column. Monospaced digits are a fixed fraction
    /// of the point size, so the column is sized from the LARGEST number it has
    /// to show — a 4-digit file must not clip its own gutter.
    static func gutterWidth(lineCount: Int) -> CGFloat {
        let digits = max(2, String(max(lineCount, 1)).count)
        return CGFloat(digits) * fontSize * 0.62 + 18
    }
}

/// A fenced code block: language header with a copy button, a line-number
/// gutter, and syntax-colored code that scrolls horizontally.
///
/// Rendered as its own view rather than as a run inside the message's text view.
/// That costs cross-block drag-selection (each block is now its own selection
/// island) and buys the three things a flat monospaced run cannot have: stable
/// line numbers, per-token color, and a copy button that yields the code alone —
/// which is what people actually do with a code block. Prose either side still
/// selects in one motion, because `MarkdownSegmenter` keeps consecutive prose in
/// a single text view.
struct CodeBlockView: View {
    /// The fence label verbatim (`swift`, `tsx`, ``). Kept raw so the header can
    /// show what the model wrote when we don't recognize it.
    let language: String
    let code: String

    @State private var copied = false

    private var resolved: SyntaxLanguage? { SyntaxLanguage(fence: language) }

    /// Header label: our name for a language we color, else the model's own
    /// fence text, else nothing to claim.
    private var label: String {
        if let resolved { return resolved.displayName }
        let trimmed = language.trimmingCharacters(in: .whitespaces)
        return trimmed.isEmpty ? "Code" : trimmed
    }

    var body: some View {
        let lines = CodeLayout.lines(code: code, language: resolved)
        let gutter = CodeBlockLayout.gutterWidth(lineCount: lines.count)

        VStack(alignment: .leading, spacing: 0) {
            header
            Divider().opacity(0.5)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(alignment: .top, spacing: 0) {
                    // Gutter and code scroll TOGETHER. A pinned gutter drifts
                    // out of alignment the moment a row's height differs, and
                    // the numbers are the one thing that must never lie.
                    VStack(alignment: .trailing, spacing: CodeBlockLayout.lineSpacing) {
                        ForEach(lines) { line in
                            Text("\(line.number)")
                                .font(.system(size: CodeBlockLayout.fontSize, design: .monospaced))
                                .foregroundStyle(CodeTheme.gutter)
                        }
                    }
                    .frame(width: gutter, alignment: .trailing)
                    .padding(.trailing, 10)

                    VStack(alignment: .leading, spacing: CodeBlockLayout.lineSpacing) {
                        ForEach(lines) { line in
                            Text(attributed(line))
                                .font(.system(size: CodeBlockLayout.fontSize, design: .monospaced))
                                .textSelection(.enabled)
                                // Never wrap: a wrapped row would occupy two
                                // lines against one gutter number.
                                .fixedSize(horizontal: true, vertical: false)
                        }
                    }
                    .padding(.trailing, 14)
                }
                .padding(.vertical, 10)
            }
        }
        .background(CodeTheme.background)
        .clipShape(RoundedRectangle(cornerRadius: CodeBlockLayout.cornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: CodeBlockLayout.cornerRadius)
                .stroke(CodeTheme.border, lineWidth: 1)
        )
    }

    private var header: some View {
        HStack(spacing: 6) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            Spacer()
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(code, forType: .string)
                copied = true
                // The tick is the whole confirmation — a copy with no feedback
                // reads as a dead button and gets clicked again.
                Task {
                    try? await Task.sleep(nanoseconds: 1_400_000_000)
                    copied = false
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 10, weight: .medium))
                    Text(copied ? "Copied" : "Copy")
                        .font(.system(size: 10, weight: .medium))
                }
                .foregroundStyle(copied ? Color.green : Color.secondary)
                .padding(.horizontal, 6)
                .padding(.vertical, 3)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .help("Copy this code block")
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(CodeTheme.header)
    }

    private func attributed(_ line: CodeLine) -> AttributedString {
        var out = AttributedString()
        for run in line.runs {
            var piece = AttributedString(run.text)
            piece.foregroundColor = CodeTheme.color(for: run.kind)
            out.append(piece)
        }
        return out
    }
}
