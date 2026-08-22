import AppKit
import Foundation
import SwaTex
import SwaTexRender

extension NSAttributedString.Key {
    /// Exact model output represented by an inline math attachment. NSTextView
    /// copy uses it to put TeX on the pasteboard instead of U+FFFC.
    static let mlxLaTeXSource = NSAttributedString.Key("com.dalcu.mlx-core.latex-source")
}

enum LaTeXTheme: String, Sendable {
    case light
    case dark

    var mathColor: SwaTex.Color {
        switch self {
        case .light: .black
        case .dark: .white
        }
    }
}

/// Thin AppKit adapter around SwaTex. Parsing, TeX layout, KaTeX metrics, font
/// selection, and drawing all stay in the library; this type only turns the
/// resulting native image into an NSTextAttachment for the existing chat view.
enum InlineLaTeXRenderer {
    private static let maximumWidth: CGFloat = 2_048
    private static let maximumHeight: CGFloat = 512

    static func attributedAttachment(
        latex: String,
        raw: String,
        theme: LaTeXTheme,
        fontSize: CGFloat = ChatMetrics.transcriptFontSize,
        fontsAvailable: Bool = LaTeXFonts.isAvailable
    ) -> NSAttributedString? {
        // No KaTeX fonts means SwaTex would trap in its own resource-bundle
        // accessor (issue #233); the caller falls back to the exact source.
        guard fontsAvailable else { return nil }

        let list: DisplayList
        do {
            list = try SwaTexEngine.displayList(
                for: latex,
                style: .text,
                color: theme.mathColor,
                cache: .shared
            )
        } catch {
            return nil
        }

        let options = RenderOptions(fontSize: fontSize, padding: 0)
        let metrics = DisplayListRenderer.metrics(for: list, options: options)
        guard metrics.width.isFinite,
              metrics.height.isFinite,
              metrics.baseline.isFinite,
              metrics.width <= maximumWidth,
              metrics.height <= maximumHeight,
              let cgImage = ImageRenderer.image(
                  for: list,
                  options: options,
                  displayScale: 2
              ) else { return nil }

        let size = NSSize(width: metrics.width, height: metrics.height)
        let image = NSImage(cgImage: cgImage, size: size)
        image.accessibilityDescription = "LaTeX: \(latex)"

        let attachment = NSTextAttachment()
        attachment.image = image
        // NSTextAttachment's origin is relative to the surrounding text
        // baseline. SwaTex reports baseline-from-top, so the remaining height
        // is the descent below that baseline.
        attachment.bounds = NSRect(
            x: 0,
            y: -(metrics.height - metrics.baseline),
            width: metrics.width,
            height: metrics.height
        )

        let attributed = NSMutableAttributedString(attachment: attachment)
        attributed.addAttribute(
            .mlxLaTeXSource,
            value: raw,
            range: NSRange(location: 0, length: attributed.length)
        )
        return attributed
    }
}

enum DisplayLaTeXRenderer {
    private static let maximumWidth: CGFloat = 8_192
    private static let maximumHeight: CGFloat = 4_096

    /// Preflight MathView so malformed or pathological model output can fall
    /// back to its exact source rather than showing a renderer error surface.
    static func canRender(
        _ latex: String,
        theme: LaTeXTheme,
        fontSize: CGFloat,
        fontsAvailable: Bool = LaTeXFonts.isAvailable
    ) -> Bool {
        guard fontsAvailable else { return false }
        guard let list = try? SwaTexEngine.displayList(
            for: latex,
            style: .display,
            color: theme.mathColor,
            cache: .shared
        ) else { return false }
        let metrics = DisplayListRenderer.metrics(
            for: list,
            options: RenderOptions(fontSize: fontSize, padding: 0)
        )
        return metrics.width.isFinite
            && metrics.height.isFinite
            && metrics.width <= maximumWidth
            && metrics.height <= maximumHeight
    }
}

enum LaTeXCopyText {
    static func string(
        from attributed: NSAttributedString,
        range requestedRange: NSRange? = nil
    ) -> String {
        let available = NSRange(location: 0, length: attributed.length)
        let selectedRange = requestedRange.map { NSIntersectionRange($0, available) } ?? available
        guard selectedRange.length > 0 else { return "" }

        let selected = attributed.attributedSubstring(from: selectedRange)
        let output = NSMutableString(string: selected.string)
        var replacements: [(NSRange, String)] = []
        selected.enumerateAttribute(
            .mlxLaTeXSource,
            in: NSRange(location: 0, length: selected.length)
        ) { value, range, _ in
            if let raw = value as? String {
                replacements.append((range, raw))
            }
        }
        for (range, raw) in replacements.reversed() {
            output.replaceCharacters(in: range, with: raw)
        }
        return output as String
    }
}
