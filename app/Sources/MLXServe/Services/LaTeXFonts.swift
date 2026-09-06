import Foundation

private final class LaTeXFontsBundleFinder {}

/// Where the KaTeX font bundle actually lives in a hand-assembled .app.
///
/// SwiftPM generates SwaTexRender's `Bundle.module` as
/// `Bundle(path: Bundle.main.bundleURL + "SwaTex_SwaTexRender.bundle")` with a
/// build-directory fallback, and traps when neither exists. For an app bundle
/// `bundleURL` is the .app itself — a location `codesign` refuses to seal
/// ("unsealed contents present in the bundle root"), so the fonts can only ship
/// in Contents/Resources and both the patched dependency (see
/// `scripts/patch-swatex-font-lookup.sh`) and this type search for them.
enum LaTeXFonts {
    static let bundleName = "SwaTex_SwaTexRender.bundle"
    /// One font that must be inside the bundle: an empty directory of the right
    /// name is a half-finished copy, not a hit.
    static let probeFont = "Fonts/KaTeX_Main-Regular.ttf"
    /// The same font in the bundle shapes SwiftPM ships: flat (`swift build`
    /// on the classic build system) and Contents/Resources (the Xcode-style
    /// build system the Xcode 26 toolchain uses, `.build/out/Products`).
    /// `Bundle.url(forResource:)` reads both; the probe must too, or every
    /// formula renders as source text on the newer toolchain.
    static let probePaths = [probeFont, "Contents/Resources/" + probeFont]

    /// Contents/Resources for a real .app; the bundle URL covers the
    /// `swift build` layout, where the resource bundle sits beside the binary;
    /// its parent covers `swift test`, where the reading bundle is the
    /// .xctest and the resource bundle is its sibling.
    static func searchLocations(resourceURL: URL?, bundleURL: URL) -> [URL] {
        [resourceURL, bundleURL, bundleURL.deletingLastPathComponent()].compactMap { $0 }
    }

    static func locate(
        searching candidates: [URL],
        fileExists: (URL) -> Bool = { FileManager.default.fileExists(atPath: $0.path) }
    ) -> URL? {
        for base in candidates {
            let bundle = base.appendingPathComponent(bundleName)
            if probePaths.contains(where: { fileExists(bundle.appendingPathComponent($0)) }) { return bundle }
        }
        return nil
    }

    /// Resolved once. False means every LaTeX segment renders as its own source
    /// text — the app must never trap on a missing resource.
    static let isAvailable: Bool = {
        let own = Bundle(for: LaTeXFontsBundleFinder.self)
        let candidates = searchLocations(resourceURL: Bundle.main.resourceURL, bundleURL: Bundle.main.bundleURL)
            + searchLocations(resourceURL: own.resourceURL, bundleURL: own.bundleURL)
        return locate(searching: candidates) != nil
    }()
}
