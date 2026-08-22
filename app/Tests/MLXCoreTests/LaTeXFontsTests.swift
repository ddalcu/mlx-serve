import XCTest

@testable import MLXCore

/// SwiftPM's generated `Bundle.module` accessor looks for the KaTeX font
/// bundle beside the .app, which is a location `codesign` refuses to seal — so
/// a hand-assembled bundle can only ever ship the fonts in Contents/Resources
/// and the app has to find them there itself (issue #233). A missing bundle
/// must cost the math, never the process.
final class LaTeXFontsTests: XCTestCase {
    private func present(_ paths: Set<String>) -> (URL) -> Bool {
        { paths.contains($0.path) }
    }

    func testFontsAreFoundInAnAppBundlesResourcesDirectory() {
        let resources = URL(fileURLWithPath: "/Applications/MLX Core.app/Contents/Resources")
        let bundle = resources.appendingPathComponent(LaTeXFonts.bundleName)
        let located = LaTeXFonts.locate(
            searching: [resources],
            fileExists: present([bundle.appendingPathComponent(LaTeXFonts.probeFont).path])
        )
        XCTAssertEqual(located, bundle)
    }

    func testAFontBundleWithoutItsFontsIsNotAHit() {
        let resources = URL(fileURLWithPath: "/Applications/MLX Core.app/Contents/Resources")
        XCTAssertNil(
            LaTeXFonts.locate(searching: [resources], fileExists: present([]))
        )
    }

    func testCandidatesAreSearchedInOrderAndTheFirstHitWins() {
        let resources = URL(fileURLWithPath: "/app/Contents/Resources")
        let root = URL(fileURLWithPath: "/app")
        let rootBundle = root.appendingPathComponent(LaTeXFonts.bundleName)
        let located = LaTeXFonts.locate(
            searching: [resources, root],
            fileExists: present([rootBundle.appendingPathComponent(LaTeXFonts.probeFont).path])
        )
        XCTAssertEqual(located, rootBundle)
    }

    /// The `swift build` layout: the resource bundle sits beside the binary,
    /// which is what `swift test` and `swift run` see.
    func testTheBuildDirectoryLayoutIsACandidate() {
        XCTAssertTrue(
            LaTeXFonts.searchLocations(
                resourceURL: URL(fileURLWithPath: "/x/Contents/Resources"),
                bundleURL: URL(fileURLWithPath: "/x")
            ).map(\.path).contains("/x"),
            "the .build/release layout keeps the bundle beside the executable's bundleURL"
        )
    }

    /// Under `swift test` the reading bundle is MLXCorePackageTests.xctest and
    /// the resource bundle is its sibling in .build/debug.
    func testTheTestBundlesSiblingLayoutIsACandidate() {
        XCTAssertTrue(
            LaTeXFonts.searchLocations(
                resourceURL: URL(fileURLWithPath: "/b/debug/T.xctest/Contents/Resources"),
                bundleURL: URL(fileURLWithPath: "/b/debug/T.xctest")
            ).map(\.path).contains("/b/debug"),
            "swift test keeps the resource bundle beside the .xctest, not inside it"
        )
    }

    func testTheFontsAreResolvableInThisTestRun() {
        XCTAssertTrue(
            LaTeXFonts.isAvailable,
            "the SwiftPM build layout must resolve, or every rendering test passes vacuously"
        )
    }

    func testInlineRenderingIsDeclinedWhenTheFontsAreMissing() {
        XCTAssertNil(
            InlineLaTeXRenderer.attributedAttachment(
                latex: "x^2",
                raw: "$x^2$",
                theme: .light,
                fontsAvailable: false
            )
        )
    }

    func testDisplayRenderingIsDeclinedWhenTheFontsAreMissing() {
        XCTAssertFalse(
            DisplayLaTeXRenderer.canRender(
                "x^2",
                theme: .light,
                fontSize: 14,
                fontsAvailable: false
            )
        )
    }
}
