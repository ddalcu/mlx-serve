import XCTest
@testable import MLXCore

/// The stage layer: what an artifact's page is given before the model's own
/// markup runs, and what it reports back.
///
/// Every assertion here is about the ONE complaint that motivated the rewrite —
/// a model-written page rendered as a mismatched slab inside the transcript
/// instead of as part of it.
final class HTMLArtifactRuntimeTests: XCTestCase {

    // MARK: - Colour reports

    func testAComputedColourRoundTrips() {
        let opaque = HTMLArtifactRuntime.parseCSSColor("rgb(15, 23, 42)")
        XCTAssertEqual(opaque?.color.red ?? -1, 15.0 / 255, accuracy: 0.001)
        XCTAssertEqual(opaque?.color.blue ?? -1, 42.0 / 255, accuracy: 0.001)
        XCTAssertEqual(opaque?.alpha ?? -1, 1, accuracy: 0.001)

        let transparent = HTMLArtifactRuntime.parseCSSColor("rgba(0, 0, 0, 0)")
        XCTAssertEqual(transparent?.alpha ?? -1, 0, accuracy: 0.001)

        // Anything that is not a computed colour must not become black.
        XCTAssertNil(HTMLArtifactRuntime.parseCSSColor("transparent"))
        XCTAssertNil(HTMLArtifactRuntime.parseCSSColor(""))
        XCTAssertNil(HTMLArtifactRuntime.parseCSSColor("rgb(a, b, c)"))
    }

    func testLuminanceDecidesLightFromDark() {
        XCTAssertTrue(HTMLArtifactRuntime.RGB(red: 0.06, green: 0.09, blue: 0.16).isDark)
        XCTAssertFalse(HTMLArtifactRuntime.RGB(red: 0.98, green: 0.98, blue: 0.99).isDark)
        // Green reads far brighter than blue at the same value — a plain average
        // calls #0000ff light.
        XCTAssertTrue(HTMLArtifactRuntime.RGB(red: 0, green: 0, blue: 1).isDark)
    }

    // MARK: - Which surface the card wears
    //
    // The whole point: a page that paints itself dark should make the BLOCK
    // dark, so the reader sees one intentional surface rather than a dark
    // rectangle sitting inside a light card.

    func testAPageThatPaintsItselfLendsTheCardItsColour() {
        let surface = HTMLArtifactRuntime.surface(background: "rgb(15, 23, 42)",
                                                  foreground: "rgb(226, 232, 240)",
                                                  hasBackgroundImage: false)
        XCTAssertNotNil(surface.fill)
        XCTAssertEqual(surface.chrome, .dark)
    }

    func testAPageThatPaintsNothingLeavesTheAppsOwnCardAlone() {
        let surface = HTMLArtifactRuntime.surface(background: "rgba(0, 0, 0, 0)",
                                                  foreground: "rgb(20, 20, 20)",
                                                  hasBackgroundImage: false)
        XCTAssertNil(surface.fill, "an unpainted fragment must float on the transcript's own card")
        XCTAssertEqual(surface.chrome, .app)
    }

    func testASemiTransparentBackgroundIsNotHoisted() {
        // Hoisting it would paint it twice — once by the card, once by the page
        // composited over the card — and the result is a colour neither side
        // chose.
        let surface = HTMLArtifactRuntime.surface(background: "rgba(15, 23, 42, 0.4)",
                                                  foreground: "rgb(20, 20, 20)",
                                                  hasBackgroundImage: false)
        XCTAssertNil(surface.fill)
    }

    func testAGradientKeepsPaintingItselfAndOnlyLendsItsMood() {
        // A gradient cannot be reduced to one fill, so the page keeps drawing
        // it edge to edge; all the card needs to know is which way the floating
        // controls should read.
        let surface = HTMLArtifactRuntime.surface(background: "rgba(0, 0, 0, 0)",
                                                  foreground: "rgb(240, 240, 245)",
                                                  hasBackgroundImage: true)
        XCTAssertNil(surface.fill)
        XCTAssertEqual(surface.chrome, .dark, "light text over a gradient means a dark surface")
    }

    // MARK: - The message the page posts back

    func testTheReportCarriesEverythingTheBlockNeeds() {
        let report = HTMLArtifactRuntime.report(from: [
            "h": 412.5,
            "bg": "rgb(15, 23, 42)",
            "fg": "rgb(226, 232, 240)",
            "img": false,
            "blocked": 2,
            "err": "draw is not defined",
        ])
        XCTAssertEqual(report?.height ?? 0, 412.5, accuracy: 0.001)
        XCTAssertEqual(report?.surface.chrome, .dark)
        XCTAssertEqual(report?.blockedRemoteLoads, 2)
        XCTAssertEqual(report?.scriptError, "draw is not defined")
    }

    func testAMalformedReportIsIgnoredRatherThanGuessedAt() {
        XCTAssertNil(HTMLArtifactRuntime.report(from: "413"))
        XCTAssertNil(HTMLArtifactRuntime.report(from: ["nothing": true]))
    }

    func testAnAbsurdHeightNeverReachesTheTranscript() {
        // A NaN frame does not break one block, it breaks the whole chat column.
        XCTAssertNil(HTMLArtifactRuntime.report(from: ["h": Double.nan])?.height)
        XCTAssertNil(HTMLArtifactRuntime.report(from: ["h": Double.infinity])?.height)
        XCTAssertNil(HTMLArtifactRuntime.report(from: ["h": -4])?.height)
    }

    func testAnErrorIsTrimmedAndNeverEmpty() {
        XCTAssertNil(HTMLArtifactRuntime.report(from: ["h": 10, "err": "   "])?.scriptError)
        XCTAssertEqual(HTMLArtifactRuntime.report(from: ["h": 10, "err": String(repeating: "x", count: 500)])?
                        .scriptError?.count, HTMLArtifactRuntime.maxDiagnosticLength)
    }

    // MARK: - What the reader is told

    func testBlockedRemoteLoadsAreNamedRatherThanRenderedAsABlankPage() {
        // The failure this replaces: a model reaches for a CDN chart library,
        // every load is blocked by design, and the reader gets an empty box
        // with no idea why.
        let text = HTMLArtifactRuntime.diagnostic(blockedRemoteLoads: 3, scriptError: nil)
        XCTAssertNotNil(text)
        XCTAssertTrue(text!.contains("3"))
        XCTAssertTrue(text!.lowercased().contains("offline"))

        XCTAssertEqual(HTMLArtifactRuntime.diagnostic(blockedRemoteLoads: 1, scriptError: nil)?
                        .contains("resources"), false, "one resource is not plural")
    }

    func testAScriptErrorOutranksTheOfflineNote() {
        let text = HTMLArtifactRuntime.diagnostic(blockedRemoteLoads: 2, scriptError: "x is not defined")
        XCTAssertEqual(text?.hasPrefix("x is not defined"), true)
        XCTAssertEqual(text?.contains("offline"), true)
    }

    func testASilentPageSaysNothing() {
        XCTAssertNil(HTMLArtifactRuntime.diagnostic(blockedRemoteLoads: 0, scriptError: nil))
    }

    // MARK: - The stage itself

    func testTheStageNeverReachesTheNetwork() {
        // Class guard, the same one the scaffold wears: every remote load is
        // blocked at the web view, so a font or reset stylesheet pulled in by
        // OUR layer would be the one request that had to be allowed.
        let staged = HTMLArtifactRuntime.stageScript(theme: .init(foreground: "#111",
                                                                  background: "#fff",
                                                                  accent: "#07f",
                                                                  dark: false))
            + HTMLArtifactRuntime.probeScript
        for needle in ["http://", "https://", "//fonts.", "<base", "url("] {
            XCTAssertFalse(staged.lowercased().contains(needle), "the stage reaches out via \(needle)")
        }
    }

    func testTheStageHandsThePageTheAppsOwnPalette() {
        // "Seamless with the chat" is a contract the model can actually write
        // against: the app's text, surface and accent colours arrive as custom
        // properties.
        let css = HTMLArtifactRuntime.stageScript(theme: .init(foreground: "#eaeaea",
                                                               background: "#1c1c1e",
                                                               accent: "#0a84ff",
                                                               dark: true))
        for token in ["--mlx-fg", "--mlx-bg", "--mlx-accent", "#eaeaea", "#0a84ff", "color-scheme"] {
            XCTAssertTrue(css.contains(token), "stage is missing \(token)")
        }
        XCTAssertTrue(css.contains("dark light"), "a dark app must not hand the page a light scheme first")
    }

    func testTheStageDefaultsCannotOutrankThePagesOwnCSS() {
        // The stage is a floor, not a costume: it is injected as the FIRST
        // stylesheet so anything the model wrote wins on cascade order. The one
        // exception is the viewport clamp, which has to win — see below.
        let css = HTMLArtifactRuntime.stageScript(theme: .init(foreground: "#111", background: "#fff",
                                                               accent: "#07f", dark: false))
        let defaults = css.components(separatedBy: HTMLArtifactRuntime.clampMarker).first ?? ""
        XCTAssertFalse(defaults.contains("!important"),
                       "a default that shouts is a default the model cannot override")
    }

    func testTheViewportClampIsTheOneRuleThatShouts() {
        // `body { min-height: 100vh }` is what a page written for a browser
        // window says, and inside a transcript it is a self-fulfilling
        // measurement: the frame is however tall we guessed, the page measures
        // exactly that, and the block is frozen at its placeholder height
        // forever. Removing the floor is the only way the page can report its
        // real size.
        let css = HTMLArtifactRuntime.stageScript(theme: .init(foreground: "#111", background: "#fff",
                                                               accent: "#07f", dark: false))
        let clamp = css.components(separatedBy: HTMLArtifactRuntime.clampMarker).last ?? ""
        XCTAssertTrue(clamp.contains("min-height: 0 !important"))
        XCTAssertTrue(clamp.contains("height: auto !important"))
    }

    func testTheProbeMeasuresTheContentRatherThanTheFrame() {
        // `documentElement` never reports less than the viewport, so a block
        // sized to it can grow and never shrink.
        XCTAssertTrue(HTMLArtifactRuntime.probeScript.contains("scrollHeight"))
        XCTAssertFalse(HTMLArtifactRuntime.probeScript.contains("documentElement.scrollHeight"))
        XCTAssertTrue(HTMLArtifactRuntime.probeScript.contains(HTMLArtifactRuntime.messageHandler))
    }

    func testTheProbeWatchesForLateLayoutAndForInteraction() {
        // A slider that reveals a row, a canvas drawn on load, a font swapping
        // in: all of them change the height AFTER the page has settled, and a
        // one-shot measurement leaves the block the wrong size for the rest of
        // the transcript's life.
        for hook in ["ResizeObserver", "MutationObserver", "load", "transitionend"] {
            XCTAssertTrue(HTMLArtifactRuntime.probeScript.contains(hook), "probe ignores \(hook)")
        }
    }
}
