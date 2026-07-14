import XCTest
@testable import MLXCore

/// Unit tests for the Settings sidebar — the category list, what's selectable
/// under which engine, and how selection and the search field interact.
///
/// SwiftUI can't be exercised directly, so (exactly as with `SettingsSearch`)
/// every *decision* lives in a pure model and the views only ask it questions.
///
/// The load-bearing invariant: **search text and a category selection never
/// coexist.** Typing forces the selection back to All; picking a category clears
/// the search. Otherwise a user lands on "Server" with a stale query still
/// narrowing it and sees an empty pane with no visible explanation.
final class SettingsCategoryTests: XCTestCase {

    // MARK: - Which categories exist for which engine

    /// The sidebar mirrors what the form actually renders — the engine-specific
    /// sections already hide themselves (flipping `--kv-quant` on a GGUF model
    /// silently no-ops), so offering them in the sidebar would be a dead end.
    func testMLXEngineHidesGgufAndDs4Categories() {
        let visible = SettingsCategory.visible(engine: .mlx, selfUpdate: true)
        XCTAssertTrue(visible.contains(.specDecode))
        XCTAssertTrue(visible.contains(.performance))
        XCTAssertFalse(visible.contains(.ggufPerformance))
        XCTAssertFalse(visible.contains(.ds4))
    }

    func testGgufEngineHidesMlxOnlyCategories() {
        let visible = SettingsCategory.visible(engine: .llama, selfUpdate: true)
        XCTAssertTrue(visible.contains(.ggufPerformance))
        XCTAssertFalse(visible.contains(.specDecode), "PLD/drafter/MTP are MLX-only kernels")
        XCTAssertFalse(visible.contains(.ds4))
    }

    func testDs4EngineShowsOnlyItsOwnEngineCategory() {
        let visible = SettingsCategory.visible(engine: .dsv4, selfUpdate: true)
        XCTAssertTrue(visible.contains(.ds4))
        XCTAssertFalse(visible.contains(.specDecode))
        XCTAssertFalse(visible.contains(.ggufPerformance))
    }

    /// Performance is ONE section now (the universal knobs merged with the
    /// MLX-only ones), so it's listed for every engine — its universal rows
    /// always apply, and only the MLX-only rows inside it come and go.
    func testPerformanceIsListedForEveryEngine() {
        for engine: ServerEngine? in [nil, .mlx, .llama, .dsv4] {
            XCTAssertTrue(SettingsCategory.visible(engine: engine, selfUpdate: true).contains(.performance),
                          "Performance must stay reachable on \(String(describing: engine))")
        }
    }

    /// No model loaded → every engine section renders so the user can pre-tune.
    /// The sidebar must offer them all, or those sections become unreachable.
    func testNoModelLoadedOffersEveryEngineCategory() {
        let visible = SettingsCategory.visible(engine: nil, selfUpdate: true)
        for c in [SettingsCategory.specDecode, .performance, .ggufPerformance, .ds4] {
            XCTAssertTrue(visible.contains(c), "\(c) must be reachable before a model loads")
        }
    }

    /// The Mac App Store updates the app itself — that section isn't built, so
    /// it must not be listed.
    func testUpdatesCategoryFollowsTheBuildFeature() {
        XCTAssertTrue(SettingsCategory.visible(engine: .mlx, selfUpdate: true).contains(.updates))
        XCTAssertFalse(SettingsCategory.visible(engine: .mlx, selfUpdate: false).contains(.updates))
    }

    /// Sidebar order must match the order the sections render in, or clicking
    /// down the list scrolls the form around at random.
    func testSidebarOrderMatchesRenderOrder() {
        XCTAssertEqual(SettingsCategory.visible(engine: .mlx, selfUpdate: true), [
            .modelFolders, .server, .specDecode, .performance,
            .requestDefaults, .voice, .sandbox, .messaging, .updates,
        ])
    }

    // MARK: - Selection ↔ search interaction (the invariant)

    func testTypingASearchForcesSelectionBackToAll() {
        XCTAssertEqual(SettingsSelection.afterQueryEdit(query: "prefix cache", current: .category(.server)), .all)
        XCTAssertEqual(SettingsSelection.afterQueryEdit(query: "kv", current: .category(.voice)), .all)
    }

    /// Clearing the field must NOT yank the user back to All — otherwise picking
    /// a category (which clears the search) would immediately undo itself.
    func testClearingTheSearchLeavesTheSelectionAlone() {
        XCTAssertEqual(SettingsSelection.afterQueryEdit(query: "", current: .category(.server)), .category(.server))
        XCTAssertEqual(SettingsSelection.afterQueryEdit(query: "   ", current: .category(.server)), .category(.server))
    }

    // MARK: - What a selection shows

    func testAllShowsEverySection() {
        for c in SettingsCategory.allCases {
            XCTAssertTrue(SettingsSelection.all.shows(c))
        }
    }

    func testACategoryShowsOnlyItself() {
        let sel = SettingsSelection.category(.server)
        XCTAssertTrue(sel.shows(.server))
        XCTAssertFalse(sel.shows(.voice))
        XCTAssertFalse(sel.shows(.performance))
    }

    // MARK: - Reconciliation

    /// Load a GGUF model while "Speculative Decoding" is selected and that
    /// category vanishes from the sidebar — the pane must not be left showing a
    /// section that no longer exists (or, worse, nothing at all).
    func testSelectionFallsBackToAllWhenItsCategoryDisappears() {
        let visible = SettingsCategory.visible(engine: .llama, selfUpdate: true)
        XCTAssertEqual(SettingsSelection.reconciled(.category(.specDecode), visible: visible), .all)
        XCTAssertEqual(SettingsSelection.reconciled(.category(.server), visible: visible), .category(.server),
                       "a still-visible category survives an engine switch")
        XCTAssertEqual(SettingsSelection.reconciled(.all, visible: visible), .all)
    }

    // MARK: - Titles

    /// The sidebar label and the section header come from ONE source, so they
    /// can't drift. The header keeps its long, explicit form; the sidebar gets a
    /// short one that fits the column.
    func testEveryCategoryHasATitleAndASidebarLabel() {
        for c in SettingsCategory.allCases {
            XCTAssertFalse(c.title.isEmpty)
            XCTAssertFalse(c.sidebarLabel.isEmpty)
            XCTAssertFalse(c.icon.isEmpty)
        }
        XCTAssertEqual(SettingsCategory.specDecode.title, "Speculative Decoding (MLX only)")
        XCTAssertEqual(SettingsCategory.specDecode.sidebarLabel, "Speculative Decoding")
    }

    // MARK: - Source audit (class guard)

    /// CLASS GUARD: every settings section must declare a `category:`, never a
    /// raw `title:` string. A section without a category has no sidebar entry —
    /// it would be silently unreachable whenever anything but "All Settings" is
    /// selected, and its title could drift from the sidebar's.
    func testEverySettingsSectionDeclaresACategory() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/SettingsView.swift")
        let body = try String(contentsOf: source, encoding: .utf8)

        // Declarations wrap: `SettingsSection(` on one line, `category:` on the
        // next — so look at the call, not a single line.
        let lines = body.split(separator: "\n", omittingEmptySubsequences: false).map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        var offenders: [String] = []
        for (i, l) in lines.enumerated() {
            if l.hasPrefix("//") || l.contains("struct ") { continue }
            guard l.contains("SettingsSection(") else { continue }
            let declaration = lines[i...min(i + 2, lines.count - 1)].joined(separator: " ")
            if !declaration.contains("category:") {
                offenders.append(declaration)
            }
        }
        XCTAssertTrue(offenders.isEmpty, """
            Every SettingsSection must be declared with `category:` so it has a \
            sidebar entry (and one source of truth for its title). Offenders:
            \(offenders.joined(separator: "\n"))
            """)
    }
}
