import XCTest
@testable import MLXCore

/// Pins the Discover table's width→column-tier decision.
///
/// The bug this guards: the table's fixed columns cost more than 600pt, so at
/// the old 720pt detail floor the flexible Model column got ~60pt — every name
/// truncated to "gem…" and badge text wrapped one character per line, blowing
/// row heights up ~300pt. The fix: WE decide column visibility from the
/// measured pane width, and narrow panes drop low-value columns instead of
/// starving the Model column.
final class ModelBrowserMetricsTests: XCTestCase {

    /// Width 0 = not measured yet → assume roomy.
    func testUnmeasuredWidthAssumesRoomy() {
        XCTAssertEqual(ModelBrowserMetrics.tier(forDetailWidth: 0), .full)
    }

    func testWidePaneShowsEveryColumn() {
        let tier = ModelBrowserMetrics.tier(forDetailWidth: 1200)
        XCTAssertEqual(tier, .full)
        XCTAssertTrue(tier.showsPulls)
        XCTAssertTrue(tier.showsLikes)
        XCTAssertTrue(tier.showsUpdated)
    }

    /// Tiers degrade stepwise: just below the full threshold the pane is
    /// medium (Likes + Updated gone), just below the medium threshold it is
    /// compact (Pulls gone too).
    func testTiersDegradeStepwiseAsThePaneNarrows() {
        let fullFloor = ModelBrowserMetrics.fixedWidth(for: .full)
            + ModelBrowserMetrics.modelColumnMinWidth
        let mediumFloor = ModelBrowserMetrics.fixedWidth(for: .medium)
            + ModelBrowserMetrics.modelColumnMinWidth

        XCTAssertEqual(ModelBrowserMetrics.tier(forDetailWidth: fullFloor), .full)
        XCTAssertEqual(ModelBrowserMetrics.tier(forDetailWidth: fullFloor - 1), .medium)
        XCTAssertEqual(ModelBrowserMetrics.tier(forDetailWidth: mediumFloor), .medium)
        XCTAssertEqual(ModelBrowserMetrics.tier(forDetailWidth: mediumFloor - 1), .compact)
    }

    /// THE invariant that kills the "gem…" bug: at every width the pane can
    /// actually be (>= its floor), the tier chosen leaves the flexible Model
    /// column at least `compactModelColumnMinWidth`. Swept, not spot-checked,
    /// so a future column width bump that silently breaks a threshold fails
    /// here.
    func testModelColumnKeepsAReadableMinimumAtEveryLegalWidth() {
        var w = ModelBrowserMetrics.minDetailWidth
        while w <= 1400 {
            let tier = ModelBrowserMetrics.tier(forDetailWidth: w)
            let modelColumn = w - ModelBrowserMetrics.fixedWidth(for: tier)
            XCTAssertGreaterThanOrEqual(
                modelColumn, ModelBrowserMetrics.compactModelColumnMinWidth,
                "at width \(w) tier \(tier) leaves only \(modelColumn)pt for the Model column"
            )
            w += 1
        }
    }

    /// Column visibility nests: anything shown at a narrower tier is shown at
    /// every wider tier — the pane only ever loses columns as it shrinks,
    /// never swaps them.
    func testColumnVisibilityIsMonotonic() {
        let ordered: [ModelBrowserMetrics.Tier] = [.compact, .medium, .full]
        for (narrower, wider) in zip(ordered, ordered.dropFirst()) {
            for keyPath in [\ModelBrowserMetrics.Tier.showsPulls,
                            \.showsLikes, \.showsUpdated] {
                if narrower[keyPath: keyPath] {
                    XCTAssertTrue(wider[keyPath: keyPath],
                                  "\(wider) hides a column that \(narrower) shows")
                }
            }
        }
    }

    /// The pane floor is self-consistent: it resolves to the compact tier and
    /// fits the compact fixed columns plus the reduced Model minimum. This is
    /// what the DiscoverPane's `.frame(minWidth:)` uses, so the window can
    /// never be resized into the clipped state in the bug screenshot.
    func testMinDetailWidthFitsTheCompactTier() {
        let floor = ModelBrowserMetrics.minDetailWidth
        XCTAssertEqual(ModelBrowserMetrics.tier(forDetailWidth: floor), .compact)
        XCTAssertEqual(
            floor - ModelBrowserMetrics.fixedWidth(for: .compact),
            ModelBrowserMetrics.compactModelColumnMinWidth
        )
    }

    /// The fixed cost is honest: each degradation step actually shrinks it
    /// (otherwise dropping the column bought nothing).
    func testFixedWidthShrinksWithEachTier() {
        let full = ModelBrowserMetrics.fixedWidth(for: .full)
        let medium = ModelBrowserMetrics.fixedWidth(for: .medium)
        let compact = ModelBrowserMetrics.fixedWidth(for: .compact)
        XCTAssertGreaterThan(full, medium)
        XCTAssertGreaterThan(medium, compact)
    }

    /// The Model Browser window must OPEN wide enough for the full tier:
    /// sidebar ideal width + full-tier detail. The old 900×600 default gave
    /// the detail ~695pt — born squeezed, which is the "initial window seems
    /// bad" half of the bug.
    func testDefaultWindowWidthFitsTheFullTierBesideTheSidebar() {
        let fullDetail = ModelBrowserMetrics.fixedWidth(for: .full)
            + ModelBrowserMetrics.modelColumnMinWidth
        XCTAssertGreaterThanOrEqual(
            ModelBrowserMetrics.defaultWindowWidth,
            fullDetail + ModelBrowserMetrics.sidebarIdealWidth
        )
    }
}
