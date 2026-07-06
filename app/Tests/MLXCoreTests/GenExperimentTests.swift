import XCTest

@testable import MLXCore

/// Pins the tray's "Experiments" section model (`GenExperiment` in
/// StatusMenuView.swift). Historically this tested a hand-copied replica
/// because the target couldn't be @testable-imported; that's no longer true,
/// and the replica had already drifted from production — so it now tests the
/// real enum.
final class GenExperimentTests: XCTestCase {

    /// The section must contain exactly the media tools, in display order,
    /// with short tile titles (the tiles are narrow — "ImageGen"-style names
    /// didn't fit). Adding a gen feature without surfacing it here is the
    /// regression this pins (3D is the fourth).
    func testSectionHasExactlyTheMediaToolsInOrder() {
        XCTAssertEqual(GenExperiment.allCases.map(\.title),
                       ["Image", "Video", "Audio", "3D"])
    }

    /// Every tile has a non-empty tooltip naming its modality — the tile
    /// title is short, so the tooltip carries the detail.
    func testHelpTextPresentPerTool() {
        for e in GenExperiment.allCases {
            XCTAssertFalse(e.help.isEmpty, "\(e.title) needs a tooltip")
        }
    }

    /// Icons are the picker's visual identity — distinct per tool.
    func testIconsAreUnique() {
        let icons = GenExperiment.allCases.map(\.icon)
        XCTAssertEqual(Set(icons).count, icons.count, "Each experiment needs a distinct icon")
    }
}
