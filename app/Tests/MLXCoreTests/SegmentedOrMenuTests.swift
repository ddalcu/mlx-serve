import XCTest
import SwiftUI
@testable import MLXCore

final class SegmentedOrMenuTests: XCTestCase {
    private enum Tier: String, CaseIterable { case fast = "Fast", good = "Good", quality = "Quality", superQuality = "Super Quality" }

    /// The Create panes' shape: a padded, leading form column.
    @MainActor
    private func columnWidth(holding column: CGFloat) -> CGFloat {
        let picker = Picker("", selection: .constant(Tier.fast)) {
            ForEach(Tier.allCases, id: \.self) { Text($0.rawValue).tag($0) }
        }.labelsHidden()
        let form = VStack(alignment: .leading) { SegmentedOrMenu(picker) }
            .padding(16)
            .frame(maxWidth: .infinity, alignment: .leading)
        return NSHostingController(rootView: form).sizeThatFits(in: CGSize(width: column, height: 400)).width
    }

    // The bar: the picker never makes the form wider than the column it is given.
    @MainActor
    func testANarrowColumnGetsTheMenuNotAnOverflowingSegmentedControl() {
        XCTAssertLessThanOrEqual(columnWidth(holding: 200), 200)
    }

    @MainActor
    private func pickerWidth(offered width: CGFloat) -> CGFloat {
        NSHostingController(rootView: SegmentedOrMenu(
            Picker("", selection: .constant(Tier.fast)) {
                ForEach(Tier.allCases, id: \.self) { Text($0.rawValue).tag($0) }
            }.labelsHidden()
        )).sizeThatFits(in: CGSize(width: width, height: 400)).width
    }

    // Four labelled segments are far wider than a one-item menu.
    @MainActor
    func testAWideColumnKeepsTheSegments() {
        XCTAssertGreaterThan(pickerWidth(offered: 1000), 250)
    }

    // Segments are judged by the narrowest they draw, not their roomier ideal width.
    @MainActor
    func testAColumnThatHoldsTheSqueezedSegmentsKeepsThem() {
        let floor = NSHostingController(rootView: Picker("", selection: .constant(Tier.fast)) {
            ForEach(Tier.allCases, id: \.self) { Text($0.rawValue).tag($0) }
        }.labelsHidden().pickerStyle(.segmented)).sizeThatFits(in: CGSize(width: 0, height: 400)).width
        let ideal = pickerWidth(offered: 1000)
        XCTAssertLessThan(floor + 20, ideal)
        XCTAssertEqual(pickerWidth(offered: floor + 10), floor + 10, accuracy: 1)
        XCTAssertLessThan(pickerWidth(offered: floor - 10), floor - 10)
    }

    /// A control that reports its natural width instead of its floor on every other ask, like
    /// a squeezed segmented control on macOS 27.
    private final class AskCount { var asks = 0 }
    private struct FlakyFloor: Layout {
        let count: AskCount
        var update = 0
        let floor: CGFloat = 300, natural: CGFloat = 500
        func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
            guard let w = proposal.width, w.isFinite else { return CGSize(width: natural, height: 20) }
            if w < floor {
                count.asks += 1
                return CGSize(width: count.asks.isMultiple(of: 2) ? natural : floor, height: 20)
            }
            return CGSize(width: min(w, natural), height: 20)
        }
        func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {}
    }

    // The bar: a floor the control once reported keeps the segments; a later natural-width answer does not flip them to a menu.
    @MainActor
    func testTheChoiceHoldsWhenTheControlMisreportsItsFloor() {
        let count = AskCount()
        let host = NSHostingController(rootView: SegmentedOrMenu(FlakyFloor(count: count) { Color.clear }))
        let widths = (0..<8).map { update -> CGFloat in
            host.rootView = SegmentedOrMenu(FlakyFloor(count: count, update: update) { Color.clear })
            return host.sizeThatFits(in: CGSize(width: 400, height: 100)).width
        }
        XCTAssertEqual(Set(widths), [400], "\(widths)")
    }

    // A segmented control that reports more width than it was offered must not widen the column.
    @MainActor
    func testTheSegmentsNeverReportMoreWidthThanTheyAreOffered() {
        let rigid = MinimumWidthIsIdeal(floors: SegmentFloors()) { Color.clear.frame(width: 500, height: 20) }
        let width = NSHostingController(rootView: rigid).sizeThatFits(in: CGSize(width: 300, height: 400)).width
        XCTAssertLessThanOrEqual(width, 300)
    }
}
