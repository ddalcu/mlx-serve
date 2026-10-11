import SwiftUI

/// A picker drawn as segments where its column can hold them, and as a menu
/// where it cannot. Pass the picker unstyled; this view picks the style.
///
/// A segmented control has a floor it will not squeeze below, and in a padded
/// form column that floor pushes the whole stack wider than the column. The
/// pane cannot find out by measuring its own section: the overflowing picker
/// is part of that measurement, so it reads back as room and never switches.
struct SegmentedOrMenu<P: View>: View {
    let picker: P
    init(_ picker: P) { self.picker = picker }

    var body: some View {
        // `ViewThatFits` judges a child by its IDEAL width; the segments are
        // judged by their floor instead, which is what has to fit.
        ViewThatFits(in: .horizontal) {
            MinimumWidthIsIdeal { picker.pickerStyle(.segmented) }
            picker.pickerStyle(.menu).fixedSize()
        }
    }
}

/// Reports its child's narrowest width as its ideal width, and never more
/// than the width it is offered: on macOS 27 a squeezed segmented control
/// reports its natural width again after an unrelated update (issue #773).
struct MinimumWidthIsIdeal: Layout {
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        guard let child = subviews.first else { return .zero }
        let size = child.sizeThatFits(ProposedViewSize(width: proposal.width ?? 0, height: proposal.height))
        return CGSize(width: min(size.width, proposal.width ?? size.width), height: size.height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        subviews.first?.place(at: bounds.origin, proposal: ProposedViewSize(bounds.size))
    }
}
