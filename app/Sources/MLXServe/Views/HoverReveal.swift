import SwiftUI

/// Short at rest, the whole thing floating over the pointer on hover.
///
/// Floating rather than growing in place, so nothing beside it moves when it
/// opens. `onContinuousHover` rather than `onHover`: the pointer's position is
/// the point of it.
struct HoverReveal<Full: View>: ViewModifier {
    enum Placement: Equatable {
        /// Over the pointer, growing right. For something small in a container
        /// that will not clip it.
        case pointer
        /// Over the pointer, but kept inside `container` (a rect in GLOBAL
        /// space) so a wide bubble cannot slide out of the column it belongs
        /// to, where a scroll view clips it and the next column draws over it.
        /// `width` is the bubble's own width, which is what makes the clamp
        /// exact.
        case pointerClamped(width: CGFloat, container: CGRect)
    }

    var placement: Placement = .pointer
    @ViewBuilder var full: () -> Full

    @State private var pointer: CGPoint?
    /// Measured: an alignment guide on overlay content does not survive
    /// further modifiers.
    @State private var height: CGFloat = 0
    /// This view's own leading edge in global space, to translate the clamp
    /// back into the local offset an overlay takes.
    @State private var originX: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .onGeometryChange(for: CGFloat.self) { $0.frame(in: .global).minX } action: { originX = $0 }
            .onContinuousHover { phase in
                switch phase {
                case .active(let location): pointer = location
                case .ended: pointer = nil
                }
            }
            .overlay(alignment: .topLeading) {
                if let pointer {
                    bubble(pointer)
                        .allowsHitTesting(false)
                        .transition(.opacity)
                }
            }
            // Later siblings draw over earlier ones, so without this whatever
            // comes next paints on top of the thing that just opened.
            .zIndex(pointer == nil ? 0 : 1)
    }

    @ViewBuilder
    private func bubble(_ pointer: CGPoint) -> some View {
        switch placement {
        case .pointer:
            // Above the pointer and slightly left of it, so the cursor never
            // sits on top of what it revealed.
            full().offset(x: pointer.x - 10, y: -24)
        case let .pointerClamped(width, container):
            full()
                .frame(width: width, alignment: .leading)
                .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { height = $0 }
                .offset(x: clampedX(pointer, width: width, container: container),
                        // Clear of the pointer, so it never covers its own text
                        // and the label underneath stays readable.
                        y: -(height + 10))
        }
    }

    /// Centred on the pointer, then pushed back inside the container.
    private func clampedX(_ pointer: CGPoint, width: CGFloat, container: CGRect) -> CGFloat {
        guard container.width > 0 else { return 0 }
        let wanted = originX + pointer.x - width / 2
        let highest = max(container.minX, container.maxX - width)
        return min(max(wanted, container.minX), highest) - originX
    }
}

extension View {
    /// Reveals `full` floating beside the pointer while this view is hovered.
    func hoverReveal<Full: View>(placement: HoverReveal<Full>.Placement = .pointer,
                                 @ViewBuilder _ full: @escaping () -> Full) -> some View {
        modifier(HoverReveal(placement: placement, full: full))
    }
}
