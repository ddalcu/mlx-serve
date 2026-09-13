import SwiftUI
import AppKit

/// An enlarge result laid over the picture SeedVR2 was handed, split by a
/// divider you drag: before on the left, after on the right.
///
/// Both halves draw into ONE frame, sized by the result. That is only honest
/// because the stored input is the restore's own canvas (`RestoreComparison`):
/// the two share pixel dimensions, so neither is stretched to meet the other,
/// and a line in one sits exactly over the same line in the other.
struct BeforeAfterSlider: View {
    let before: NSImage
    let after: NSImage

    /// Where the divider sits, as a fraction of the width. Survives picking
    /// another result in the strip on purpose: flipping through enlarges
    /// compares each one at the spot you were already looking at.
    @State private var position: CGFloat = 0.5

    var body: some View {
        Image(nsImage: after)
            .resizable()
            .interpolation(.high)
            .scaledToFit()
            .overlay {
                GeometryReader { geo in
                    let width = geo.size.width
                    let x = width * position
                    ZStack(alignment: .topLeading) {
                        Image(nsImage: before)
                            .resizable()
                            .interpolation(.high)
                            .frame(width: width, height: geo.size.height)
                            .mask(alignment: .leading) {
                                Rectangle().frame(width: x)
                            }
                        handle(x: x, height: geo.size.height)
                        badges
                    }
                    .contentShape(Rectangle())
                    // Distance 0: a click jumps the divider to the pointer, so
                    // "show me this corner" is one click, not a drag to it.
                    .gesture(
                        DragGesture(minimumDistance: 0).onChanged { drag in
                            position = RestoreComparison.fraction(x: drag.location.x, width: width)
                        }
                    )
                    .pointerStyle(.columnResize)
                }
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel("Before and after comparison")
            .accessibilityValue("\(Int((position * 100).rounded())) percent before")
            .accessibilityAdjustableAction { direction in
                switch direction {
                case .increment: position = min(1, position + 0.1)
                case .decrement: position = max(0, position - 0.1)
                @unknown default: break
                }
            }
    }

    /// The divider line plus a grip, drawn over both pictures. Never hit-tested:
    /// the whole frame is the drag target, not the 28pt circle.
    private func handle(x: CGFloat, height: CGFloat) -> some View {
        ZStack {
            Rectangle()
                .fill(Color.white)
                .frame(width: 2, height: height)
                .shadow(color: .black.opacity(0.45), radius: 1.5)
            Circle()
                .fill(.regularMaterial)
                .frame(width: 28, height: 28)
                .overlay(
                    Image(systemName: "arrow.left.and.right")
                        .font(.system(size: 11, weight: .semibold))
                )
                .shadow(color: .black.opacity(0.3), radius: 2)
        }
        .position(x: x, y: height / 2)
        .allowsHitTesting(false)
    }

    /// Each label fades out once its side is nearly gone, so a divider parked
    /// at an edge does not leave "Before" floating over the "after".
    private var badges: some View {
        HStack {
            badge("Before").opacity(position > 0.08 ? 1 : 0)
            Spacer()
            badge("After").opacity(position < 0.92 ? 1 : 0)
        }
        .padding(8)
        .animation(.easeOut(duration: 0.15), value: position > 0.08)
        .animation(.easeOut(duration: 0.15), value: position < 0.92)
        .allowsHitTesting(false)
    }

    private func badge(_ text: String) -> some View {
        Text(text)
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(.thinMaterial, in: Capsule())
    }
}
