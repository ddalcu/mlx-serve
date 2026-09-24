import SwiftUI
import AppKit

/// The note waiting for the agent's next step: a faded user bubble above the
/// composer, where the message will land. Pause takes it back into the
/// composer.
struct SteeringNoteRow: View {
    let note: String
    let onPause: () -> Void

    @State private var contentHeight: CGFloat = 0
    /// The last line is fully in view (the note is scrolled to its end).
    @State private var atEnd = false
    @State private var scrolled: CGFloat = 0
    /// Shown while the note scrolls, gone 200 ms after it stops.
    @State private var knobShown = false
    @State private var scrollGeneration = 0

    /// Three quarters of the composer's Send button.
    private static let pauseIconSize = ChatMetrics.composerIconSize * 0.75
    private static let maxHeight: CGFloat = {
        let font = NSFont.preferredFont(forTextStyle: .body)
        return ceil(font.ascender - font.descender + font.leading) * 7
    }()
    private static let fadeHeight = maxHeight * 2 / 7
    private static let bubbleFill = Color.accentColor.opacity(0.8)

    private var viewportHeight: CGFloat { min(contentHeight, Self.maxHeight) }
    private var overflows: Bool { contentHeight > Self.maxHeight + 1 }

    var body: some View {
        HStack(alignment: .center, spacing: 8) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 5) {
                    Image(systemName: "clock.arrow.trianglehead.clockwise.rotate.90.path.dotted")
                        .symbolEffect(.rotate.clockwise.byLayer, options: .repeat(.continuous))
                    Text("About to send…")
                        .fontWeight(.bold)
                }
                .font(.caption2)
                .opacity(0.7)
                ScrollView(.vertical, showsIndicators: false) {
                    Text(note)
                        .font(.body)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.trailing, overflows ? 10 : 0)
                        .onGeometryChange(for: CGFloat.self) { $0.size.height } action: { contentHeight = $0 }
                        .onGeometryChange(for: Bool.self) {
                            $0.frame(in: .scrollView).maxY <= viewportHeight + 1
                        } action: { end in withAnimation(.easeInOut(duration: 0.15)) { atEnd = end } }
                        .onGeometryChange(for: CGFloat.self) {
                            -$0.frame(in: .scrollView).minY
                        } action: { offset in
                            guard offset != scrolled else { return }
                            scrolled = offset
                            knobShown = true
                            scrollGeneration += 1
                            let generation = scrollGeneration
                            Task { @MainActor in
                                try? await Task.sleep(for: .milliseconds(200))
                                if generation == scrollGeneration {
                                    withAnimation(.easeOut(duration: 0.15)) { knobShown = false }
                                }
                            }
                        }
                }
                .frame(height: viewportHeight)
                // The text itself fades: a coloured overlay would stack on the
                // translucent bubble. Scrolled to the end, the fade lifts.
                .mask(alignment: .top) {
                    VStack(spacing: 0) {
                        Color.black
                        if overflows {
                            LinearGradient(colors: [.black, .clear], startPoint: .top, endPoint: .bottom)
                                .overlay(Color.black.opacity(atEnd ? 1 : 0))
                                .frame(height: Self.fadeHeight)
                        }
                    }
                }
                // The system scroller's knob has no colour to set, so it is
                // drawn here from the offset, above the fade.
                .overlay(alignment: .topTrailing) {
                    if overflows && knobShown {
                        let knob = max(20, viewportHeight * viewportHeight / contentHeight)
                        Capsule()
                            .fill(.white.opacity(0.55))
                            .frame(width: 4, height: knob)
                            .offset(y: (viewportHeight - knob) * min(1, max(0, scrolled / (contentHeight - viewportHeight))))
                            .allowsHitTesting(false)
                    }
                }
            }
            .foregroundStyle(.white)
            .padding(.horizontal, ChatMetrics.bubblePaddingH)
            .padding(.vertical, ChatMetrics.bubblePaddingV)
            .background(Self.bubbleFill,
                        in: RoundedRectangle(cornerRadius: ChatMetrics.bubbleCornerRadius))
            Button(action: onPause) {
                Image(systemName: "arrow.down.circle.badge.pause")
                    .font(.system(size: Self.pauseIconSize))
                    .foregroundStyle(Color.accentColor)
            }
            .buttonStyle(.plain)
            .help("Take the note back into the composer; nothing is sent while you edit")
        }
        .frame(maxWidth: ChatMetrics.userBubbleMaxWidth, alignment: .trailing)
        .frame(maxWidth: .infinity, alignment: .trailing)
    }
}
