import SwiftUI

/// What one composer control says when you hover it.
///
/// The Think / Tools / MCP / agent / paperclip controls are bare glyphs — the
/// captions came off when they moved out of the toolbar band — so state reads
/// from colour alone and the only place the WORDS live is here. A native
/// `.help` tooltip was carrying them, which meant a ~1.5 s wait, no formatting,
/// and no room for the two facts that actually save a trip into a menu (what a
/// right-click does, and which workspace the tools will run against).
///
/// Pure data on purpose: the card view has no seam worth testing, the sentences
/// do (`ComposerTipTests`).
struct ComposerTip: Equatable {
    /// One line, ~32 chars: the control's name and, for a toggle, its state.
    var title: String
    var body: String
    /// A dimmer trailing line for a runtime fact (the workspace path). Truncates
    /// rather than wrapping — a deep path would otherwise be most of the card.
    var detail: String? = nil

    /// Long enough that crossing the row on the way to Send doesn't flash five
    /// cards, short enough that a deliberate hover feels answered.
    static let hoverDelay: TimeInterval = 0.45

    static func agent(name: String?) -> ComposerTip {
        guard let name else {
            return ComposerTip(title: "Agent",
                               body: "Using the app's defaults. Click to pick one.")
        }
        return ComposerTip(title: "Agent · \(name)",
                           body: "Its prompt, tools, voice and model. Click to switch.")
    }

    static func attachments(audioSupported: Bool) -> ComposerTip {
        ComposerTip(
            title: "Attach",
            body: audioSupported
                ? "Image, PDF or audio — or a folder to ask questions about."
                : "Image or PDF — or a folder to ask questions about.")
    }

    static func thinking(isOn: Bool) -> ComposerTip {
        ComposerTip(title: "Thinking · \(state(isOn))",
                    body: "Reasoning trace before the answer. Click to turn it \(opposite(isOn)).")
    }

    static func tools(isOn: Bool, workspace: String?) -> ComposerTip {
        ComposerTip(
            title: "Tools · \(state(isOn))",
            body: "Shell, files, web, media. Click to turn it \(opposite(isOn)); right-click to pick tools and set the workspace.",
            detail: "Workspace: \(workspace ?? "not set")")
    }

    static func mcp(isOn: Bool) -> ComposerTip {
        ComposerTip(
            title: "MCP · \(state(isOn))",
            body: "Adds your enabled MCP servers' tools. Click to turn it \(opposite(isOn)); right-click for the Marketplace.")
    }

    private static func state(_ isOn: Bool) -> String { isOn ? "ON" : "OFF" }
    private static func opposite(_ isOn: Bool) -> String { isOn ? "off" : "on" }
}

// MARK: - Presentation

/// The hovered control's tip plus where it is, handed up to whoever draws it.
struct ComposerTipAnchor: Equatable {
    var tip: ComposerTip
    var bounds: Anchor<CGRect>
}

struct ComposerTipKey: PreferenceKey {
    static let defaultValue: ComposerTipAnchor? = nil
    static func reduce(value: inout ComposerTipAnchor?, nextValue: () -> ComposerTipAnchor?) {
        value = nextValue() ?? value
    }
}

extension View {
    /// Show `tip` when the pointer rests on this control.
    ///
    /// The control publishes; it does NOT draw. The composer container clips to
    /// its rounded rect, so a card overlaid here would be cut off at the
    /// container's edge and land on top of the text field — see
    /// `composerTipOverlay`, which renders it outside the clip.
    func composerTip(_ tip: ComposerTip) -> some View {
        modifier(ComposerTipHover(tip: tip))
    }

    /// Draw whichever control is currently hovered. Apply to the composer
    /// container AFTER its `clipShape`.
    func composerTipOverlay() -> some View {
        overlayPreferenceValue(ComposerTipKey.self) { anchor in
            GeometryReader { proxy in
                if let anchor {
                    let rect = proxy[anchor.bounds]
                    ComposerTipCard(tip: anchor.tip)
                        .frame(width: ComposerTipCard.width, alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                        // Leading-aligned on the control, clamped so a control
                        // near the right edge doesn't push the card off-window.
                        // (All five live at the left of the row, so leading is
                        // also the placement that never covers Send.)
                        .offset(x: max(0, min(rect.minX, proxy.size.width - ComposerTipCard.width)))
                        // A frame this tall puts the card's BOTTOM 8pt above the
                        // control; the card is free to overflow the frame — and
                        // the container — upward, which is the whole point.
                        .frame(width: proxy.size.width,
                               height: max(0, rect.minY - 8),
                               alignment: .bottomLeading)
                }
            }
            // Never a click target: the card sits over the controls it explains,
            // and the wrench is the most-flipped thing in the composer.
            .allowsHitTesting(false)
        }
    }
}

/// Publishes `tip` upward while the pointer rests on the control.
private struct ComposerTipHover: ViewModifier {
    let tip: ComposerTip
    @State private var shown = false
    /// Bumped on every hover change; the delayed reveal only fires if its own
    /// token is still current, which cancels a pointer that moved on.
    @State private var token = 0

    func body(content: Content) -> some View {
        content
            .onHover { inside in
                token &+= 1
                guard inside else {
                    shown = false
                    return
                }
                let mine = token
                DispatchQueue.main.asyncAfter(deadline: .now() + ComposerTip.hoverDelay) {
                    guard mine == token else { return }
                    withAnimation(.easeOut(duration: 0.12)) { shown = true }
                }
            }
            .anchorPreference(key: ComposerTipKey.self, value: .bounds) {
                shown ? ComposerTipAnchor(tip: tip, bounds: $0) : nil
            }
    }
}

struct ComposerTipCard: View {
    static let width: CGFloat = 250

    let tip: ComposerTip

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(tip.title)
                .font(.caption.weight(.semibold))
            Text(tip.body)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let detail = tip.detail {
                Text(detail)
                    .font(.caption2.monospaced())
                    .foregroundStyle(.tertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
        }
        .multilineTextAlignment(.leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(Color.secondary.opacity(0.25), lineWidth: 0.5)
        )
        .shadow(color: .black.opacity(0.18), radius: 8, y: 3)
    }
}
