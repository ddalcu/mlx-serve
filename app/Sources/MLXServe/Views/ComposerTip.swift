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

    // `agent(name:)` retired with the composer's agent chip — the picker sits
    // next to New Chat now (a session's agent is fixed once it exists), and a
    // card for a control that no longer renders is a sentence nobody can reach.

    static func attachments(audioSupported: Bool, videoSupported: Bool = false) -> ComposerTip {
        let body: String
        switch (videoSupported, audioSupported) {
        case (true, true): body = "Image, video, PDF or audio — or a folder to ask questions about."
        case (true, false): body = "Image, video or PDF — or a folder to ask questions about."
        case (false, true): body = "Image, PDF or audio — or a folder to ask questions about."
        case (false, false): body = "Image or PDF — or a folder to ask questions about."
        }
        return ComposerTip(title: "Attach", body: body)
    }

    static func thinking(isOn: Bool, lockedBy agent: String? = nil) -> ComposerTip {
        ComposerTip(title: "Thinking · \(state(isOn))",
                    body: agent.map(locked)
                        ?? "Reasoning trace before the answer. Click to turn it \(opposite(isOn)); right-click to set effort.")
    }

    static func tools(isOn: Bool, workspace: String?, lockedBy agent: String? = nil) -> ComposerTip {
        ComposerTip(
            title: "Tools · \(state(isOn))",
            body: agent.map(locked)
                ?? "Shell, files, web, media. Click to turn it \(opposite(isOn)); right-click to pick tools and set the workspace.",
            // Still what every file and shell call resolves against, even when
            // it's the agent's folder rather than the chat's.
            detail: "Workspace: \(workspace ?? "not set")")
    }

    static func mcp(isOn: Bool, lockedBy agent: String? = nil) -> ComposerTip {
        ComposerTip(
            title: "MCP · \(state(isOn))",
            body: agent.map(locked)
                ?? "Adds your enabled MCP servers' tools. Click to turn it \(opposite(isOn)); right-click for the Marketplace.")
    }

    /// The body a control gets while its agent owns it. The title still carries
    /// ON/OFF — that's what the disc's colour is saying — but offering a click
    /// that can't happen is the dead-offer class, so this names the agent and
    /// where the setting actually lives instead.
    private static func locked(_ agent: String) -> String {
        "Set by the agent \(agent). Edit the agent to change it."
    }

    private static func state(_ isOn: Bool) -> String { isOn ? "ON" : "OFF" }
    private static func opposite(_ isOn: Bool) -> String { isOn ? "off" : "on" }
}

// MARK: - Hover lifecycle

/// When the card is up, extracted from the view so the one behaviour worth
/// pinning is testable: a dismiss must CANCEL a reveal that is already in
/// flight. Hover, then click before the delay elapses, and without the token
/// bump the card pops up on top of the menu you just opened.
struct ComposerTipHoverState: Equatable {
    private(set) var shown = false
    /// Bumped on every event; a delayed reveal only fires while its own token is
    /// still current.
    private(set) var token = 0

    /// The pointer entered — returns the token the delayed reveal must present.
    mutating func hoverBegan() -> Int {
        token &+= 1
        return token
    }

    mutating func hoverEnded() {
        token &+= 1
        shown = false
    }

    /// A menu opened over the composer (either mouse button, or press-and-hold).
    mutating func dismiss() {
        token &+= 1
        shown = false
    }

    /// The delayed reveal fires. False when something happened since — the
    /// pointer moved on, or a menu took over.
    @discardableResult
    mutating func reveal(token t: Int) -> Bool {
        guard t == token else { return false }
        shown = true
        return true
    }
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
    @State private var state = ComposerTipHoverState()

    func body(content: Content) -> some View {
        content
            .onHover { inside in
                guard inside else {
                    state.hoverEnded()
                    return
                }
                let mine = state.hoverBegan()
                DispatchQueue.main.asyncAfter(deadline: .now() + ComposerTip.hoverDelay) {
                    withAnimation(.easeOut(duration: 0.12)) { _ = state.reveal(token: mine) }
                }
            }
            // The card is a non-hit-testing overlay, so the only thing that takes
            // it down is the pointer leaving — and opening a menu over the
            // composer does NOT deliver a hover-exit, so the card sat under the
            // open menu until you hovered the control again. SwiftUI's `Menu` and
            // `.contextMenu` are both NSMenus, so this one observer covers a
            // left-click on the agent/attach menus, a right-click on the
            // Tools/MCP discs, and press-and-hold.
            .onReceive(NotificationCenter.default.publisher(for: NSMenu.didBeginTrackingNotification)) { _ in
                withAnimation(.easeOut(duration: 0.10)) { state.dismiss() }
            }
            .anchorPreference(key: ComposerTipKey.self, value: .bounds) {
                state.shown ? ComposerTipAnchor(tip: tip, bounds: $0) : nil
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
