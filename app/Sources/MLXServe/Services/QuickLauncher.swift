import AppKit
import SwiftUI
import Carbon.HIToolbox

// MARK: - Hotkey combo

/// The launcher's summon combo: ⌃Space. Fixed for v1 — note macOS binds
/// ⌃Space to "Select the previous input source" when multiple input sources
/// are enabled, and that system shortcut wins over any app registration
/// (RegisterEventHotKey still succeeds). The tray row's help text points
/// there when "nothing happens".
enum QuickLauncherHotKey {
    static let keyCode: UInt32 = UInt32(kVK_Space)
    static let carbonModifiers: UInt32 = UInt32(controlKey)
    static let display = "⌃Space"
}

// MARK: - Pure logic

/// Everything about the quick launcher that can be decided without AppKit —
/// same extraction pattern as `ComposerLayout` (the panel + Carbon hotkey are
/// untestable surfaces; this is the piece the unit tests pin).
enum QuickLauncherLogic {
    enum SubmitDecision: Equatable {
        case ignore                 // nothing to send
        case blocked(String)        // show the reason, don't submit
        case stopThenSubmit         // our own turn is mid-flight: supersede it
        case submit
    }

    /// The engine runs one turn at a time app-wide and `runTurn` silently
    /// no-ops while `isGenerating` — so a busy engine must either be stopped
    /// first (our own conversation: the new question supersedes) or block the
    /// submit (another chat's turn: never clobber it).
    static func submitDecision(text: String,
                               serverRunning: Bool,
                               composer: ChatTurnEngine.ComposerState) -> SubmitDecision {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return .ignore }
        guard serverRunning else {
            return .blocked("Server is not running — start it from the menu bar tray.")
        }
        switch composer {
        case .busyElsewhere: return .blocked("The model is answering another chat — try again in a moment.")
        case .generatingHere: return .stopThenSubmit
        case .idle: return .submit
        }
    }

    /// A launcher conversation rides a normal sidebar chat session; start a
    /// fresh one when there is none yet or the user deleted it underneath us.
    static func needsNewSession(current: UUID?, existing: [UUID]) -> Bool {
        guard let current else { return true }
        return !existing.contains(current)
    }

    /// Launcher turns are plain chat: no agent tools, no MCP, no thinking, no
    /// voice styling — a quick question deserves a fast, clean answer. The
    /// full agent belongs in the chat window ("Open in chat" is one keystroke).
    static func turnConfig() -> ChatTurnEngine.TurnConfig {
        ChatTurnEngine.TurnConfig(agentMode: false,
                                  mcpMode: false,
                                  enableThinking: false,
                                  voiceStyle: false,
                                  workingDirectory: nil)
    }

    // MARK: Geometry

    static let panelWidth: CGFloat = 680

    /// Two stable sizes (no per-token resize churn): input-only, and expanded
    /// once a conversation exists. `hasNotice` adds a row for blocked-submit
    /// messages so they don't clip in the compact state.
    static func panelHeight(hasConversation: Bool, hasNotice: Bool = false) -> CGFloat {
        (hasConversation ? 480 : 76) + (hasNotice ? 36 : 0)
    }

    /// Spotlight-style placement: horizontally centered, top edge pinned 25%
    /// down the screen. Cocoa coordinates (origin = bottom-left).
    static func panelOrigin(panelSize: CGSize, screenFrame: CGRect) -> CGPoint {
        let x = max(screenFrame.minX, screenFrame.midX - panelSize.width / 2)
        let top = screenFrame.maxY - screenFrame.height * 0.25
        return CGPoint(x: x, y: top - panelSize.height)
    }
}

// MARK: - Carbon hotkey wrapper

/// Minimal global-hotkey wrapper over Carbon's `RegisterEventHotKey` — the
/// classic launcher recipe. Needs NO Accessibility / Input-Monitoring TCC
/// grant and no entitlement (unlike NSEvent global monitors or a CGEventTap),
/// and is MAS-safe. The callback hops to the main queue before invoking
/// `onHotKey`.
final class HotKeyCenter {
    var onHotKey: (() -> Void)?
    private var hotKeyRef: EventHotKeyRef?
    private var handlerRef: EventHandlerRef?
    private static let hotKeyID = EventHotKeyID(signature: OSType(0x4D4C_5851), id: 1) // 'MLXQ'

    /// Returns false when the system refuses the registration (combo already
    /// taken by another app). Note a *system* shortcut on the same combo does
    /// not fail here — it just swallows the keystroke before us.
    @discardableResult
    func register(keyCode: UInt32, carbonModifiers: UInt32) -> Bool {
        unregister()
        var spec = EventTypeSpec(eventClass: OSType(kEventClassKeyboard),
                                 eventKind: UInt32(kEventHotKeyPressed))
        InstallEventHandler(GetEventDispatcherTarget(), { _, event, userData in
            var pressed = EventHotKeyID()
            GetEventParameter(event, EventParamName(kEventParamDirectObject),
                              EventParamType(typeEventHotKeyID), nil,
                              MemoryLayout<EventHotKeyID>.size, nil, &pressed)
            guard let userData,
                  pressed.signature == HotKeyCenter.hotKeyID.signature,
                  pressed.id == HotKeyCenter.hotKeyID.id else { return noErr }
            let center = Unmanaged<HotKeyCenter>.fromOpaque(userData).takeUnretainedValue()
            DispatchQueue.main.async { center.onHotKey?() }
            return noErr
        }, 1, &spec, Unmanaged.passUnretained(self).toOpaque(), &handlerRef)
        let status = RegisterEventHotKey(keyCode, carbonModifiers, Self.hotKeyID,
                                         GetEventDispatcherTarget(), 0, &hotKeyRef)
        return status == noErr && hotKeyRef != nil
    }

    func unregister() {
        if let hotKeyRef { UnregisterEventHotKey(hotKeyRef); self.hotKeyRef = nil }
        if let handlerRef { RemoveEventHandler(handlerRef); self.handlerRef = nil }
    }

    deinit { unregister() }
}

// MARK: - Panel

/// Borderless panels can't become key by default — override so the text field
/// gets keystrokes. `.nonactivatingPanel` keeps the frontmost app active
/// (Spotlight behavior): we take the keyboard, not the whole app.
final class QuickLauncherPanel: NSPanel {
    var onCancel: (() -> Void)?
    override var canBecomeKey: Bool { true }
    override func cancelOperation(_ sender: Any?) { onCancel?() }

    /// Standard editing shortcuts. While the panel is key the OWNING app is
    /// usually inactive (that's the point of `.nonactivatingPanel`), so no
    /// main menu is installed to translate ⌘V/⌘C/⌘X/⌘A into edit actions —
    /// route them down the responder chain by hand or paste simply doesn't
    /// work. SwiftUI's own shortcuts (⌘N/⌘↩/⌘.) still resolve via super.
    override func performKeyEquivalent(with event: NSEvent) -> Bool {
        if event.modifierFlags.intersection(.deviceIndependentFlagsMask) == .command {
            switch event.charactersIgnoringModifiers {
            case "v": if NSApp.sendAction(#selector(NSText.paste(_:)), to: nil, from: self) { return true }
            case "c": if NSApp.sendAction(#selector(NSText.copy(_:)), to: nil, from: self) { return true }
            case "x": if NSApp.sendAction(#selector(NSText.cut(_:)), to: nil, from: self) { return true }
            case "a": if NSApp.sendAction(#selector(NSResponder.selectAll(_:)), to: nil, from: self) { return true }
            default: break
            }
        }
        return super.performKeyEquivalent(with: event)
    }
}

// MARK: - Controller

/// Owns the ⌃Space hotkey and the floating prompt panel. App-level (AppState
/// holds it for the app's lifetime) and window-independent, like the voice
/// controller. Turns run through the shared `ChatTurnEngine` into a normal
/// sidebar chat session, so a quick answer is never lost — "Open in chat"
/// just focuses that session in the chat window.
@MainActor
final class QuickLauncherController: NSObject, ObservableObject, NSWindowDelegate {
    /// `unowned` because AppState owns the controller for the app's lifetime
    /// (same pattern as ChatTurnEngine.appState).
    unowned let appState: AppState

    /// The sidebar session the launcher is currently conversing in. Kept
    /// across summons so follow-up questions carry context; ⌘N starts fresh.
    @Published private(set) var sessionId: UUID?
    /// Blocked-submit explanation (server down / engine busy elsewhere).
    @Published private(set) var statusMessage: String?
    /// Bumped on every show so the view re-focuses the text field.
    @Published private(set) var focusTick = 0

    private let hotKey = HotKeyCenter()
    private var panel: QuickLauncherPanel?

    init(appState: AppState) {
        self.appState = appState
        super.init()
        hotKey.onHotKey = { [weak self] in
            Task { @MainActor in self?.toggle() }
        }
    }

    func setEnabled(_ on: Bool) {
        if on {
            hotKey.register(keyCode: QuickLauncherHotKey.keyCode,
                            carbonModifiers: QuickLauncherHotKey.carbonModifiers)
        } else {
            hotKey.unregister()
            hide()
        }
    }

    func toggle() {
        if panel?.isVisible == true { hide() } else { show() }
    }

    func show() {
        let panel = ensurePanel()
        updatePanelFrame(keepTopEdge: false)
        panel.makeKeyAndOrderFront(nil)
        panel.orderFrontRegardless()
        focusTick += 1
    }

    func hide() {
        panel?.orderOut(nil)
        statusMessage = nil
    }

    /// Forget the current conversation (the session stays in the chat
    /// sidebar); the next submit starts a new one.
    func newConversation() {
        sessionId = nil
        statusMessage = nil
        updatePanelFrame(keepTopEdge: true)
    }

    /// Focus the launcher's session in the chat window. The window itself is
    /// opened by the menu-bar label observing `quickLauncherChatOpenTick` —
    /// SwiftUI `Window` scenes can only be opened via the `openWindow`
    /// environment, and the always-installed label is the established bridge
    /// (see the task-notification deep-link).
    func openInChat() {
        if let sessionId { appState.activeChatId = sessionId }
        appState.quickLauncherChatOpenTick += 1
        hide()
    }

    /// Returns true when a turn was started (the view clears its field).
    @discardableResult
    func submit(_ rawText: String) -> Bool {
        let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        switch QuickLauncherLogic.submitDecision(text: text,
                                                 serverRunning: appState.server.status == .running,
                                                 composer: composerState()) {
        case .ignore:
            return false
        case .blocked(let why):
            statusMessage = why
            updatePanelFrame(keepTopEdge: true)
            return false
        case .stopThenSubmit:
            appState.chatEngine.stop()
            startTurn(text)
            return true
        case .submit:
            startTurn(text)
            return true
        }
    }

    private func composerState() -> ChatTurnEngine.ComposerState {
        guard let sessionId else {
            return appState.chatEngine.isGenerating ? .busyElsewhere : .idle
        }
        return appState.chatEngine.composerState(for: sessionId)
    }

    private func startTurn(_ text: String) {
        statusMessage = nil
        let sid: UUID
        if QuickLauncherLogic.needsNewSession(current: sessionId,
                                              existing: appState.chatSessions.map(\.id)) {
            sid = appState.newChatSession()
            sessionId = sid
        } else {
            sid = sessionId! // needsNewSession(false) implies non-nil
        }
        appState.chatEngine.runTurn(sessionId: sid, userText: text, images: nil, audio: nil,
                                    config: QuickLauncherLogic.turnConfig(),
                                    approval: { _ in false })
        updatePanelFrame(keepTopEdge: true)
    }

    // MARK: Panel plumbing

    private func ensurePanel() -> QuickLauncherPanel {
        if let panel { return panel }
        let size = CGSize(width: QuickLauncherLogic.panelWidth,
                          height: QuickLauncherLogic.panelHeight(hasConversation: false))
        let p = QuickLauncherPanel(contentRect: NSRect(origin: .zero, size: size),
                                   styleMask: [.borderless, .nonactivatingPanel],
                                   backing: .buffered, defer: false)
        p.isFloatingPanel = true
        p.level = .floating
        p.backgroundColor = .clear
        p.isOpaque = false
        p.hasShadow = true
        p.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary, .transient]
        p.hidesOnDeactivate = false
        p.isMovableByWindowBackground = true
        p.isReleasedWhenClosed = false
        p.becomesKeyOnlyIfNeeded = false
        p.delegate = self
        p.onCancel = { [weak self] in self?.hide() }

        // Rounded translucent card: NSVisualEffectView backing + the SwiftUI
        // content pinned inside. The window itself is clear.
        let effect = NSVisualEffectView()
        effect.material = .popover
        effect.state = .active
        effect.blendingMode = .behindWindow
        effect.wantsLayer = true
        effect.layer?.cornerRadius = 16
        effect.layer?.cornerCurve = .continuous
        effect.layer?.masksToBounds = true

        let hosting = NSHostingView(rootView: AnyView(
            QuickLauncherView(controller: self, engine: appState.chatEngine)
                .environmentObject(appState)
        ))
        hosting.translatesAutoresizingMaskIntoConstraints = false
        effect.addSubview(hosting)
        NSLayoutConstraint.activate([
            hosting.topAnchor.constraint(equalTo: effect.topAnchor),
            hosting.bottomAnchor.constraint(equalTo: effect.bottomAnchor),
            hosting.leadingAnchor.constraint(equalTo: effect.leadingAnchor),
            hosting.trailingAnchor.constraint(equalTo: effect.trailingAnchor),
        ])
        p.contentView = effect
        panel = p
        return p
    }

    /// Reposition/resize the panel. On summon it centers on the screen under
    /// the mouse (Spotlight opens on the active screen; the cursor is the
    /// closest proxy); on state changes while visible it keeps the top edge
    /// pinned so growth extends downward.
    private func updatePanelFrame(keepTopEdge: Bool) {
        guard let panel else { return }
        let height = QuickLauncherLogic.panelHeight(hasConversation: sessionId != nil,
                                                    hasNotice: statusMessage != nil)
        let size = CGSize(width: QuickLauncherLogic.panelWidth, height: height)
        var frame = NSRect(origin: panel.frame.origin, size: size)
        if keepTopEdge, panel.isVisible {
            frame.origin.y = panel.frame.maxY - height
        } else if let screen = targetScreen() {
            frame.origin = QuickLauncherLogic.panelOrigin(panelSize: size,
                                                          screenFrame: screen.visibleFrame)
        }
        panel.setFrame(frame, display: true)
    }

    private func targetScreen() -> NSScreen? {
        let mouse = NSEvent.mouseLocation
        return NSScreen.screens.first { NSMouseInRect(mouse, $0.frame, false) }
            ?? NSScreen.main ?? NSScreen.screens.first
    }

    // MARK: NSWindowDelegate

    /// Transient like Spotlight: clicking anywhere else dismisses the panel.
    /// Generation, if in flight, continues in the sidebar session.
    nonisolated func windowDidResignKey(_ notification: Notification) {
        Task { @MainActor in self.hide() }
    }
}
