import AppKit
import SwiftUI
import Carbon.HIToolbox

// MARK: - Hotkey combo

/// The launcher's summon combo. Defaults to ⌃Space; Settings ▸ Interface can
/// rebind it (`QuickLauncherHotKeyStore`) — some keyboard layouts and input
/// methods already claim ⌃Space for "Select the previous input source", and
/// that system shortcut wins over any app registration (RegisterEventHotKey
/// still succeeds; the tray row's help text points there when "nothing
/// happens").
enum QuickLauncherHotKey {
    static let defaultKeyCode: UInt32 = UInt32(kVK_Space)
    static let defaultCarbonModifiers: UInt32 = UInt32(controlKey)

    static var keyCode: UInt32 { QuickLauncherHotKeyStore.keyCode }
    static var carbonModifiers: UInt32 { QuickLauncherHotKeyStore.carbonModifiers }
    static var display: String { HotKeyDisplay.string(keyCode: keyCode, carbonModifiers: carbonModifiers) }
}

/// Persisted override for the combo above. Two raw `Int`s in UserDefaults —
/// absent means "use the default", so shipping a new default later still
/// reaches everyone who never touched Settings.
enum QuickLauncherHotKeyStore {
    private static let keyCodeKey = "quickLauncherKeyCode"
    private static let modifiersKey = "quickLauncherModifiers"

    static var keyCode: UInt32 {
        guard let stored = UserDefaults.standard.object(forKey: keyCodeKey) as? Int else {
            return QuickLauncherHotKey.defaultKeyCode
        }
        return UInt32(stored)
    }

    static var carbonModifiers: UInt32 {
        guard let stored = UserDefaults.standard.object(forKey: modifiersKey) as? Int else {
            return QuickLauncherHotKey.defaultCarbonModifiers
        }
        return UInt32(stored)
    }

    static var isDefault: Bool {
        UserDefaults.standard.object(forKey: keyCodeKey) == nil
    }

    static func set(keyCode: UInt32, carbonModifiers: UInt32) {
        UserDefaults.standard.set(Int(keyCode), forKey: keyCodeKey)
        UserDefaults.standard.set(Int(carbonModifiers), forKey: modifiersKey)
    }

    static func reset() {
        UserDefaults.standard.removeObject(forKey: keyCodeKey)
        UserDefaults.standard.removeObject(forKey: modifiersKey)
    }
}

/// Renders a Carbon (keyCode, modifiers) pair the way macOS shows shortcuts
/// (⌃⌥⇧⌘ prefix, then the key). Names come from the ANSI layout — a full
/// per-layout translation needs `UCKeyTranslate`, and every shortcut recorder
/// carries the same approximation on non-US layouts. The recorder only binds
/// keys this table can name (`hasName`).
enum HotKeyDisplay {
    static func string(keyCode: UInt32, carbonModifiers: UInt32) -> String {
        modifierSymbols(carbonModifiers) + keySymbol(keyCode)
    }

    static func modifierSymbols(_ carbonModifiers: UInt32) -> String {
        var s = ""
        if carbonModifiers & UInt32(controlKey) != 0 { s += "⌃" }
        if carbonModifiers & UInt32(optionKey) != 0 { s += "⌥" }
        if carbonModifiers & UInt32(shiftKey) != 0 { s += "⇧" }
        if carbonModifiers & UInt32(cmdKey) != 0 { s += "⌘" }
        return s
    }

    static func keySymbol(_ keyCode: UInt32) -> String {
        // The recorder refuses unmapped keys (`HotKeyCapture.verdict`), so the
        // fallback only renders for a combo stored by an older build.
        keyNames[keyCode] ?? "Key \(keyCode)"
    }

    static func hasName(_ keyCode: UInt32) -> Bool { keyNames[keyCode] != nil }

    private static let keyNames: [UInt32: String] = {
        var names: [UInt32: String] = [
            UInt32(kVK_ANSI_A): "A", UInt32(kVK_ANSI_B): "B", UInt32(kVK_ANSI_C): "C",
            UInt32(kVK_ANSI_D): "D", UInt32(kVK_ANSI_E): "E", UInt32(kVK_ANSI_F): "F",
            UInt32(kVK_ANSI_G): "G", UInt32(kVK_ANSI_H): "H", UInt32(kVK_ANSI_I): "I",
            UInt32(kVK_ANSI_J): "J", UInt32(kVK_ANSI_K): "K", UInt32(kVK_ANSI_L): "L",
            UInt32(kVK_ANSI_M): "M", UInt32(kVK_ANSI_N): "N", UInt32(kVK_ANSI_O): "O",
            UInt32(kVK_ANSI_P): "P", UInt32(kVK_ANSI_Q): "Q", UInt32(kVK_ANSI_R): "R",
            UInt32(kVK_ANSI_S): "S", UInt32(kVK_ANSI_T): "T", UInt32(kVK_ANSI_U): "U",
            UInt32(kVK_ANSI_V): "V", UInt32(kVK_ANSI_W): "W", UInt32(kVK_ANSI_X): "X",
            UInt32(kVK_ANSI_Y): "Y", UInt32(kVK_ANSI_Z): "Z",
            UInt32(kVK_ANSI_0): "0", UInt32(kVK_ANSI_1): "1", UInt32(kVK_ANSI_2): "2",
            UInt32(kVK_ANSI_3): "3", UInt32(kVK_ANSI_4): "4", UInt32(kVK_ANSI_5): "5",
            UInt32(kVK_ANSI_6): "6", UInt32(kVK_ANSI_7): "7", UInt32(kVK_ANSI_8): "8",
            UInt32(kVK_ANSI_9): "9",
            UInt32(kVK_ANSI_Minus): "-", UInt32(kVK_ANSI_Equal): "=",
            UInt32(kVK_ANSI_LeftBracket): "[", UInt32(kVK_ANSI_RightBracket): "]",
            UInt32(kVK_ANSI_Backslash): "\\", UInt32(kVK_ANSI_Semicolon): ";",
            UInt32(kVK_ANSI_Quote): "'", UInt32(kVK_ANSI_Comma): ",",
            UInt32(kVK_ANSI_Period): ".", UInt32(kVK_ANSI_Slash): "/",
            UInt32(kVK_ANSI_Grave): "`",
            UInt32(kVK_Space): "Space", UInt32(kVK_Return): "↩", UInt32(kVK_Tab): "⇥",
            UInt32(kVK_Escape): "⎋", UInt32(kVK_Delete): "⌫",
            UInt32(kVK_LeftArrow): "←", UInt32(kVK_RightArrow): "→",
            UInt32(kVK_UpArrow): "↑", UInt32(kVK_DownArrow): "↓",
        ]
        let fKeys: [UInt32] = [
            UInt32(kVK_F1), UInt32(kVK_F2), UInt32(kVK_F3), UInt32(kVK_F4), UInt32(kVK_F5),
            UInt32(kVK_F6), UInt32(kVK_F7), UInt32(kVK_F8), UInt32(kVK_F9), UInt32(kVK_F10),
            UInt32(kVK_F11), UInt32(kVK_F12),
        ]
        for (i, code) in fKeys.enumerated() { names[code] = "F\(i + 1)" }
        return names
    }()
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

    /// The engine is multi-turn: another chat's turn never blocks the
    /// launcher. Only OUR OWN in-flight answer needs superseding (stop first —
    /// the new question replaces the old one in the same conversation).
    static func submitDecision(text: String,
                               serverRunning: Bool,
                               composer: ChatTurnEngine.ComposerState) -> SubmitDecision {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return .ignore }
        guard serverRunning else {
            return .blocked("Server is not running — start it from the menu bar tray.")
        }
        switch composer {
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

    /// Launcher turns are plain chat: no tools, no MCP, no thinking, no voice
    /// styling — a quick question deserves a fast, clean answer.
    ///
    /// The app-level agent's PERSONA and sampling do apply (that's what picking
    /// one means), but its tool loop deliberately does not: this panel has no
    /// tool-call cards and no approval surface, so a loop here would run blind.
    /// "Open in chat" is one keystroke away, and that window has both.
    static func turnConfig(resolved: ResolvedAgentSettings = ResolvedAgentSettings()) -> ChatTurnEngine.TurnConfig {
        var config = ChatTurnEngine.TurnConfig.from(resolved)
        config.agentMode = false
        config.mcpMode = false
        config.tools = []
        return config
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

    /// Re-registers with whatever combo `QuickLauncherHotKeyStore` currently
    /// holds. Returns false when the system refused the combo (owned by
    /// another app) — the recorder row reverts and says so. While disabled
    /// the combo is only saved, which cannot fail: `setEnabled` picks it up
    /// the next time it's turned on.
    @discardableResult
    func updateHotKey() -> Bool {
        guard appState.quickLauncherEnabled else { return true }
        return hotKey.register(keyCode: QuickLauncherHotKey.keyCode,
                               carbonModifiers: QuickLauncherHotKey.carbonModifiers)
    }

    func toggle() {
        if panel?.isVisible == true { hide() } else { show() }
    }

    func show() {
        let panel = ensurePanel()
        // The SwiftUI content forces its scheme via `.appAppearance()`, but
        // the panel's own chrome — the NSVisualEffectView material — follows
        // the WINDOW's appearance, and forced-dark content over a
        // system-light vibrancy reads as a broken half-theme. Re-read per
        // summon: Settings can change it while the panel exists.
        panel.appearance = AppAppearanceMode.current.nsAppearance
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
    /// opened by the menu-bar label observing `pendingChatOpenTick` —
    /// SwiftUI `Window` scenes can only be opened via the `openWindow`
    /// environment, and the always-installed label is the established bridge
    /// (see the task-notification deep-link).
    func openInChat() {
        if let sessionId { appState.activeChatId = sessionId }
        // The chat window may be sitting on another pane (Models, Create…) —
        // "Open in chat" means the transcript, so switch the mode explicitly.
        // (Setting activeChatId alone does not: the sidebar deliberately has
        // no blanket onChange, because deleteSession's fallback also moves it.)
        appState.showConversation()
        appState.pendingChatOpenTick += 1
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
            stopOwnTurn()
            startTurn(text)
            return true
        case .submit:
            startTurn(text)
            return true
        }
    }

    /// Stop the LAUNCHER's own in-flight turn — never anyone else's. The
    /// engine is multi-turn now, so a chat tab's answer keeps streaming.
    func stopOwnTurn() {
        guard let sessionId else { return }
        appState.chatEngine.stop(sessionId: sessionId)
    }

    private func composerState() -> ChatTurnEngine.ComposerState {
        guard let sessionId else { return .idle }
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
        let resolved = appState.resolvedAgentSettings(agentId: appState.defaultAgentId)
        appState.chatEngine.runTurn(sessionId: sid, userText: text, images: nil, audio: nil,
                                    config: QuickLauncherLogic.turnConfig(resolved: resolved),
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
                .appAppearance()
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
