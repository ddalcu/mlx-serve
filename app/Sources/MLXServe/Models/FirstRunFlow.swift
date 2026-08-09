import Foundation

/// The two first-run decisions, as pure data.
///
/// Both surfaces they drive are awkward to test directly — the welcome window
/// is a bare `NSHostingView` outside the SwiftUI Scene graph, and the chat gate
/// is a sheet over a window — so the DECISION lives here and each view stays a
/// thin renderer of it. Same shape as `ChatModeToggles`/`ModelUseState`.

// MARK: - What a launch opens

/// What the app puts on screen at launch, given whether the user ticked
/// "Don't show again" on the welcome window.
enum LaunchDecision: Equatable {
    /// Show the welcome window (which opens Chat when dismissed).
    case showWelcome
    /// Skip straight to the chat window.
    case openChat

    /// `hasChatModels` is taken and deliberately does NOT change the answer: a
    /// user who suppressed the welcome gets Chat whether or not anything is
    /// downloaded, because the CHAT GATE offers the same starter card there.
    /// Re-showing a window they turned off — on the one launch where they have
    /// nothing, i.e. exactly when they'd retick it — is how a "don't show
    /// again" box loses the user's trust. It's a parameter so that stays a
    /// pinned decision rather than an omission.
    static func resolve(welcomeSuppressed: Bool, hasChatModels: Bool) -> LaunchDecision {
        welcomeSuppressed ? .openChat : .showWelcome
    }

    /// Constant, and that IS the point: the welcome is a SHEET on the chat
    /// window now, and a sheet with no host window is a screen nobody can see.
    /// `.showWelcome` used to skip opening Chat — the welcome floated over an
    /// empty desktop and every exit had to remember to open one.
    var opensChatWindow: Bool { true }

    var presentsWelcome: Bool { self == .showWelcome }

    /// UserDefaults key behind `welcomeSuppressed`. Absent ⇒ false ⇒ the
    /// welcome shows, which is the pre-existing behaviour on every launch.
    static let suppressDefaultsKey = "suppressWelcomeWindow"
}

// MARK: - Leaving the welcome window

/// Every way out of the welcome window, and what each one leaves on screen.
///
/// Live dead end: "Browse all models" dismissed the window and opened ONLY the
/// Model Browser. Close that browser — or download a model without spotting its
/// "Use" button — and you are looking at an empty desktop with the app in the
/// menu bar, unable to get back to the screen you started from: the welcome
/// window is `.floating`, `_welcomeWindow` is already closed, and nothing
/// re-opens it until the next launch.
///
/// So the rule is a TYPE, not a habit: an exit always opens Chat and always
/// closes the window. Browse opens the browser too, on top of the chat window
/// it just queued, which is why closing the browser now lands on a composer.
enum WelcomeExit: CaseIterable, Equatable {
    /// The footer's primary button.
    case startChatting
    /// A model row's Get/Use control, once the model is loaded.
    case useModel
    /// "Browse all models" in the Run-models panel.
    case browseModels

    /// Deliberately constant: it is the invariant, and a test asserts it over
    /// `allCases` so a new exit can't opt out by forgetting.
    var opensChat: Bool { true }

    var opensModelBrowser: Bool { self == .browseModels }

    /// Also constant. It used to be load-bearing against the dead end above:
    /// the welcome floated over everything, so anything it opened was
    /// invisible until it closed. As a SHEET on the chat window the dead end
    /// is unbuildable — whatever dismisses it, a composer is what's behind it —
    /// so this is now simply what "leaving" means.
    var closesWelcome: Bool { true }
}

// MARK: - The chat gate

/// Whether the chat window must block on "you need a model first", and what
/// that block says.
///
/// The condition is a CHAT-CAPABLE model, not "any model": someone whose only
/// download is an image backend has a full `~/.mlx-serve/models` and still
/// cannot send a message. `localModels` already covers LM Studio's folder and
/// the Hugging Face hub cache (`DownloadManager.discoverLocalModels`), so
/// anyone who arrived with models never sees the sheet — and because it's
/// `@Published`, the sheet clears itself the moment a download lands.
///
/// LAN-discovered chat models count as usable, same as `trayHasNoUsableModels`:
/// a Mac with nothing downloaded can still chat on a peer's model, and a gate
/// that blocked it would lock the user out of a conversation they can already
/// have.
enum ChatGateState: Equatable {
    /// A chat model is available — no sheet.
    case hidden
    /// Nothing to chat with, and nothing on the way.
    case needsModel
    /// Nothing to chat with yet, but the starter model is transferring.
    case downloading(progress: Double)

    /// - Parameters:
    ///   - activeDownload: fractional progress (0…1) of the starter model's
    ///     transfer, or nil when nothing is in flight. A FAILED transfer passes
    ///     nil and lands on `.needsModel`, where the card's own control offers
    ///     Resume/Retry.
    ///   - lanChatModelCount: chat models this Mac can reach on a peer.
    static func resolve(localModels: [LocalModel],
                        activeDownload: Double?,
                        lanChatModelCount: Int = 0) -> ChatGateState {
        // A usable model wins even mid-transfer: downloading a SECOND model
        // must never block a chat the user can already have.
        if lanChatModelCount > 0 { return .hidden }
        if localModels.contains(where: \.isChatPickable) { return .hidden }
        if let progress = activeDownload { return .downloading(progress: min(max(progress, 0), 1)) }
        return .needsModel
    }

    var isBlocking: Bool { self != .hidden }
}
