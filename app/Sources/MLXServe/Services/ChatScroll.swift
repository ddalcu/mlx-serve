import CoreGraphics
import Combine
import SwiftUI

/// Who is moving the transcript right now.
///
/// The distinction is the whole game: "the content got taller" and "the user
/// scrolled" both change the same numbers, and only one of them means stop
/// following. The old code tried to tell them apart with an app-global
/// `NSEvent` scroll-wheel monitor, which saw scrolls in every other window too
/// and could not see a scroller-thumb drag, a keyboard scroll, or a window
/// resize at all.
enum ChatScrollDriver: Equatable {
    case idle
    /// The user's gesture — including the momentum still carrying from it.
    case user
    /// A scroll we asked for.
    case us

    init(_ phase: ScrollPhase) {
        switch phase {
        case .tracking, .interacting, .decelerating: self = .user
        case .animating: self = .us
        case .idle: self = .idle
        @unknown default: self = .idle
        }
    }
}

enum ChatScrollEvent: Equatable {
    /// The transcript appeared, or switched to a different conversation.
    case transcriptShown
    case userSentMessage
    case jumpTapped
    case driverChanged(ChatScrollDriver)
    /// Negative when the scroll view is rubber-banding past the end.
    case geometryChanged(distanceFromBottom: CGFloat)
}

enum ChatScrollAction: Equatable {
    case none
    case toBottom(animated: Bool)
}

/// Decides whether the transcript follows the newest line, and when to scroll.
///
/// Pure, so the rules are pinned by `ChatScrollTests` instead of by trying a
/// build and scrolling around. `ChatScrollModel` owns one of these for the view.
struct ChatScrollState: Equatable {

    /// Land within this of the end and following re-engages — about two lines,
    /// so finishing a fling or dragging the scroller down does it reliably.
    static let bottomTolerance: CGFloat = 32

    /// While following, any visible gap below the last line gets snapped shut.
    /// Sub-point so layout noise doesn't churn, small enough that a clipped
    /// line never survives a frame.
    ///
    /// This is a safety net, not the mechanism: the view's bottom size-change
    /// anchor normally keeps the end glued with no scroll at all, so on the
    /// streaming path this rule fires for the cases the anchor can't see — a row
    /// re-measuring after its image or syntax highlighting lands, a restore, a
    /// window resize.
    static let correctionSlack: CGFloat = 0.5

    private(set) var isPinnedToBottom = true
    private(set) var driver: ChatScrollDriver = .idle

    mutating func handle(_ event: ChatScrollEvent) -> ChatScrollAction {
        switch event {
        case .transcriptShown:
            isPinnedToBottom = true
            return .toBottom(animated: false)

        case .userSentMessage, .jumpTapped:
            isPinnedToBottom = true
            return .toBottom(animated: true)

        case .driverChanged(let driver):
            self.driver = driver
            return .none

        case .geometryChanged(let distance):
            // Being at the end re-engages no matter who put us there; leaving it
            // only disengages when the user did the leaving. Everything else —
            // a taller message, a card appearing, a window resize — is content
            // moving under a reader who has not asked for anything.
            if distance <= Self.bottomTolerance {
                isPinnedToBottom = true
            } else if driver == .user {
                isPinnedToBottom = false
            }

            // A scroll of ours is mid-flight: correcting would cancel it.
            guard isPinnedToBottom, driver == .idle,
                  distance > Self.correctionSlack else { return .none }
            return .toBottom(animated: false)
        }
    }

    /// How far the end of the content sits below the bottom of the viewport.
    ///
    /// Split out from `ScrollGeometry` so the arithmetic is testable and so the
    /// bottom inset can't be forgotten: a scroll-edge/safe-area inset is part of
    /// the scrollable range, and leaving it out reports a residual distance that
    /// never reaches zero — which would make the correction rule fire forever.
    static func distanceFromBottom(contentHeight: CGFloat,
                                   offsetY: CGFloat,
                                   containerHeight: CGFloat,
                                   bottomInset: CGFloat) -> CGFloat {
        (contentHeight + bottomInset) - (offsetY + containerHeight)
    }

    static func distanceFromBottom(_ geometry: ScrollGeometry) -> CGFloat {
        distanceFromBottom(contentHeight: geometry.contentSize.height,
                           offsetY: geometry.contentOffset.y,
                           containerHeight: geometry.containerSize.height,
                           bottomInset: geometry.contentInsets.bottom)
    }
}

/// View-side holder for `ChatScrollState`.
///
/// A class, not `@State`, because scroll geometry arrives on every frame of
/// every gesture and writing that into view state would re-evaluate the whole
/// chat body ~60 times a second. Only `isPinnedToBottom` is published — it is
/// the one thing the view draws from (the bottom size-change anchor and the
/// jump-to-latest button), and it changes a handful of times per conversation.
@MainActor
final class ChatScrollModel: ObservableObject {
    @Published private(set) var isPinnedToBottom = true
    private var state = ChatScrollState()

    func apply(_ event: ChatScrollEvent) -> ChatScrollAction {
        let action = state.handle(event)
        if isPinnedToBottom != state.isPinnedToBottom {
            isPinnedToBottom = state.isPinnedToBottom
        }
        return action
    }
}
