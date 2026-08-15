import SwiftUI

/// The chat window's "you need a model first" gate.
///
/// Opening Chat with nothing downloaded used to give an empty transcript, a
/// "Select a model" pill and a disabled Send — a dead end with nothing on
/// screen saying why. This blocks instead, and offers the SAME starter
/// recommendation the welcome window does (`RecommendedStarterCard`, one view,
/// one `starterPick`), so the two can't drift.
///
/// It fires on chat-CAPABLE models, not on "any model": someone whose only
/// download is an image backend has a full models folder and still can't send
/// a message. And it clears itself — `AppState.localModels` is `@Published`
/// and the card refreshes it when the transfer lands.
///
/// Cancel CLOSES the window. Dismissing to the dead composer underneath is the
/// state this sheet exists to replace.
struct ChatModelGateSheet: View {
    let pick: RecommendedModelPick
    let onCancel: () -> Void

    @EnvironmentObject var appState: AppState
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var server: ServerManager

    /// Recomputed from the observed download manager, so the copy follows the
    /// transfer without this view owning any state of its own.
    private var state: ChatGateState {
        let active = downloads.downloads[pick.repoId]
        return ChatGateState.resolve(
            localModels: appState.localModels,
            activeDownload: active?.status == .downloading ? active?.progress : nil,
            lanChatModelCount: server.lanModels(capability: "chat").count
        )
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                Text(headline)
                    .font(.title3.weight(.semibold))
                Text(subhead)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            RecommendedStarterCard(pick: pick)

            HStack {
                Spacer()
                Button("Cancel", action: onCancel)
                    .keyboardShortcut(.cancelAction)
            }
        }
        .padding(20)
        .frame(width: 380)
    }

    private var headline: String {
        switch state {
        case .downloading: return "Getting your model ready"
        default:           return "You need a model to chat"
        }
    }

    private var subhead: String {
        switch state {
        case .downloading:
            return "Chat opens as soon as this finishes — you can leave it running."
        default:
            return "MLX Core runs models on your own Mac, so there's a one-time download first."
        }
    }
}
