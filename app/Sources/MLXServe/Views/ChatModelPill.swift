import SwiftUI

/// The chat window's model picker: which model this conversation is talking to,
/// with a status dot, sitting at the LEADING edge of the toolbar.
///
/// Leading placement is deliberate. The trailing cluster (voice, settings, the
/// three mode pills) is at its width budget, and anything runtime-variable added
/// there re-triggers the » overflow-eviction class — which is also why this
/// pill's label is width-CAPPED and truncated rather than sized to the model
/// name. A 60-character Hugging Face repo id must not be able to push the
/// toolbar around.
///
/// Selection goes through `ChatModelSelection`, the same definition the menu-bar
/// tray uses, so the two pickers cannot disagree about what is loaded.
struct ChatModelPill: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @Environment(\.openWindow) private var openWindow

    /// False when embedded in the toolbar's floating cluster, which draws ONE
    /// capsule around the model picker, voice and settings together — a second
    /// capsule inside it reads as a button inside a button.
    var showsBackground: Bool = true
    /// Composer placement: sized and weighted like the rest of that row (which
    /// is where the picker lives now — it configures the message you are about
    /// to send, not the window), and it carries the download affordances a
    /// toolbar pill had no room for.
    var compact: Bool = false

    @EnvironmentObject private var downloads: DownloadManager

    /// Hard cap on the name's width. The pill's size is what keeps it safe in
    /// the toolbar, so this is a contract, not a nicety.
    private static let maxNameWidth: CGFloat = 210
    /// Tighter cap in the composer row, which has its own budget — see the
    /// toolbar-eviction note above; the same discipline applies here.
    private static let compactNameWidth: CGFloat = 150

    /// What the PILL shows: the model name without its org.
    ///
    /// The org is the half of a Hugging Face id that's identical across most of
    /// your models, and it was eating the width budget from the left while the
    /// middle-truncation ate the part that identifies the model
    /// ("mlx-commun…B-it-qat-4bit"). The MENU keeps full ids — that's where
    /// you're choosing between them, and two orgs can ship the same name.
    ///
    /// A LAN id is `org/model@peer`, so taking the last path component keeps
    /// the peer; anything without a slash, or ending in one, is left alone
    /// rather than rendered as an empty pill.
    static func headerName(_ full: String) -> String {
        guard let slash = full.lastIndex(of: "/") else { return full }
        let tail = full[full.index(after: slash)...]
        return tail.isEmpty ? full : String(tail)
    }

    private var lanChatModels: [ModelInfo] { server.lanModels(capability: "chat") }
    private var pickableModels: [LocalModel] { appState.localModels.filter(\.isChatPickable) }

    /// What to show on the pill. The LAN id wins for the same reason it wins in
    /// the selection tag: while chatting over the network, the local model is
    /// not the one answering.
    private var displayName: String {
        if let lanId = server.lanChatModelId { return lanId }
        if let loaded = server.chatModelInfo?.name, !loaded.isEmpty { return loaded }
        if let picked = pickableModels.first(where: { $0.path == appState.selectedModelPath }) {
            return picked.displayLabel
        }
        return "Select a model"
    }

    /// Green once the server is up AND a chat model is actually resident —
    /// "running with nothing loaded" is not ready to answer, and a green dot
    /// there is a lie the first message exposes.
    private var statusColor: Color {
        guard server.status == .running else { return .secondary.opacity(0.5) }
        if server.lanChatModelId != nil { return .green }
        return server.chatModelInfo == nil ? .orange : .green
    }

    private var selection: Binding<String> {
        Binding(
            get: { ChatModelSelection.tag(localPath: appState.selectedModelPath,
                                          lanChatModelId: server.lanChatModelId) },
            set: { picked in
                switch ChatModelSelection.action(for: picked) {
                case .selectLan(let id):
                    appState.selectLanModel(id)
                case .selectLocal(let path):
                    server.lanChatModelId = nil
                    appState.selectedModelPath = path
                }
            }
        )
    }

    /// A live transfer for the model this chat is pointed at, if any. The pill
    /// is where the user is already looking when they wonder why nothing
    /// answers, so a download in flight belongs here rather than only in the
    /// browser two clicks away.
    private var activeDownload: DownloadManager.DownloadState? {
        guard server.lanChatModelId == nil else { return nil }
        return downloads.downloads.values.first { $0.status == .downloading }
    }

    /// True when this Mac has nothing chat-pickable on disk — the state where
    /// the pill's job is to offer the download, not a list to choose from.
    private var needsDownload: Bool {
        server.lanChatModelId == nil && pickableModels.isEmpty && activeDownload == nil
    }

    var body: some View {
        Menu {
            menuContent
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 5) {
                    Image(systemName: "cpu")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                    Text(Self.headerName(displayName))
                        .font(compact ? .caption.weight(.medium) : .callout.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: compact ? Self.compactNameWidth : Self.maxNameWidth,
                               alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 8, weight: .semibold))
                        .foregroundStyle(.secondary)
                    if needsDownload {
                        // Nothing on disk: the one thing to do is get one, so
                        // say it with the symbol rather than a dot that only
                        // reports.
                        Image(systemName: "arrow.down.circle.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(Color.accentColor)
                    } else {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 6, height: 6)
                    }
                }
                if let active = activeDownload {
                    // Deliberately wordless: a hairline that fills. The figures
                    // live in the browser; here it only has to say "something is
                    // arriving, that's why it can't answer yet".
                    ProgressView(value: max(0, min(1, active.fileProgress)))
                        .progressViewStyle(.linear)
                        .tint(.green)
                        .frame(height: 2)
                        .frame(maxWidth: compact ? Self.compactNameWidth : Self.maxNameWidth)
                }
            }
            .padding(.horizontal, showsBackground ? 10 : 4)
            .padding(.vertical, 4)
            .background(showsBackground ? Color.secondary.opacity(0.12) : Color.clear)
            .clipShape(Capsule())
            .contentShape(Capsule())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help("Chat model. Click to switch — models on this Mac and any shared by other Macs on your network.")
    }

    @ViewBuilder
    private var menuContent: some View {
        let pickable = pickableModels
        if pickable.isEmpty && lanChatModels.isEmpty {
            Text("No chat models downloaded")
            Divider()
        } else {
            // Same duplicate-name suffixing as the tray: a menu keys its
            // checkmark by row TITLE, so two same-named rows both tick.
            let dupNames = LocalModel.duplicateNames(in: pickable)
            ForEach(LocalModelSource.allCases, id: \.self) { source in
                let models = pickable.filter { $0.source == source }
                if !models.isEmpty {
                    Section(source.sectionTitle) {
                        ForEach(models) { model in
                            row(title: rowLabel(model, dupNames: dupNames),
                                tag: model.path)
                        }
                    }
                }
            }
            if !lanChatModels.isEmpty {
                Section("On Your Network") {
                    ForEach(lanChatModels, id: \.name) { m in
                        row(title: m.lanDisplayName, tag: "lan:" + m.name)
                    }
                }
            }
            Divider()
        }
        Divider()
        Button("Manage Models…") {
            // A MODE of this window now, not a window of its own — so the
            // picker's "manage" route lands beside the picker rather than on
            // top of it. AppState.showModels is the one way in, and it is the
            // LAST row: everything above it is a model you can pick right now.
            appState.showModels()
        }
    }

    private func row(title: String, tag: String) -> some View {
        Button {
            selection.wrappedValue = tag
        } label: {
            if selection.wrappedValue == tag {
                Label(title, systemImage: "checkmark")
            } else {
                Text(title)
            }
        }
    }

    /// `displayLabel`, not `name`: a GGUF repo ships several quants and each is
    /// its own row, so the row has to say which quant it loads.
    private func rowLabel(_ model: LocalModel, dupNames: Set<String>) -> String {
        let label = model.displayLabel
        return dupNames.contains(label) ? "\(label) · \(model.engine.shortLabel)" : label
    }
}
