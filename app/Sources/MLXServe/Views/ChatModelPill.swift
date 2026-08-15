import SwiftUI

/// The chat window's model picker: which model this conversation is talking to,
/// with a status dot, sitting at the LEADING edge of the toolbar.
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
    static func headerName(_ full: String) -> String {
        guard let slash = full.lastIndex(of: "/") else { return full }
        let tail = full[full.index(after: slash)...]
        return tail.isEmpty ? full : String(tail)
    }

    private var lanChatModels: [ModelInfo] { server.lanModels(capability: "chat") }
    private var pickableModels: [LocalModel] { appState.localModels.filter(\.isChatPickable) }

    /// The name and the in-flight state, decided in ONE pure place shared with
    /// the tray's rules (`ChatModelSelection.pillState`) — a per-surface copy is
    /// how one picker starts naming a model the other doesn't.
    private var pill: ChatModelPillState {
        ChatModelSelection.pillState(lanChatModelId: server.lanChatModelId,
                                     residentName: server.chatModelInfo?.name,
                                     loadingPath: appState.loadingModelPath,
                                     selectedPath: appState.selectedModelPath,
                                     models: pickableModels)
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

    /// A live transfer THIS CHAT is waiting on, if any. The pill is where the
    /// user is already looking when they wonder why nothing answers, so a
    /// download in flight belongs here rather than only in the browser two
    /// clicks away — but it has to be the download that explains the silence.
    /// Excluding media bundles was half of that (a 30 GB video pack grew a
    /// hairline under the chat model's name); the other half is that a CHAT
    /// model fetched in the background drew a bar under a different model that
    /// was already answering perfectly well. It is keyed to the pill's own
    /// model now, the way every other download surface keys its row.
    ///
    /// `hasChatModelOnDisk` is the exception, and the reason the hairline
    /// exists: with nothing pickable on this Mac the composer cannot answer at
    /// all, so whatever chat model is arriving IS what it is waiting for.
    static func chatDownload(in downloads: [String: DownloadManager.DownloadState],
                             mediaRepos: Set<String>,
                             selectedModelPath: String,
                             hasChatModelOnDisk: Bool) -> DownloadManager.DownloadState? {
        downloads
            .filter { repoId, state in
                guard state.status == .downloading, !mediaRepos.contains(repoId) else { return false }
                return hasChatModelOnDisk ? isTransfer(of: repoId, for: selectedModelPath) : true
            }
            .min { $0.key < $1.key }?  // stable pick when two are running
            .value
    }

    /// Whether `repoId` is the model at `path`. A download has no `LocalModel`
    /// until its files land, so the join is the layout the downloader writes:
    /// `<root>/<org>/<name>`, i.e. the path's last two components ARE the repo
    /// id. A GGUF selection points at the file inside that folder, so its repo
    /// is one level further up.
    static func isTransfer(of repoId: String, for path: String) -> Bool {
        guard !repoId.isEmpty, !path.isEmpty else { return false }
        var dir = path as NSString
        while dir.hasSuffix("/") { dir = dir.deletingLastPathComponent as NSString }
        if dir.pathExtension.lowercased() == "gguf" { dir = dir.deletingLastPathComponent as NSString }
        let org = (dir.deletingLastPathComponent as NSString).lastPathComponent
        let name = dir.lastPathComponent
        return repoId == "\(org)/\(name)" || repoId == name
    }

    private var activeDownload: DownloadManager.DownloadState? {
        guard server.lanChatModelId == nil else { return nil }
        return Self.chatDownload(in: downloads.downloads,
                                 mediaRepos: downloads.mediaBundleRepos,
                                 selectedModelPath: appState.selectedModelPath,
                                 hasChatModelOnDisk: !pickableModels.isEmpty)
    }

    /// True when this Mac has nothing chat-pickable on disk — the state where
    /// the pill's job is to offer the download, not a list to choose from.
    private var needsDownload: Bool {
        server.lanChatModelId == nil && pickableModels.isEmpty && activeDownload == nil
    }

    var body: some View {
        let pill = self.pill
        return Menu {
            menuContent
        } label: {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 5) {
                    Image(systemName: "cpu")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(.secondary)
                    // The READABLE name (`ModelDisplayName`). The repo id it is
                    // built from stays the identity and is still what the menu
                    // rows carry underneath — this is the label, not a rename.
                    Text(ModelDisplayName.pretty(pill.name))
                        .font(compact ? .callout.weight(.semibold) : .callout.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: compact ? Self.compactNameWidth : Self.maxNameWidth,
                               alignment: .leading)
                        .fixedSize(horizontal: false, vertical: true)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.secondary)
                    if pill.isLoading {
                        // A switch in flight. The dot would say GREEN here —
                        // the server is running and a model is resident, just
                        // not this one — so the spinner replaces it rather than
                        // sitting beside it.
                        ProgressView()
                            .controlSize(.small)
                            .scaleEffect(0.6)
                            .frame(width: 8, height: 8)
                    } else if needsDownload {
                        // Nothing on disk: the one thing to do is get one, so
                        // say it with the symbol rather than a dot that only
                        // reports.
                        Image(systemName: "arrow.down.circle.fill")
                            .font(.system(size: 11))
                            .foregroundStyle(Color.accentColor)
                    } else {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 8, height: 8)
                    }
                }
                if let active = activeDownload {
                    // Deliberately wordless: a hairline that fills. The figures
                    // live in the browser; here it only has to say "something is
                    // arriving, that's why it can't answer yet".
                    ProgressView(value: max(0, min(1, active.progress)))
                        .progressViewStyle(.linear)
                        .tint(.green)
                        .frame(height: 3)
                        .frame(maxWidth: compact ? Self.compactNameWidth : Self.maxNameWidth)
                }
            }
            .padding(.horizontal, showsBackground ? 12 : 4)
            .padding(.vertical, 4)
            // The composer row's own control height, so the pill lines up with
            // the discs and the send button rather than sitting a few points
            // shy of them. A FLOOR, not a fixed height: a download adds the
            // progress hairline under the name and the capsule grows to hold
            // it, which is the shape in the mockup.
            .frame(minHeight: compact ? ChatMetrics.composerIconSize : 0)
            .background(showsBackground ? Color.secondary.opacity(0.12) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: compact ? 10 : 999, style: .continuous))
            .contentShape(Rectangle())
        }
        .menuStyle(.button)
        .buttonStyle(.plain)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(pill.isLoading
              ? "Loading \(ModelDisplayName.pretty(pill.name))…"
              : "Chat model. Click to switch — models on this Mac and any shared by other Macs on your network.")
    }

    @ViewBuilder
    private var menuContent: some View {
        let pickable = pickableModels
        if pickable.isEmpty && lanChatModels.isEmpty {
            Text("No chat models downloaded")
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
