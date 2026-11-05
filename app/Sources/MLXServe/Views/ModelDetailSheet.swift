import SwiftUI

/// A model's card: its README from Hugging Face, with a button to the repo
/// page on top. Presented from every browser row (Recommended, Discover,
/// My Models, Media) via `.sheet(item:)` with a `ModelCardRequest`.
///
/// Reads no `@EnvironmentObject` on purpose — a sheet does not inherit the
/// environment it hangs on, and this one needs nothing from it. The branch
/// bar (`picksBranch`) does, so the sites that ask for it inject its objects.
struct ModelDetailSheet: View {
    let request: ModelCardRequest
    @Environment(\.dismiss) private var dismiss

    private enum Load { case loading, ready(String), failed(String) }
    @State private var load: Load = .loading
    @State private var revision: String

    init(request: ModelCardRequest) {
        self.request = request
        _revision = State(initialValue: request.revision ?? "main")
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(request.title)
                        .font(.app(.title3).weight(.semibold))
                        .lineLimit(1)
                    Text(L10n.text(request.repoId))
                        .font(.app(.caption))
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Spacer()
                if let url = ModelCard.pageURL(repoId: request.repoId, revision: revision) {
                    Button {
                        NSWorkspace.shared.open(url)
                    } label: {
                        Label("Open on Hugging Face", systemImage: "arrow.up.forward.square")
                    }
                }
                Button { dismiss() } label: { Text("Close")
                    .font(.app(.body)) }
                    .keyboardShortcut(.cancelAction)
            }
            .padding(16)
            Divider()
            if request.picksBranch {
                BranchBar(repoId: request.repoId, requested: request.revision, revision: $revision)
            }
            content
        }
        .frame(minWidth: 640, idealWidth: 760, minHeight: 420, idealHeight: 640)
        .task(id: revision) { await fetch() }
    }

    @ViewBuilder
    private var content: some View {
        switch load {
        case .loading:
            ProgressView("Loading model card…")
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .failed(let reason):
            VStack(spacing: 8) {
                Image(systemName: "doc.text.magnifyingglass")
                    .font(.app(.largeTitle))
                    .foregroundStyle(.secondary)
                Text(reason)
                    .foregroundStyle(.secondary).font(.app(.body))
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        case .ready(let markdown):
            ScrollView {
                MarkdownText(markdown)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(16)
            }
        }
    }

    private func fetch() async {
        guard let url = ModelCard.readmeURL(repoId: request.repoId, revision: revision) else {
            load = .failed("No Hugging Face page for this model.")
            return
        }
        do {
            let (data, response) = try await DownloadSession.shared.data(for: DownloadManager.hfApiRequest(url))
            guard let http = response as? HTTPURLResponse, http.statusCode == 200,
                  let text = String(data: data, encoding: .utf8) else {
                load = .failed("This model has no README on Hugging Face.")
                return
            }
            load = .ready(ModelCard.stripFrontMatter(text))
        } catch {
            load = .failed("Could not load the model card: \(error.localizedDescription)")
        }
    }
}

/// A repo whose models live on branches: each branch with the size its
/// download fetches, and that branch's download. Draws nothing for a repo
/// with only main, so such a card looks as it always did.
private struct BranchBar: View {
    let repoId: String
    let requested: String?
    @Binding var revision: String
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager
    @State private var branches: [HFBranch] = []

    private var source: HFSource { HFSource(repo: repoId, revision: revision) }
    private var key: String { source.localRepoId }

    var body: some View {
        Group {
            if HFBranch.needsPicker(branches) {
                HStack(spacing: 12) {
                    Picker("Branch", selection: $revision) {
                        ForEach(branches) { b in
                            Text(verbatim: "\(b.name) · \(MemoryInfo.format(b.bytes))").tag(b.name)
                        }
                    }
                    .fixedSize()
                    Spacer()
                    action
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                Divider()
            }
        }
        .task(id: repoId) {
            branches = await downloads.listBranches(repoId: repoId)
            if let pick = HFBranch.preselect(branches, requested: requested) { revision = pick }
        }
    }

    @ViewBuilder
    private var action: some View {
        if let state = downloads.downloads[key], state.status == .downloading {
            HStack(spacing: 6) {
                ProgressView(value: state.progress).frame(width: 80)
                Text(verbatim: state.percentFormatted)
                    .font(.app(.caption).monospacedDigit())
                    .foregroundStyle(.secondary)
                Button {
                    downloads.cancel(key)
                    appState.refreshModels()
                } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Cancel download")
            }
        } else if downloads.isReady(key) {
            Text("✓ On disk")
                .font(.app(.caption).weight(.medium))
                .foregroundStyle(.green)
        } else {
            Button {
                let commit = branches.first { $0.name == revision }?.commit
                downloads.start(repoId: repoId, revision: revision, commit: commit) {
                    appState.refreshModels()
                    server.rescanModels()
                }
            } label: {
                Text(L10n.text(downloads.hasPartialDownload(key) ? "Resume" : "Download")).font(.app(.body))
            }
        }
    }
}
