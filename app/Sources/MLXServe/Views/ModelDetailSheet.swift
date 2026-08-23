import SwiftUI

/// A model's card: its README from Hugging Face, with a button to the repo
/// page on top. Presented from every browser row (Recommended, Discover,
/// My Models, Media) via `.sheet(item:)` with a `ModelCardRequest`.
///
/// Reads no `@EnvironmentObject` on purpose — a sheet does not inherit the
/// environment it hangs on, and this one needs nothing from it.
struct ModelDetailSheet: View {
    let request: ModelCardRequest
    @Environment(\.dismiss) private var dismiss

    private enum Load { case loading, ready(String), failed(String) }
    @State private var load: Load = .loading

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(request.title)
                        .font(.title3.weight(.semibold))
                        .lineLimit(1)
                    Text(request.repoId)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Spacer()
                if let url = ModelCard.pageURL(repoId: request.repoId) {
                    Button {
                        NSWorkspace.shared.open(url)
                    } label: {
                        Label("Open on Hugging Face", systemImage: "arrow.up.forward.square")
                    }
                }
                Button("Close") { dismiss() }
                    .keyboardShortcut(.cancelAction)
            }
            .padding(16)
            Divider()
            content
        }
        .frame(minWidth: 640, idealWidth: 760, minHeight: 420, idealHeight: 640)
        .task(id: request.repoId) { await fetch() }
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
                    .font(.largeTitle)
                    .foregroundStyle(.secondary)
                Text(reason)
                    .foregroundStyle(.secondary)
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
        guard let url = ModelCard.readmeURL(repoId: request.repoId) else {
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
