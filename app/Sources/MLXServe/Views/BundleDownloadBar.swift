import SwiftUI

/// Inline "this model needs downloading" gate for the media-gen panes. Shows a
/// Download button (with the bundle's total size) when the model isn't on disk,
/// live progress while downloading (which component / file), a Retry on
/// failure, or nothing when the bundle is fully present. Bundles pull only the
/// files the engine reads, and LTX's bundle also pulls its Gemma-3 text
/// encoder — all surfaced here as one action.
struct BundleDownloadBar: View {
    let bundle: MediaBundle
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    var body: some View {
        if downloads.bundleReady(bundle) {
            EmptyView()
        } else if let active = downloads.activeBundleComponent(bundle) {
            if active.state.status == .failed {
                failedRow(active.state)
            } else {
                downloadingRow(active)
            }
        } else {
            notStartedRow
        }
    }

    private var notStartedRow: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("This model isn't downloaded yet.")
                .font(.caption).foregroundStyle(.secondary)
            Button {
                downloads.startBundle(bundle) { appState.refreshModels() }
            } label: {
                Label("Download (\(bundle.approxSizeLabel))", systemImage: "arrow.down.circle")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            if bundle.components.count > 1 {
                Text("Includes \(bundle.components.count) models (e.g. the text encoder).")
                    .font(.caption2).foregroundStyle(.tertiary)
            }
        }
    }

    private func downloadingRow(_ a: (repo: String, index: Int, count: Int, state: DownloadManager.DownloadState)) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                ProgressView(value: a.state.fileProgress).frame(maxWidth: .infinity)
                Text("\(a.state.percentFormatted) \(a.state.speedFormatted)")
                    .font(.system(size: 9).monospacedDigit()).foregroundStyle(.secondary)
                Button { downloads.cancelBundle(bundle) } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Cancel download")
            }
            let label = a.count > 1 ? "Downloading model \(a.index)/\(a.count): " : "Downloading: "
            Text(label + (a.state.currentFile.isEmpty ? a.state.statusText : a.state.currentFile))
                .font(.system(size: 9)).foregroundStyle(.secondary)
                .lineLimit(1).truncationMode(.middle)
        }
    }

    private func failedRow(_ state: DownloadManager.DownloadState) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(state.error ?? "Download failed")
                .font(.caption2).foregroundStyle(.red).lineLimit(2)
            Button("Retry") {
                downloads.startBundle(bundle) { appState.refreshModels() }
            }
            .buttonStyle(.bordered).controlSize(.small)
        }
    }
}
