import SwiftUI

struct ModelDownloadView: View {
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(gemmaModelOptionsTrayMenu) { option in
                ModelDownloadRow(option: option)
            }
        }
        .padding(.top, 4)
    }
}

struct ModelDownloadRow: View {
    let option: GemmaModelOption
    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    private var state: DownloadManager.DownloadState? {
        downloads.downloads[option.repoId]
    }
    private var isReady: Bool {
        downloads.isReady(option.repoId)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 1) {
                    Text(L10n.text(option.displayName))
                        .font(.app(.caption).weight(.medium))
                    Text(L10n.text(option.sizeEstimate))
                        .font(.app(.caption2))
                        .foregroundStyle(.tertiary)
                }
                Spacer()

                if isReady {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.app(.caption))
                } else if let state, state.status == .downloading {
                    HStack(spacing: 6) {
                        VStack(alignment: .trailing, spacing: 1) {
                            ProgressView(value: state.progress)
                                .frame(width: 60)
                            Text("\(state.percentFormatted) \(state.speedFormatted)")
                                .font(.app(.caption2).monospacedDigit())
                                .foregroundStyle(.secondary)
                        }
                        Button {
                            downloads.cancel(option.repoId)
                        } label: {
                            Image(systemName: "xmark.circle.fill")
                                .foregroundStyle(.secondary)
                                .font(.app(.caption))
                        }
                        .buttonStyle(.plain)
                        .help("Cancel download")
                    }
                } else if let state, state.status == .completed {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                        .font(.app(.caption))
                } else if let state, state.status == .failed {
                    Button(L10n.text(downloads.hasPartialDownload(option.repoId) ? "Resume" : "Retry")) {
                        startDownload()
                    }
                    .font(.app(.caption))
                    .controlSize(.mini)
                } else {
                    Button(L10n.text(downloads.hasPartialDownload(option.repoId) ? "Resume" : "Download")) {
                        startDownload()
                    }
                    .font(.app(.caption))
                    .controlSize(.mini)
                }
            }

            // Status text for active downloads
            if let state, state.status == .downloading, !state.statusText.isEmpty {
                Text("[\(state.fileIndex)/\(state.fileCount)] \(state.statusText)")
                    .font(.app(.caption2))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }
            if let state, state.status == .failed, let error = state.error {
                Text(error)
                    .font(.app(.caption2))
                    .foregroundStyle(.red)
                    .lineLimit(2)
            }
        }
        .padding(.vertical, 3)
    }

    /// Kick off a download, picking the GGUF single-file path for ds4-backed
    /// entries and the standard safetensors-tree path for everything else.
    /// The tracked-task wrappers handle cancellation; `refreshModels` runs
    /// after completion, failure, or cancel so the picker stays in sync.
    private func startDownload() {
        let refresh: @MainActor () -> Void = { appState.refreshModels() }
        if let gguf = option.ggufFilename {
            downloads.startGguf(repoId: option.repoId, ggufFilename: gguf, onFinish: refresh)
        } else {
            downloads.start(repoId: option.repoId, onFinish: refresh)
        }
    }
}
