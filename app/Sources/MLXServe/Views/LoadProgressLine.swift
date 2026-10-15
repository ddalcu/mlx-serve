import SwiftUI

/// A model load's bar and line (`LoadProgress`): determinate while the server counts the phase,
/// indeterminate before it reports one and while warming up.
struct LoadProgressLine: View {
    let name: String
    let progress: LoadProgress?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            LoadProgressBar(progress: progress)
            Text(verbatim: L10n.format("Loading %@ · %@", name, progress?.label ?? L10n.text("Starting")))
                .font(.app(.caption))
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
    }
}

struct LoadProgressBar: View {
    let progress: LoadProgress?

    var body: some View {
        if let f = progress?.fraction {
            ProgressView(value: f).progressViewStyle(.linear)
        } else {
            ProgressView().progressViewStyle(.linear)
        }
    }
}

/// The chat model loading (above the composer, in the tray). Its own observing view, so a poll
/// redraws this row and not the view around it.
struct ModelLoadProgressRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var server: ServerManager

    var body: some View {
        if let path = appState.loadingModelPath {
            LoadProgressLine(name: ModelDisplayName.pretty(URL(fileURLWithPath: path).lastPathComponent),
                             progress: LoadProgress.lookup(server.loadProgress, path: path))
                .padding(.horizontal, 6)
        }
    }
}

/// Every model this app is loading (`ServerManager.loadProgress`), for the Create panes; nothing
/// when none is.
struct LoadingModelsRow: View {
    @EnvironmentObject var server: ServerManager

    var body: some View {
        if !server.loadProgress.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                ForEach(server.loadProgress.keys.sorted(), id: \.self) { path in
                    LoadProgressLine(name: ModelDisplayName.pretty(URL(fileURLWithPath: path).lastPathComponent),
                                     progress: server.loadProgress[path])
                }
            }
            .frame(maxWidth: 360)
        }
    }
}
