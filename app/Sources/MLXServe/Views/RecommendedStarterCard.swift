import SwiftUI

/// The one-model recommendation, as a card: what it is, how big it is, its
/// name, and a Download button that reports progress in place.
///
/// ONE view, used by both first-run surfaces — the welcome window and the chat
/// gate. Two copies is how the two recommendations drift apart, and they are
/// shown minutes (sometimes seconds) apart to the same person. The pick itself
/// comes from `RecommendedModelPick.starterPick(physicalMemoryBytes:)`, which
/// the Model Browser's "Best for your Mac" card also reads.
///
/// When the transfer finishes the card doesn't just mark itself done: it
/// selects the model and starts the server (`useModelAndAwaitReady`), because
/// the point of the whole flow is to end in a working chat rather than in a
/// folder of weights the user still has to go and switch on.
struct RecommendedStarterCard: View {
    let pick: RecommendedModelPick
    /// Called once the model is downloaded, selected and the server is up (or
    /// gave up waiting). The welcome window uses it to move the user into Chat.
    var onReady: (() -> Void)? = nil

    @EnvironmentObject var downloads: DownloadManager
    @EnvironmentObject var appState: AppState

    private var state: DownloadManager.DownloadState? { downloads.downloads[pick.repoId] }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "sparkles")
                    .font(.system(size: 16))
                    .foregroundColor(.accentColor)
                    .frame(width: 24, alignment: .center)
                VStack(alignment: .leading, spacing: 2) {
                    Text(Self.lead(for: pick))
                        .font(.subheadline.weight(.semibold))
                        .fixedSize(horizontal: false, vertical: true)
                    Text(pick.name)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
            }

            action
        }
        .padding(12)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.accentColor.opacity(0.08)))
    }

    @ViewBuilder private var action: some View {
        if let state, state.status == .downloading {
            VStack(alignment: .leading, spacing: 4) {
                ProgressView(value: state.fileProgress)
                HStack(spacing: 6) {
                    Text("Downloading \(state.percentFormatted) \(state.speedFormatted)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Spacer(minLength: 0)
                    Button("Cancel") {
                        downloads.cancel(pick.repoId)
                        appState.refreshModels()
                    }
                    .buttonStyle(.plain)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
        } else {
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    startDownload()
                } label: {
                    Text(Self.actionTitle(hasPartial: downloads.hasPartialDownload(pick.repoId),
                                          failed: state?.status == .failed))
                        .font(.subheadline.weight(.medium))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 3)
                }
                .buttonStyle(.borderedProminent)
                if let error = state?.error, state?.status == .failed {
                    Text(error)
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .lineLimit(2)
                }
            }
        }
    }

    private func startDownload() {
        // A GGUF pick names one quant out of a folder of them, so it takes the
        // quant path (which also pulls the ds4 MTP draft head). No starter tier
        // is a GGUF pick today — but the card is shared, and a card that
        // assumes safetensors would download the whole 87 GB repo the first
        // time one is.
        if let filename = pick.ggufFilename {
            downloads.startGguf(repoId: pick.repoId, ggufFilename: filename) { finish() }
        } else {
            downloads.start(repoId: pick.repoId) { finish() }
        }
    }

    private func finish() {
        appState.refreshModels()
        guard let dir = downloads.existingModelDir(for: pick.repoId) else { return }
        // A GGUF model's loadable path is the FILE, not the repo folder.
        let path = pick.ggufFilename.map { (dir as NSString).appendingPathComponent($0) } ?? dir
        Task {
            _ = await appState.useModelAndAwaitReady(atPath: path)
            onReady?()
        }
    }

    // MARK: - Copy (pure, so it's testable)

    /// The card's lead line: what this model IS, then its download size.
    ///
    /// Deliberately not the catalog's `tagline` — those are COMPARISONS ("The
    /// sweet spot", "Sharper reasoning") that only mean anything next to the
    /// other rows in the Model Browser, and read as noise on a card showing one
    /// model to someone who has never downloaded any.
    static func lead(for pick: RecommendedModelPick) -> String {
        let what: String
        switch pick.id {
        case "gemma-4-e2b":     what = "A small, quick assistant"
        case "gemma-4-e4b":     what = "A fast, capable assistant"
        case "gemma-4-12b":     what = "A capable all-round assistant"
        case "qwen36-27b-mtp":  what = "A strong assistant, great at code"
        default:                what = "A local AI assistant"
        }
        return "\(what) · \(String(format: "%.1f GB", pick.sizeGB))"
    }

    /// The button's words. A partial transfer says Resume rather than Download,
    /// so a user who quit mid-download isn't told they're starting over.
    static func actionTitle(hasPartial: Bool, failed: Bool) -> String {
        if failed { return hasPartial ? "Resume Download" : "Try Again" }
        if hasPartial { return "Resume Download" }
        return "Download"
    }
}
