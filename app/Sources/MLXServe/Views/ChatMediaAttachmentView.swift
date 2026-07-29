import SwiftUI
import AppKit
import AVKit
import AVFoundation

/// The ONE AVPlayerView wrapper in the app.
///
/// SwiftUI's `VideoPlayer` fatal-aborts inside generic metadata resolution under
/// state transitions on macOS 26.4, so every video surface goes through
/// AVPlayerView instead — and through this one type, so the workaround can't
/// half-apply. Used by the Video generation window and by the chat transcript.
struct AVPlayerViewRepresentable: NSViewRepresentable {
    let player: AVPlayer

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.player = player
        view.controlsStyle = .inline
        return view
    }

    func updateNSView(_ nsView: AVPlayerView, context: Context) {
        if nsView.player !== player {
            nsView.player = player
        }
    }
}

/// A generated track or clip in the transcript.
///
/// The file lives in `~/.mlx-serve/generations`, which the user owns and may
/// empty — so a missing file is a normal state with its own row, never a blank
/// space or a player that silently does nothing.
struct ChatMediaAttachmentView: View {
    let ref: ChatMediaRef

    var body: some View {
        if !ref.exists {
            // An image's BYTES live in the transcript, so the picture is still
            // on screen when its file is gone — a warning row under a perfectly
            // visible image is noise, and the Reveal button would open nothing.
            // A track or a clip has nothing left to show, so it says so.
            if ref.kind != .image { missingRow }
        } else {
            switch ref.kind {
            case .audio: ChatAudioAttachment(ref: ref)
            case .video: ChatVideoAttachment(ref: ref)
            // The picture itself rides `ChatMessage.images`; this is the caption
            // + Reveal-in-Finder row that sits under it, the same one a track
            // and a clip get.
            case .image: ChatMediaCaption(ref: ref).frame(maxWidth: 400)
            }
        }
    }

    private var missingRow: some View {
        Label("\(ref.filename) — file no longer on disk", systemImage: "questionmark.folder")
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(.quaternary.opacity(0.4))
            .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

/// Prompt + Reveal-in-Finder, shared by the audio and video rows. The prompt is
/// what makes a bare timestamped filename mean something months later.
private struct ChatMediaCaption: View {
    let ref: ChatMediaRef

    var body: some View {
        HStack(spacing: 6) {
            Text(ref.prompt.isEmpty ? ref.filename : ref.prompt)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .truncationMode(.tail)
            Spacer(minLength: 4)
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: ref.path)])
            } label: {
                Image(systemName: "folder")
                    .font(.system(size: 11))
                    .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)
            .help("Reveal in Finder")
        }
    }
}

/// Compact player row for a generated clip or track.
///
/// `AudioClipPlayer` (NSSound) rather than AVPlayer: on macOS 26 an AVFoundation
/// audio I/O unit's voice-isolation evaluation consults the microphone TCC
/// service, which pops a mic-permission prompt the first time you press play.
/// The player is SHARED, so starting one clip stops the last — two tracks
/// overlapping in a transcript is never what anyone wants.
private struct ChatAudioAttachment: View {
    let ref: ChatMediaRef
    @ObservedObject private var player = AudioClipPlayer.shared
    /// Read ONCE on appear. Reading it in `body` would open the WAV on every
    /// view update, and a streaming reply re-renders the transcript ~20×/s.
    @State private var duration: String?

    private var isPlaying: Bool { player.playingPath == ref.path }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 10) {
                Button {
                    isPlaying ? player.stop() : player.play(ref.path)
                } label: {
                    Image(systemName: isPlaying ? "stop.circle.fill" : "play.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.tint)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(isPlaying ? "Stop" : "Play")

                VStack(alignment: .leading, spacing: 1) {
                    Text(ref.filename)
                        .font(.caption.weight(.medium))
                        .lineLimit(1)
                        .truncationMode(.middle)
                    if let duration {
                        Text(duration)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                Spacer(minLength: 4)
            }
            ChatMediaCaption(ref: ref)
        }
        .padding(10)
        .frame(maxWidth: 420, alignment: .leading)
        .background(Color(.controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .onAppear { if duration == nil { duration = Self.durationText(ref.path) } }
    }

    /// `m:ss` from the file's own header. Best-effort — an unreadable clip just
    /// shows no duration rather than blocking the row.
    static func durationText(_ path: String) -> String? {
        guard let file = try? AVAudioFile(forReading: URL(fileURLWithPath: path)) else { return nil }
        let rate = file.processingFormat.sampleRate
        guard rate > 0 else { return nil }
        let seconds = Int((Double(file.length) / rate).rounded())
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }
}

/// Inline video player for a generated clip.
private struct ChatVideoAttachment: View {
    let ref: ChatMediaRef
    @State private var player: AVPlayer?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Group {
                if let player {
                    AVPlayerViewRepresentable(player: player)
                } else {
                    Color.black.opacity(0.15)
                }
            }
            .frame(maxWidth: 420, minHeight: 220, maxHeight: 260)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            ChatMediaCaption(ref: ref)
                .frame(maxWidth: 420)
        }
        // Built on appear, not in the initializer: a transcript can hold many
        // clips and an AVPlayer per row would be built during every view update.
        .onAppear {
            if player == nil { player = AVPlayer(url: URL(fileURLWithPath: ref.path)) }
        }
        .onDisappear { player?.pause() }
    }
}

/// The in-flight media generation, under the tool-call row that started it.
///
/// Determinate when the engine reports a total, indeterminate when it doesn't
/// (TTS length is model-determined). The elapsed clock is a
/// `TimelineView(.periodic)` — NOT a repeating animation or a Timer: a
/// `repeatForever` animation is what wedges the tray popover (see `VoicePulse`).
struct MediaProgressCard: View {
    let progress: MediaGenProgress

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Image(systemName: progress.kind.icon)
                        .font(.callout)
                        .foregroundStyle(.tint)
                    Text(progress.title)
                        .font(.caption.weight(.semibold))
                    Spacer(minLength: 8)
                    TimelineView(.periodic(from: progress.startedAt, by: 1)) { context in
                        Text(progress.elapsedText(now: context.date))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                Group {
                    if let fraction = progress.fraction {
                        ProgressView(value: fraction)
                    } else {
                        ProgressView()
                            .progressViewStyle(.linear)
                    }
                }
                .frame(maxWidth: .infinity)

                Text(progress.detailText)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
            .padding(.horizontal, ChatMetrics.bubblePaddingH)
            .padding(.vertical, ChatMetrics.bubblePaddingV)
            .frame(maxWidth: 420, alignment: .leading)
            .background(Color(.controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: ChatMetrics.bubbleCornerRadius))

            Spacer(minLength: 60)
        }
    }
}
