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
            // Every kind says so now. An image used to be the exception because
            // its bytes rode the transcript, so the picture stayed on screen
            // with the file gone; it is drawn FROM the file today, and a
            // silently blank space is the one thing worse than a warning row.
            missingRow
        } else {
            switch ref.kind {
            case .audio: ChatAudioAttachment(ref: ref)
            case .video: ChatVideoAttachment(ref: ref)
            case .image: ChatImageAttachment(ref: ref)
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

/// A generated picture in the transcript, drawn from its file.
///
/// The file in `~/.mlx-serve/generations` is what the generator wrote, so this
/// shows the original rather than the re-encoded JPEG the history used to
/// carry beside it. Double-click opens it, the same gesture an uploaded image
/// has in the bubble above.
private struct ChatImageAttachment: View {
    let ref: ChatMediaRef

    /// Read ONCE on appear. Reading it in `body` would re-open the file on
    /// every render, and the transcript re-renders on every streamed token.
    @State private var image: NSImage?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            if let image {
                let box = ChatImagePreview.displaySize(
                    for: image,
                    maxHeight: ChatMetrics.generatedImageHeight,
                    maxWidth: ChatMetrics.generatedMediaMaxWidth)
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    // The picture's own box, so the rounded corners round the
                    // PICTURE — see `displaySize`.
                    .frame(width: box.width, height: box.height)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
                    .onTapGesture(count: 2) {
                        NSWorkspace.shared.open(URL(fileURLWithPath: ref.path))
                    }
                    .help("Double-click to open")
            }
            // Same shape as the clip below it: the prompt is what makes a
            // timestamped filename mean anything months later, so compact
            // tightens it to one line rather than dropping it. A button laid
            // over the picture read poorly on a light image and was a pattern
            // this app uses nowhere else.
            ChatMediaCaption(ref: ref, lines: ChatMetrics.compactMode ? 1 : 2)
        }
        // Leading, not the default centre: without it a picture narrower than
        // the cap sat centred in the cap and read as indented from a left edge
        // every other row in the transcript shares.
        .frame(maxWidth: ChatMetrics.generatedMediaMaxWidth, alignment: .leading)
        .onAppear {
            if image == nil { image = NSImage(contentsOfFile: ref.path) }
        }
    }
}

/// Reveal the generated file in Finder. Its own type because it appears both
/// under a row and beside a track's play button, and they must be the same
/// control.
private struct RevealInFinderButton: View {
    let path: String

    var body: some View {
        Button {
            NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
        } label: {
            Image(systemName: "folder")
                .font(.system(size: 11))
                .foregroundStyle(.secondary)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help("Reveal in Finder")
    }
}

/// Prompt + Reveal-in-Finder, shared by the image, audio and video rows. The
/// prompt is what makes a bare timestamped filename mean something months later.
private struct ChatMediaCaption: View {
    let ref: ChatMediaRef
    /// One line in compact mode, where the caption survives only because a
    /// video player has no free corner to put the button in.
    var lines: Int = 2

    var body: some View {
        HStack(spacing: 6) {
            Text(ref.prompt.isEmpty ? ref.filename : ref.prompt)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(lines)
                .truncationMode(.tail)
            Spacer(minLength: 4)
            RevealInFinderButton(path: ref.path)
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
                // Compact drops the caption, so the button joins the play row
                // rather than leaving the track with no way to its file.
                if ChatMetrics.compactMode { RevealInFinderButton(path: ref.path) }
            }
            if !ChatMetrics.compactMode { ChatMediaCaption(ref: ref) }
        }
        .padding(10)
        .frame(maxWidth: ChatMetrics.generatedMediaMaxWidth, alignment: .leading)
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
            // A fixed box: the clip's own ratio isn't known until the asset
            // loads, and a player resizing under the reply as it loads is worse
            // than letterboxing inside a steady frame.
            .frame(maxWidth: ChatMetrics.generatedMediaMaxWidth,
                   minHeight: ChatMetrics.generatedVideoHeight,
                   maxHeight: ChatMetrics.generatedVideoHeight)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            // Every corner of an AVPlayerView belongs to its own controls
            // (AirPlay and volume on top, transport below), so the button
            // stays under the player and the caption tightens to one line
            // instead.
            ChatMediaCaption(ref: ref, lines: ChatMetrics.compactMode ? 1 : 2)
                .frame(maxWidth: ChatMetrics.generatedMediaMaxWidth, alignment: .leading)
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
