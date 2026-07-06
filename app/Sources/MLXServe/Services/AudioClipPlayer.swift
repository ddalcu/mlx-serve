import AppKit

/// Output-only playback of generated audio clips (music tracks, voice output,
/// reference previews) via `NSSound`.
///
/// Why not `AVPlayer`/`AVAudioEngine`: on macOS 26 those bring up an
/// AVFoundation audio I/O unit whose voice-isolation evaluation consults the
/// microphone TCC service — which pops a "would like to access the Microphone"
/// prompt the first time you play a generated track, even though nothing needs
/// the mic (same CoreAudio-HAL mechanism the launch-time cue prompt hit; see
/// `SystemLoadingCue`). `NSSound` is a plain output path — it never opens a
/// capture stream, so playback can't trigger a mic prompt.
///
/// Tracks the currently-playing file so the history shelves can highlight it,
/// and clears it when playback finishes on its own.
@MainActor
final class AudioClipPlayer: NSObject, ObservableObject, NSSoundDelegate {
    /// The file currently playing (or paused), else nil. Drives shelf highlight.
    @Published private(set) var playingPath: String?

    private var sound: NSSound?

    /// Play `path` from the start, replacing whatever is playing.
    func play(_ path: String) {
        stop()
        guard let s = NSSound(contentsOfFile: path, byReference: true) else { return }
        s.delegate = self
        sound = s
        playingPath = path
        s.play()
    }

    /// Pause in place; `play` the same path (or `resume`) starts it again.
    func pause() {
        sound?.pause()
    }

    /// Resume a paused clip; no-op if nothing is loaded.
    func resume() {
        sound?.resume()
    }

    /// Stop and forget the current clip.
    func stop() {
        sound?.stop()
        sound = nil
        playingPath = nil
    }

    nonisolated func sound(_ sound: NSSound, didFinishPlaying finished: Bool) {
        Task { @MainActor in
            // Only clear if this is still the active clip (a new play() may have
            // already replaced it).
            if self.sound === sound {
                self.sound = nil
                self.playingPath = nil
            }
        }
    }
}
