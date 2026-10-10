import AppKit
import AVFoundation

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
    /// One player for the chat transcript. A transcript can hold many clips and
    /// each row is its own view, so a per-row player would let two tracks play
    /// over each other — the media panes keep their own instance because each is
    /// a single pane with a single shelf.
    static let shared = AudioClipPlayer()

    /// The file currently playing (or paused), else nil. Drives shelf highlight.
    @Published private(set) var playingPath: String?

    /// The last clip played, kept after it stops: the Music pane's player
    /// shows it, so a finished track stays loaded instead of vanishing.
    @Published private(set) var loadedPath: String?

    /// Paused mid-clip: `playingPath` stays set, so the shelf keeps the row lit
    /// and the player shows the frozen clock.
    @Published private(set) var isPaused = false

    /// Asked when a clip ends on its own; answer true to take over from here
    /// (AI Radio), else `autoNext` decides.
    var onNaturalFinish: (() -> Bool)?

    /// The playlist the player steps through: the shelf on screen sets it.
    var queue: [String] = []

    @Published var autoNext = UserDefaults.standard.bool(forKey: "mlxamp.autoNext") {
        didSet { UserDefaults.standard.set(autoNext, forKey: "mlxamp.autoNext") }
    }
    @Published var shuffle = UserDefaults.standard.bool(forKey: "mlxamp.shuffle") {
        didSet { UserDefaults.standard.set(shuffle, forKey: "mlxamp.shuffle") }
    }

    /// 0...1, applied to the playing clip and to every one after it.
    @Published var volume = 0.8 { didSet { sound?.volume = Float(volume) } }

    private var sound: NSSound?

    /// Seconds into the playing clip; 0 when stopped.
    var elapsed: TimeInterval { sound?.currentTime ?? 0 }

    /// Jump within the loaded clip, starting it if it is stopped.
    func seek(to seconds: TimeInterval) {
        guard let path = loadedPath else { return }
        if sound == nil { play(path) }
        sound?.currentTime = max(0, seconds)
    }

    struct Info { let seconds: TimeInterval; let sampleRate: Double; let channels: Int; let kbps: Int }

    private static var infoCache: [String: Info] = [:]

    /// Length and format of an audio file, cached; nil when it cannot be read.
    static func info(_ path: String) -> Info? {
        if let hit = infoCache[path] { return hit }
        guard let file = try? AVAudioFile(forReading: URL(fileURLWithPath: path)) else { return nil }
        let rate = file.processingFormat.sampleRate
        let seconds = rate > 0 ? Double(file.length) / rate : 0
        let bytes = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? Int) ?? 0
        let info = Info(seconds: seconds, sampleRate: rate, channels: Int(file.processingFormat.channelCount),
                        kbps: seconds > 0 ? Int(Double(bytes) * 8 / seconds / 1000) : 0)
        infoCache[path] = info
        return info
    }

    /// m:ss, the way a playlist writes a length.
    static func clock(_ seconds: TimeInterval) -> String {
        let t = Int(max(0, seconds))
        return String(format: "%d:%02d", t / 60, t % 60)
    }

    /// Play `path` from the start, replacing whatever is playing.
    func play(_ path: String) {
        stop()
        guard let s = NSSound(contentsOfFile: path, byReference: true) else { return }
        s.delegate = self
        s.volume = Float(volume)
        sound = s
        playingPath = path
        loadedPath = path
        s.play()
    }

    /// Stop and forget the current clip.
    func stop() {
        sound?.stop()
        sound = nil
        playingPath = nil
        isPaused = false
    }

    func togglePause() {
        guard let sound else { return }
        if isPaused { sound.resume() } else { sound.pause() }
        isPaused.toggle()
    }

    /// Play the track `delta` rows from `from` (default: the loaded one) — a
    /// random other one when shuffling. Past either end it stops, unless `wrap`.
    func advance(_ delta: Int, from path: String? = nil, wrap: Bool = false) {
        guard let current = path ?? loadedPath, let i = queue.firstIndex(of: current) else { return }
        if shuffle, queue.count > 1 {
            play(queue.filter { $0 != current }.randomElement()!)
        } else if queue.indices.contains(i + delta) {
            play(queue[i + delta])
        } else if wrap {
            play(queue[((i + delta) % queue.count + queue.count) % queue.count])
        }
    }

    nonisolated func sound(_ sound: NSSound, didFinishPlaying finished: Bool) {
        Task { @MainActor in
            // Only clear if this is still the active clip (a new play() may have
            // already replaced it).
            if self.sound === sound {
                self.sound = nil
                self.playingPath = nil
                if finished {
                    if self.onNaturalFinish?() == true { return }
                    if self.autoNext { self.advance(1, wrap: true) }
                }
            }
        }
    }
}
