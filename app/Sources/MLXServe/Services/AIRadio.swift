import Foundation

/// The nudge each AI Radio track is pushed with. A station keeps one theme, so
/// without a nudge the model writes the same song again: every track names two
/// aspects to change, rotating through the list so neighbours never share one.
enum RadioDirection {
    static let aspects: [[String]] = [
        ["a slower tempo", "a laid-back mid tempo", "an upbeat tempo", "a fast, driving tempo"],
        ["a warm, nostalgic mood", "a bright, hopeful mood", "a darker, moodier feel", "a dreamy, floating atmosphere",
         "a tense, cinematic mood", "a playful, light feel"],
        ["a piano lead", "a warm electric guitar lead", "soft synth pads with a plucked lead", "strings and cello",
         "a saxophone or flute lead", "mallets and marimba", "an analog bass-led groove", "an acoustic guitar lead"],
        ["a different sub-genre of the theme", "an unexpected sub-genre blend", "a more traditional take",
         "a more experimental take"],
        ["lo-fi, tape-saturated production", "clean, spacious production", "gritty, punchy drums",
         "wide ambient reverb", "a minimal, sparse arrangement", "a lush, layered arrangement"],
        ["a swung groove", "a straight four-on-the-floor pulse", "a syncopated rhythm", "no drums, a beatless texture"],
        ["calm, steady energy", "slowly building energy", "high, peak energy", "gentle, winding-down energy"],
        ["a 1970s flavour", "a 1980s flavour", "a 1990s flavour", "a modern, contemporary sound"],
    ]

    /// Two aspects to change for track `index`; `salt` makes each station walk the lists differently.
    static func nudge(forTrack index: Int, salt: Int) -> String {
        (0..<2).map { k in
            let aspect = (index * 2 + k + salt) % aspects.count
            let values = aspects[aspect]
            // A small integer hash, so the value does not cycle with the aspect.
            var h = UInt32(truncatingIfNeeded: index) &* 2_654_435_761 &+ UInt32(aspect) &* 40_503 &+ UInt32(truncatingIfNeeded: salt) &* 97
            h ^= h >> 15; h = h &* 2_246_822_519; h ^= h >> 13
            return values[Int(h % UInt32(values.count))]
        }.joined(separator: "; ")
    }

    /// The prompt when no chat model is there to write one.
    static func fallbackPrompt(theme: String, nudge: String) -> String {
        "\(theme), \(nudge.replacingOccurrences(of: "; ", with: ", "))"
    }
}

/// The radio strip doubles as a progress bar: a dim fill grows behind the status text, and
/// while there is no number yet (writing the prompt, loading the model) a block sweeps to and fro.
enum RadioStrip {
    static let blockWidth: CGFloat = 24
    /// Skin pixels per second.
    static let blockSpeed: Double = 40

    /// Whole pixels of fill for `progress` (0...1; nil = nothing to show).
    static func fillWidth(progress: Double?, width: CGFloat) -> CGFloat {
        guard let progress else { return 0 }
        return (width * CGFloat(min(1, max(0, progress)))).rounded(.down)
    }

    /// Where the sweeping block's left edge is `time` seconds in: whole pixels, bouncing between the edges.
    static func blockOffset(time: Double, width: CGFloat) -> CGFloat {
        let travel = Int(max(1, width - blockWidth))
        let position = Int(time * blockSpeed) % (2 * travel)
        return CGFloat(position <= travel ? position : 2 * travel - position)
    }
}

/// Tracks generated ahead of playback, and the prompts that made them.
struct RadioQueue {
    static let historyLimit = 8

    private(set) var ready: [String] = []
    private(set) var recentPrompts: [String] = []

    /// One track buffered is enough: generation shares the GPU with everything else.
    var needsTrack: Bool { ready.isEmpty }

    mutating func push(prompt: String, path: String) {
        ready.append(path)
        recentPrompts.append(prompt)
        if recentPrompts.count > Self.historyLimit { recentPrompts.removeFirst(recentPrompts.count - Self.historyLimit) }
    }

    mutating func pop() -> String? { ready.isEmpty ? nil : ready.removeFirst() }
}

/// AI Radio: writes a fresh prompt per track from one theme, generates it while
/// the previous one plays, and keeps going until stopped. One instance, like the
/// player, so it outlives the pane.
@MainActor
final class AIRadio: ObservableObject {
    static let shared = AIRadio()

    @Published private(set) var isOn = false
    /// One line for the player's radio strip: what it is doing right now. Plain, not published:
    /// the strip reads it as it draws, so a progress tick never re-renders the pane.
    private(set) var status = ""
    /// How far the track being made is, 0...1; nil while there is no number (the strip sweeps a block).
    private(set) var progress: Double?

    private var queue = RadioQueue()
    private var loop: Task<Void, Never>?
    private var service: MusicGenService?
    private var server: ServerManager?
    private var unloadModelId: String?

    private let player = AudioClipPlayer.shared
    private static let maxFailuresInARow = 3

    func start(theme: String, template: MusicGenRequest, service: MusicGenService, server: ServerManager,
               downloads: DownloadManager, appState: AppState) {
        let theme = theme.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isOn, !theme.isEmpty else { return }
        isOn = true
        status = L10n.text("Starting…")
        progress = nil
        queue = RadioQueue()
        self.service = service
        self.server = server
        // The station keeps the model resident (see `stationRequest`) and gives it back at the end.
        unloadModelId = template.keepResident || template.lanModelId != nil ? nil
            : ServerManager.resolveModelDir(repo: template.model.repo).map { ($0 as NSString).lastPathComponent }
        let request = Self.stationRequest(from: template)
        player.onNaturalFinish = { [weak self] in self?.trackEnded() ?? false }
        loop = Task { await run(theme: theme, request: request, downloads: downloads, appState: appState) }
    }

    /// A track a station makes waits behind a GPU that is also playing the last one, so tracks
    /// stay short and Music 3 / YuE2 take fewer refinement steps than their defaults.
    nonisolated static let maxTrackSeconds = 120
    nonisolated static let maxSteps = 16

    /// The pane's settings, bent to what an endless station needs.
    nonisolated static func stationRequest(from template: MusicGenRequest) -> MusicGenRequest {
        var request = template
        // Reloading a multi-GB model between tracks would stall the stream.
        request.keepResident = true
        request.task = .text2music
        request.srcAudioPath = nil
        // Each prompt decides its own tempo and key.
        request.bpm = nil; request.keyscale = ""; request.timesignature = ""
        request.durationSeconds = min(request.durationSeconds, maxTrackSeconds)
        // A model that ends its own song is bounded by its lyrics, not by a cap that would cut them.
        request.autoLength = request.model.supportsAutoLength
        if request.model.supportsSteps {
            request.steps = min(request.steps ?? request.model.fixedSteps, maxSteps)
        }
        return request
    }

    func stop() {
        guard isOn else { return }
        isOn = false
        status = ""
        progress = nil
        loop?.cancel()
        loop = nil
        player.onNaturalFinish = nil
        queue = RadioQueue()
        let service = service, server = server, id = unloadModelId
        service?.cancel()
        self.service = nil; self.server = nil; unloadModelId = nil
        if let id, let server {
            Task {
                for _ in 0..<20 where service?.isRunning == true { try? await Task.sleep(for: .milliseconds(250)) }
                try? await server.unloadModel(id: id)
            }
        }
    }

    /// Next key: jump to the buffered track now.
    func skip() {
        guard isOn else { return }
        if !playNext() { status = L10n.text("Waiting for the next track…") }
    }

    private func trackEnded() -> Bool {
        guard isOn else { return false }
        if !playNext() { status = L10n.text("Waiting for the next track…") }
        return true
    }

    @discardableResult
    private func playNext() -> Bool {
        guard let path = queue.pop() else { return false }
        player.play(path)
        return true
    }

    private func run(theme: String, request: MusicGenRequest, downloads: DownloadManager, appState: AppState) async {
        guard let service, let server else { return }
        let salt = Int.random(in: 0..<1000)
        var index = 0, failures = 0
        while isOn, !Task.isCancelled {
            if !queue.needsTrack {
                try? await Task.sleep(for: .milliseconds(500))
                continue
            }
            status = L10n.text("Writing the next track…")
            progress = nil
            let nudge = RadioDirection.nudge(forTrack: index, salt: salt)
            let (prompt, lyrics) = await compose(theme: theme, nudge: nudge, request: request, appState: appState)
            guard isOn, !Task.isCancelled else { break }

            var next = request
            next.prompt = prompt
            next.lyrics = lyrics
            next.instrumental = request.model.supportsInstrumental
            next.seed = -1
            status = L10n.text("Composing…")
            progress = nil
            service.generate(next, server: server, downloads: downloads)
            while service.isRunning, isOn {
                if case .running(let step, let total, let message) = service.phase {
                    status = message
                    progress = total > 0 ? Double(step) / Double(total) : nil
                }
                try? await Task.sleep(for: .milliseconds(400))
            }
            guard isOn, !Task.isCancelled else { break }

            if case .completed(let path) = service.phase {
                failures = 0
                index += 1
                queue.push(prompt: prompt, path: path)
                if player.playingPath == nil { playNext() }
                status = L10n.text("Next track ready")
                progress = 1
            } else {
                failures += 1
                var why = L10n.text("Music generation failed.")
                if case .failed(let message) = service.phase { why = message }
                if failures >= Self.maxFailuresInARow {
                    isOn = false
                    player.onNaturalFinish = nil
                    status = L10n.format("AI Radio stopped: %@", why)
                    self.service = nil; self.server = nil
                    return
                }
                status = L10n.format("Retrying: %@", why)
                progress = nil
                try? await Task.sleep(for: .seconds(3))
            }
        }
    }

    /// The next track's style prompt (and lyrics, for a model that always sings),
    /// written by the chat model; the local nudge stands in when there is none.
    private func compose(theme: String, nudge: String, request: MusicGenRequest,
                         appState: AppState) async -> (prompt: String, lyrics: String) {
        let model = request.model
        let ask = MusicPromptRewriter.radioRequest(theme: theme, recent: queue.recentPrompts, nudge: nudge,
                                                   family: model.family, instrumental: model.supportsInstrumental)
        var prompt = RadioDirection.fallbackPrompt(theme: theme, nudge: nudge)
        if let reply = try? await AgentComposer.complete(userText: ask.user, systemPrompt: ask.system,
                                                         appState: appState, maxTokens: 512) {
            let cleaned = PromptRewriter.clean(reply)
            if !cleaned.isEmpty { prompt = cleaned }
        }
        guard !model.supportsInstrumental else { return (prompt, "") }
        let lyricsAsk = MusicPromptRewriter.radioLyricsRequest(theme: theme, style: prompt, nudge: nudge,
                                                               recent: queue.recentPrompts, family: model.family,
                                                               fallbackLanguage: request.vocalLanguage)
        let reply = try? await AgentComposer.complete(userText: lyricsAsk.user, systemPrompt: lyricsAsk.system,
                                                      appState: appState, maxTokens: 1024)
        return (prompt, reply.map(PromptRewriter.clean) ?? "")
    }
}
