import Foundation

/// What an in-chat media tool produces. Speech and music are separate KINDS for
/// the same reason they are separate tools — they take different arguments, cost
/// wildly different amounts of time, and read differently in the transcript.
enum MediaKind: String, Codable, Sendable, CaseIterable {
    case image, speech, music, video

    /// The tool's wire name — one place, so a rename can't half-land.
    var toolName: String {
        switch self {
        case .image:  return "generate_image"
        case .speech: return "generate_speech"
        case .music:  return "generate_music"
        case .video:  return "generate_video"
        }
    }

    /// How the thing that comes out is attached to a message.
    var attachmentKind: ChatMediaRef.Kind {
        switch self {
        case .image:           return .image
        case .speech, .music:  return .audio
        case .video:           return .video
        }
    }

    var icon: String {
        switch self {
        case .image:  return "photo"
        case .speech: return "waveform"
        case .music:  return "music.note"
        case .video:  return "film"
        }
    }

    /// Heading on the progress card. Distinct per kind — a shared title makes
    /// two very different jobs (6 seconds vs. four minutes) look like one.
    var progressTitle: String {
        switch self {
        case .image:  return "Generating image"
        case .speech: return "Synthesizing speech"
        case .music:  return "Composing music"
        case .video:  return "Rendering video"
        }
    }

    /// How the refusal sentence names the thing the model wanted.
    var article: String {
        switch self {
        case .image:  return "an image"
        case .speech: return "spoken audio"
        case .music:  return "a track"
        case .video:  return "a video"
        }
    }
}

/// The fast baseline a CHAT generation runs at.
///
/// Deliberately separate from `ImageGenSettings.load()` and friends: those are
/// the tray windows' sticky settings and stay the place for full-quality work.
/// A chat generation blocks decode on the one GPU while it runs, so it is a
/// quick preview — the model can adjust within the clamps below, and nothing
/// else.
enum MediaChatDefaults {
    /// Image step tier. `.fast` on every preset that has real steps; distilled
    /// models ignore tiers entirely (`stepsAreFixed`).
    static let imageQuality: QualityPreset = .fast

    /// Speech rate multiplier and the range the model may move inside.
    static let speechSpeed: Double = 1.0
    static let speechSpeedRange: ClosedRange<Double> = 0.5...2.0

    /// Track length. The Music window keeps its own 60 s default; 30 s is about
    /// half the wait for something you are listening to in a chat.
    static let musicSeconds = 30
    /// The server's own accepted range (`duration_seconds`), mirrored here so a
    /// model asking for 9000 gets a track rather than a 400.
    static let musicSecondsRange = 10...600

    /// Clip length. LTX tops out around 8 s of video, but a chat clip is a
    /// preview and every extra second is another ~30 s of GPU.
    static let videoSeconds: Double = 2
    static let videoMaxSeconds: Double = 4
    /// Preview steps are the MODEL's own fast tier, never a shared constant:
    /// the old `videoSteps = 8` was LTX's fast preset applied to everything,
    /// and H3 is not step-distilled — its validated floor is 16, so a chat
    /// preview burned 15+ minutes of GPU on an off-recipe clip. LTX still
    /// resolves to 8.
    static func videoSteps(for model: VideoModelPreset) -> Int {
        model.settings(.fast).steps
    }
    static let videoMode: VideoPipelineMode = .oneStage
}

/// Parsing and clamping for the four media tools, plus the per-turn budget.
///
/// Everything here is pure: it takes what the model emitted (all strings — tool
/// arguments arrive as a `[String: String]`) plus the preset it will run on, and
/// returns a request the service can execute. A model-supplied number never
/// reaches a service unclamped, because a 2B asking for 60 seconds of video is
/// not a hypothetical.
enum MediaToolArgs {

    /// A required argument the model didn't send. The message names the missing
    /// key and shows a call it can copy — an error steer with no example is the
    /// class this codebase already has a guard for.
    struct MissingArgument: LocalizedError {
        let tool: String
        let key: String
        let example: String
        var errorDescription: String? {
            "Error: \(tool) needs a non-empty \"\(key)\". Example: \(example)"
        }
    }

    // MARK: - Shared parsing

    /// Trimmed value for `key`, or nil when absent/blank.
    static func text(_ args: [String: String], _ key: String) -> String? {
        guard let raw = args[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else { return nil }
        return raw
    }

    private static func required(_ args: [String: String], _ key: String,
                                 tool: String, example: String) throws -> String {
        guard let v = text(args, key) else {
            throw MissingArgument(tool: tool, key: key, example: example)
        }
        return v
    }

    /// What a `size` argument asked for.
    enum RequestedShape: Equatable {
        /// Explicit pixels — `1024x1024`.
        case pixels(width: Int, height: Int)
        /// A bare ratio — `16:9`. Measured live: asked for "widescreen", a model
        /// sends `{"size":"16:9"}`. Refusing that spelling means silently
        /// ignoring the shape the user actually asked for.
        case aspect(Double)
    }

    /// Read a `size` argument in the spellings models emit: `1024x1024`,
    /// `1024X1024`, `1024 x 1024`, `1024×1024`, `16:9`. nil for anything else —
    /// a size we can't read must fall back, never guess.
    static func parseShape(_ raw: String?) -> RequestedShape? {
        guard let raw, !raw.isEmpty else { return nil }
        let normalized = raw
            .replacingOccurrences(of: "\u{00D7}", with: "x")
            .replacingOccurrences(of: "X", with: "x")
            .filter { !$0.isWhitespace }
        if let pair = twoNumbers(normalized, separator: "x") {
            return .pixels(width: Int(pair.0), height: Int(pair.1))
        }
        if let pair = twoNumbers(normalized, separator: ":") {
            return .aspect(pair.0 / pair.1)
        }
        return nil
    }

    private static func twoNumbers(_ s: String, separator: Character) -> (Double, Double)? {
        let parts = s.split(separator: separator)
        guard parts.count == 2,
              let a = Double(parts[0]), let b = Double(parts[1]),
              a > 0, b > 0, a.isFinite, b.isFinite else { return nil }
        return (a, b)
    }

    /// How far two aspects may differ and still count as "the same shape", as a
    /// log ratio. 1.75 and 16:9 are 0.016 apart and read identically; 4:3 and
    /// 16:9 are 0.28 apart and don't.
    static let aspectTolerance = 0.12

    /// The trained bucket for a bare `aspect` — SIZE taken from `anchor`.
    ///
    /// A ratio carries no size of its own, so the two obvious rankings are both
    /// wrong: by area it picks the smallest bucket (16:9 "is" 144 pixels), and
    /// by exact aspect it picks the LARGEST 16:9 the model has (measured: a chat
    /// "widescreen" landed on 2048×1152, eight times the pixels of the size the
    /// user actually works at, for a preview). So: every bucket whose shape
    /// matches within `aspectTolerance` is a candidate, and among those the one
    /// closest in area to the user's own saved size wins. When nothing matches
    /// the shape, the closest shape wins outright.
    static func nearest(aspect: Double, in options: [ResolutionOption],
                        anchor: ResolutionOption) -> ResolutionOption? {
        let usable = options.filter { !$0.isMatchSource && $0.width > 0 && $0.height > 0 }
        guard !usable.isEmpty else { return nil }
        let scored = usable.map {
            ($0, abs(log(Double($0.width) / Double($0.height) / aspect)))
        }
        guard let closestShape = scored.map(\.1).min() else { return nil }
        let cutoff = max(closestShape, aspectTolerance)
        let anchorArea = Double(max(anchor.width * anchor.height, 1))
        return scored.filter { $0.1 <= cutoff + 1e-9 }
            .min { abs(log(Double($0.0.width * $0.0.height) / anchorArea))
                 < abs(log(Double($1.0.width * $1.0.height) / anchorArea)) }?.0
    }

    /// The bucket a `size` argument resolves to for `model`, or `saved` when it
    /// asked for nothing readable.
    static func resolution(_ raw: String?, options: [ResolutionOption],
                           saved: ResolutionOption) -> ResolutionOption {
        switch parseShape(raw) {
        case .pixels(let w, let h):
            return nearest(w, h, in: options) ?? saved
        case .aspect(let r):
            return nearest(aspect: r, in: options, anchor: saved) ?? saved
        case nil:
            return saved
        }
    }

    /// The trained bucket closest to `(w, h)`. Models are trained at fixed
    /// resolutions; an off-grid size mostly works but produces visible
    /// artefacts, so a requested size SNAPS instead of being passed through.
    /// Ranks by relative shape error first, then by area, so "1920x1080" lands
    /// on a 16:9 bucket rather than merely the biggest one.
    static func nearest(_ w: Int, _ h: Int, in options: [ResolutionOption]) -> ResolutionOption? {
        let usable = options.filter { !$0.isMatchSource && $0.width > 0 && $0.height > 0 }
        guard !usable.isEmpty else { return nil }
        if let exact = usable.first(where: { $0.width == w && $0.height == h }) { return exact }
        let wantAspect = Double(w) / Double(h)
        let wantArea = Double(w * h)
        return usable.min { a, b in
            let aspectA = abs(log(Double(a.width) / Double(a.height) / wantAspect))
            let aspectB = abs(log(Double(b.width) / Double(b.height) / wantAspect))
            if abs(aspectA - aspectB) > 0.01 { return aspectA < aspectB }
            return abs(log(Double(a.width * a.height) / wantArea))
                 < abs(log(Double(b.width * b.height) / wantArea))
        }
    }

    // MARK: - Image

    /// The bucket a `size` argument resolves to. Absent or unreadable → the
    /// user's own saved resolution (a bucket is a preference, not a quality
    /// tier — only the step count is forced fast for chat).
    static func imageSize(_ raw: String?, model: ImageModelPreset,
                          saved: ResolutionOption) -> ResolutionOption {
        resolution(raw, options: model.resolutions, saved: saved)
    }

    static func image(_ args: [String: String], model: ImageModelPreset,
                      saved: ResolutionOption, seed: Int,
                      keepResident: Bool, lanId: String?) throws -> ImageGenRequest {
        let prompt = try required(args, "prompt", tool: "generate_image",
                                  example: #"{"prompt": "a red fox in the snow at golden hour"}"#)
        let resolution = imageSize(args["size"], model: model, saved: saved)
        // A distilled schedule is shut: 8 steps costs 2× and 12 costs 4× for a
        // DIFFERENT image, not a better one.
        let steps = model.stepsAreFixed ? model.fixedSteps
                                        : model.settings(MediaChatDefaults.imageQuality).steps
        return ImageGenRequest(
            model: model, prompt: prompt, seed: seed,
            width: resolution.width, height: resolution.height, steps: steps,
            keepResident: keepResident, lanModelId: lanId)
    }

    // MARK: - Speech

    static func speechSpeed(_ raw: String?) -> Double {
        guard let raw, let v = Double(raw), v.isFinite else { return MediaChatDefaults.speechSpeed }
        return min(max(v, MediaChatDefaults.speechSpeedRange.lowerBound),
                   MediaChatDefaults.speechSpeedRange.upperBound)
    }

    static func speech(_ args: [String: String], model: AudioModelPreset,
                       keepResident: Bool, lanId: String?) throws -> AudioGenRequest {
        let text = try required(args, "text", tool: "generate_speech",
                                example: #"{"text": "Your coffee is ready."}"#)
        return AudioGenRequest(
            model: model, text: text,
            speed: speechSpeed(args["speed"]),
            keepResident: keepResident, lanModelId: lanId)
    }

    // MARK: - Music

    static func musicSeconds(_ raw: String?, lyrics: String = "") -> Int {
        // Models write "45" but also "45 seconds" and "45.0" — read the leading
        // number rather than refusing a perfectly clear request.
        guard let raw, let v = Double(raw.prefix(while: { $0.isNumber || $0 == "." })), v.isFinite else {
            return clampMusicSeconds(secondsForLyrics(lyrics) ?? MediaChatDefaults.musicSeconds)
        }
        return clampMusicSeconds(Int(v))
    }

    private static func clampMusicSeconds(_ v: Int) -> Int {
        min(max(v, MediaChatDefaults.musicSecondsRange.lowerBound),
            MediaChatDefaults.musicSecondsRange.upperBound)
    }

    /// Seconds a lyric sheet needs, or nil when there is nothing sung. The
    /// flat 30 s default cut full songs off mid-verse (and the tool schema
    /// invited it: "omit for 30"), so an omitted duration is derived from the
    /// words instead: ~4 s a sung line plus 15 for intro/outro, rounded up to
    /// a quarter minute. Section tags are directives, not lines to sing.
    /// Erring long is the cheap direction — Music 3 treats the duration as an
    /// upper bound and ACE-Step fills the tail, while erring short truncates
    /// the song the user asked for.
    static func secondsForLyrics(_ lyrics: String) -> Int? {
        let sung = lyrics.split(separator: "\n").filter { line in
            let t = line.trimmingCharacters(in: .whitespaces)
            return !t.isEmpty && !(t.hasPrefix("[") && t.hasSuffix("]"))
        }
        guard !sung.isEmpty else { return nil }
        let raw = 15 + sung.count * 4
        return (raw + 14) / 15 * 15
    }

    /// Beats per minute, clamped to the engine's `[30,300]` — outside it the
    /// server returns a 400, so the clamp is the difference between a fast track
    /// and a failed turn. nil (omit the field) when there's no number to read:
    /// unlike duration there is no sensible default, and the engine's own
    /// convention for "let the model decide" is an ABSENT field.
    static func musicBpm(_ raw: String?) -> Int? {
        guard let raw,
              let v = Double(raw.trimmingCharacters(in: .whitespaces)
                                .prefix(while: { $0.isNumber || $0 == "." })),
              v.isFinite else { return nil }
        return min(max(Int(v), 30), 300)
    }

    /// The catalogue's own spelling of a key, or "" to omit.
    ///
    /// `keyscale` is NOT validated server-side — an unrecognised string goes
    /// straight into the conditioning as junk. So anything outside
    /// `MusicOptions.keyscales` is dropped and the model picks its own key,
    /// which is a better track than one conditioned on nonsense.
    static func musicKeyscale(_ raw: String?) -> String {
        guard let want = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !want.isEmpty else { return "" }
        return MusicOptions.keyscales.first { $0.lowercased() == want } ?? ""
    }

    /// Beats per bar on the wire, from either spelling a model will write: the
    /// picker's "4/4" or the bare "4". "" to omit.
    static func musicTimeSignature(_ raw: String?) -> String {
        guard let want = raw?.trimmingCharacters(in: .whitespacesAndNewlines),
              !want.isEmpty else { return "" }
        return MusicOptions.timeSignatures.first { $0.label == want || $0.value == want }?.value ?? ""
    }

    /// A language CODE, from either a code ("es") or a name ("Spanish").
    /// Anything we don't serve keeps `fallback` — the user's own setting — since
    /// conditioning the singer on a code the model invented is worse than
    /// ignoring the request.
    static func musicLanguage(_ raw: String?, fallback: String) -> String {
        guard let want = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
              !want.isEmpty else { return fallback }
        if want == "auto" { return "unknown" }
        return MusicOptions.languages.first {
            $0.code.lowercased() == want || $0.label.lowercased() == want
        }?.code ?? fallback
    }

    /// `language` is the user's saved vocal-language setting — the fallback when
    /// the model asks for one we don't serve.
    static func music(_ args: [String: String], model: MusicModelPreset,
                      language: String, keepResident: Bool,
                      lanId: String?) throws -> MusicGenRequest {
        let prompt = try required(args, "prompt", tool: "generate_music",
                                  example: #"{"prompt": "warm lo-fi hip hop with a mellow Rhodes piano"}"#)
        return MusicGenRequest(
            model: model, prompt: prompt,
            lyrics: text(args, "lyrics") ?? "",
            vocalLanguage: musicLanguage(args["vocal_language"], fallback: language),
            bpm: musicBpm(args["bpm"]),
            keyscale: musicKeyscale(args["keyscale"]),
            timesignature: musicTimeSignature(args["time_signature"]),
            durationSeconds: musicSeconds(args["duration_seconds"], lyrics: text(args, "lyrics") ?? ""),
            keepResident: keepResident, lanModelId: lanId)
    }

    // MARK: - Video

    /// Frame count for a requested clip length, clamped to the chat ceiling and
    /// landing on the model's own `8N+1` ladder (an off-ladder count is a 400).
    static func videoFrames(_ raw: String?, model: VideoModelPreset) -> Int {
        let requested = raw.flatMap { Double($0.prefix(while: { $0.isNumber || $0 == "." })) }
        let seconds = min(max(requested?.isFinite == true ? requested! : MediaChatDefaults.videoSeconds,
                              1.0 / Double(max(model.fps, 1))),
                          MediaChatDefaults.videoMaxSeconds)
        let ladderFloor = model.frameOptions.first ?? 9
        let cap = model.framesCovering(durationSeconds: MediaChatDefaults.videoMaxSeconds) ?? ladderFloor
        let n = model.framesCovering(durationSeconds: seconds) ?? ladderFloor
        return min(n, cap)
    }

    static func video(_ args: [String: String], model: VideoModelPreset,
                      saved: ResolutionOption, keepResident: Bool,
                      lanId: String?) throws -> VideoGenRequest {
        let prompt = try required(args, "prompt", tool: "generate_video",
                                  example: #"{"prompt": "a timelapse of clouds over a mountain range"}"#)
        let resolution = resolution(args["size"], options: model.resolutions, saved: saved)
        return VideoGenRequest(
            model: model, prompt: prompt,
            width: resolution.width, height: resolution.height,
            numFrames: videoFrames(args["seconds"], model: model),
            fps: model.fps,
            mode: MediaChatDefaults.videoMode,
            steps: MediaChatDefaults.videoSteps(for: model),
            cfgScale: 1.0,
            keepResident: keepResident,
            lanModelId: lanId)
    }
}

/// ONE media generation per user turn.
///
/// The web console learned this live: a 2B answered a single request with four
/// GPU generations. A per-round cap can't bound that (the model just calls again
/// next round); a budget spanning the whole turn can. The refusal has to be a
/// SENTENCE the model can relay — a bare code or an empty result just gets
/// retried until the loop gives up.
///
/// The turn is a TOKEN the caller passes, not a `reset()` it has to remember. It
/// started as a reset call at the top of the agent loop and the headless test
/// harness — a second driver on the same engine — inherited a spent budget and
/// refused every generation it was ever asked for, permanently and silently. A
/// token can't be forgotten: a driver that doesn't have one can't call at all,
/// and any unseen token is a new turn.
///
/// The spend is PER TOKEN, not "reset when the token changes": with the
/// multi-turn engine two turns generate CONCURRENTLY and interleave their
/// claims, and a single stored token reset the budget on every alternation —
/// unlimited generations for both. Recent tokens are kept (bounded) so an
/// interleaved claim always finds its own tally.
struct MediaTurnBudget {
    static let limit = 1
    /// How many turn tallies to retain. Only turns that actually claimed are
    /// stored; the cap exists so a long-lived engine can't grow unbounded.
    static let retainedTurns = 16
    private var spentByTurn: [UUID: Int] = [:]
    private var order: [UUID] = []   // insertion order, for eviction

    /// nil = go ahead. Non-nil = the refusal to hand back as the tool result.
    mutating func claim(_ kind: MediaKind, turn: UUID) -> String? {
        if spentByTurn[turn] == nil {
            spentByTurn[turn] = 0
            order.append(turn)
            if order.count > Self.retainedTurns {
                spentByTurn.removeValue(forKey: order.removeFirst())
            }
        }
        guard let n = spentByTurn[turn], n < Self.limit else { return Self.refusal(for: kind) }
        spentByTurn[turn] = n + 1
        return nil
    }

    /// Leads with the FACT, not the rule.
    ///
    /// The first wording opened with the policy and asked the model to "tell the
    /// user what you made" — and a 4B, having been refused, reported that both
    /// things had been generated (live 2026-07-28). A refused tool result is the
    /// model's only evidence that something did not happen, so it has to say
    /// that first and in words the model can copy.
    static func refusal(for kind: MediaKind) -> String {
        "NOT GENERATED — \(kind.article) was not created. Only one media generation is allowed per message and this message already used it. Do not call any generate_ tool again in this turn. Tell the user plainly that you made only the first item, that \(kind.article) was NOT made, and that they can ask for it in a new message."
    }
}
