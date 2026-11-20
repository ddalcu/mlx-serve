import AppKit
import Foundation

/// One template from blanks to a finished clip: lines → review → render →
/// captions. Owned by `AppState`, because the Templates pane unmounts when the
/// window switches modes and a four-minute render must not go with it.
@MainActor
final class TemplateRunner: ObservableObject {
    enum Phase: Equatable { case filling, writing, review, rendering, captioning, done, failed(String) }

    @Published private(set) var template: MediaTemplate?
    /// The gallery is showing; a template in progress stays loaded behind it.
    @Published var onGallery = true
    @Published var fill = TemplateFill()
    @Published var shape: ShareShape = .story
    @Published var lines: [String: String] = [:]
    @Published private(set) var phase: Phase = .filling
    @Published private(set) var progress: MediaGenProgress?
    @Published private(set) var preview: NSImage?
    @Published private(set) var clip: URL?
    @Published private(set) var captioned: URL?
    /// The looping GIF of a meme template.
    @Published private(set) var gif: URL?
    @Published var showsCaptions = true
    /// Said under the lines when they are the template's starters, not the model's.
    @Published private(set) var linesNote: String?
    /// Why a video that should have captions has none, when the speech models were there to try.
    @Published private(set) var captionNote: String?
    /// The history entry of the video on the stage.
    private var entryId: UUID?
    private var seed = Int.random(in: 1...999_999)
    private var job: Task<Void, Never>?
    /// True until a job's task has returned, Cancel or not: a cancelled render
    /// still unwinds (encoding, unloading its model), and nothing may start over it.
    @Published private(set) var working = false
    let history: TemplateHistory

    init(history: TemplateHistory? = nil) {
        self.history = history ?? TemplateHistory()
    }

    var isBusy: Bool { working || [.writing, .rendering, .captioning].contains(phase) }

    /// Runs `body` as the one job; a cancelled job's late writes are dropped by
    /// the `Task.checkCancellation()` calls inside it.
    private func start(_ body: @escaping @MainActor () async -> Void) {
        working = true
        job = Task {
            await body()
            working = false
        }
    }

    /// The finished clip: captioned when there is one and captions are on.
    var result: URL? { showsCaptions ? (captioned ?? clip) : clip }

    /// What the stage plays. Nothing until the run is done: the clean clip
    /// started under the caption burn and then played again with captions.
    var playable: URL? { Self.playable(phase: phase, result: result) }

    nonisolated static func playable(phase: Phase, result: URL?) -> URL? { phase == .done ? result : nil }

    /// Where `t` can be made now: the one answer the page shows and `make` routes by.
    static func readiness(_ t: MediaTemplate, shape: ShareShape, app: AppState) -> TemplateReadiness {
        let need: UInt64, seconds: Double?
        switch t.kind {
        case .video:
            let m = MediaTemplate.videoModel
            let frames = TemplateText.frames(seconds: t.seconds)
            need = H3Plan.peakBytes(model: m, width: shape.width, height: shape.height, frames: frames, fast: false)
            seconds = H3TimeEstimate.seconds(model: m, width: shape.width, height: shape.height, frames: frames,
                                             steps: m.fixedSteps ?? 8, fast: false)
        case .image:
            need = UInt64(MediaTemplate.imageModel.approxRAMGB) << 30
            seconds = nil
        case .audio:
            need = UInt64(MediaTemplate.songModel.approxRAMGB) << 30
            seconds = nil
        }
        return .of(onDisk: app.downloads.bundleReady(t.bundle), neededBytes: need,
                   usableBytes: SystemMemoryInfo.current().usableBytes, seconds: seconds,
                   peer: TemplateReadiness.peer(for: t, in: app.server.lanModels(capability: t.lanCapability)))
    }

    func open(_ t: MediaTemplate) {
        onGallery = false
        guard template != t, !isBusy else { return }
        template = t
        fill = t.defaultFill()
        shape = t.defaultShape
        lines = [:]
        linesNote = nil
        phase = .filling
        clip = nil
        captioned = nil
        gif = nil
        preview = nil
        seed = Int.random(in: 1...999_999)
    }

    /// Back to the blanks, keeping what was typed.
    func editFill() {
        guard !isBusy else { return }
        phase = .filling
    }

    func writeLines(app: AppState) {
        guard let t = template, !isBusy else { return }
        if t.writesOwnLines {
            lines = t.ownLines(fill)
            linesNote = nil
            phase = .review
            return
        }
        phase = .writing
        let fill = fill
        start { [self] in
            var written: [String: String]? = nil
            do {
                let reply = try await AgentComposer.complete(userText: t.ask(fill), systemPrompt: TemplateText.writerSystem,
                                                             appState: app, maxTokens: t.kind == .audio ? 700 : 300)
                written = TemplateText.parseLines(reply, keys: t.lineKeys(fill))
            } catch is CancellationError {
                return
            } catch {}
            guard !Task.isCancelled else { return }
            lines = written ?? t.fallbackLines(fill)
            linesNote = written == nil
                ? L10n.text("No chat model answered, so these are starter lines. Edit them, or pick a chat model and write again.")
                : nil
            phase = .review
        }
    }

    /// Starts the render; false when it cannot start (busy, a blank left, a
    /// line unwritten, or no model here or on the network).
    @discardableResult
    func make(app: AppState) -> Bool {
        guard let t = template, !isBusy else { return false }
        fill = fill.resolved(in: app.cast.members)
        guard t.missing(fill) == nil, !t.linesMissing(fill, lines) else { return false }
        let peer: String?
        switch Self.readiness(t, shape: shape, app: app) {
        case .ready: peer = nil
        case .onPeer(let id): peer = id
        case .needsDownload, .tooBig: return false
        }
        let (fill, lines, shape, seed) = (fill, lines, shape, seed)
        let made = TemplateHistoryEntry(template: t.rawValue, clip: "", cast: fill.cast.mapValues(\.id),
                                        answers: fill.answers, lines: lines, shape: shape.rawValue, seed: seed)
        phase = .rendering
        progress = nil
        preview = nil
        clip = nil
        captioned = nil
        gif = nil
        captionNote = nil
        entryId = made.id
        let top = fill.text("top")
        let onProgress: (MediaGenProgress) -> Void = { [weak self] in self?.progress = $0 }
        start { [self] in
            do {
                let url: URL
                switch t.kind {
                case .video:
                    url = URL(fileURLWithPath: try await app.videoGen.generateForAgent(
                        Self.request(t, fill: fill, lines: lines, shape: shape, seed: seed, cast: app.cast, peer: peer),
                        server: app.server, onProgress: onProgress, onPreview: { [weak self] in self?.preview = $0 }))
                case .image:
                    url = URL(fileURLWithPath: try await app.imageGen.generateForAgent(
                        Self.imageRequest(t, fill: fill, shape: shape, seed: seed, cast: app.cast, peer: peer),
                        server: app.server, onProgress: onProgress))
                    try Task.checkCancellation()
                    preview = NSImage(contentsOf: url)
                case .audio:
                    let song = URL(fileURLWithPath: try await app.musicGen.generateForAgent(
                        Self.songRequest(t, fill: fill, lines: lines, seed: seed, peer: peer), server: app.server, onProgress: onProgress))
                    url = Self.songVideoURL(song)
                    let photo = t.cast(fill).first.map { app.cast.photoURL($0) } ?? song
                    let cover = await songCover(t, fill: fill, photo: photo, seed: seed, app: app, onProgress: onProgress)
                    try Task.checkCancellation()
                    try await SongVideo.make(cover: cover, title: lines["title"] ?? "", song: song, shape: shape, to: url)
                }
                try Task.checkCancellation()
                clip = url
                phase = .captioning
                switch t.kind {
                case .video: await burnCaptions(url, t, top: top, app: app)
                case .image: captioned = Self.memeText(url, top: top, bottom: fill.text("bottom"))
                case .audio: break
                }
                if t.makesGIF {
                    let out = url.deletingPathExtension().appendingPathExtension("gif")
                    let looped = (try? await LoopingGIF.make(from: captioned ?? url, to: out)) != nil
                    try Task.checkCancellation()
                    gif = looped ? out : nil
                }
                try Task.checkCancellation()
                var entry = made
                entry.clip = url.path
                entry.captioned = captioned?.path
                entry.gif = gif?.path
                history.add(entry)
                phase = .done
            } catch {
                // Cancel has already put the page back.
                if !Task.isCancelled, !(error is CancellationError) { phase = .failed(error.localizedDescription) }
            }
        }
        return true
    }

    /// The song's cover: their photo restyled where the template asks for it
    /// and the picture model is here or on the network, else the photo itself.
    private func songCover(_ t: MediaTemplate, fill: TemplateFill, photo: URL, seed: Int, app: AppState,
                           onProgress: @escaping (MediaGenProgress) -> Void) async -> URL {
        guard let prompt = t.songCoverPrompt(fill) else { return photo }
        let m = MediaTemplate.imageModel
        let peer = app.downloads.bundleReady(m.bundle) ? nil
            : TemplateReadiness.peer(repo: m.repo, capability: "image", in: app.server.lanModels(capability: "image"))
        guard app.downloads.bundleReady(m.bundle) || peer != nil else { return photo }
        var r = Self.imageRequest(t, fill: fill, shape: .square, seed: seed, cast: app.cast, peer: peer)
        r.prompt = prompt
        guard let path = try? await app.imageGen.generateForAgent(r, server: app.server, onProgress: onProgress) else { return photo }
        return URL(fileURLWithPath: path)
    }

    /// Where a song's shareable video goes: beside the song, under a name that
    /// leaves the Music tab's own sidecar for the song alone.
    nonisolated static func songVideoURL(_ song: URL) -> URL {
        song.deletingPathExtension().appendingPathExtension("song.mp4")
    }

    /// A video from the history, back on its template page with the cast that
    /// is still here.
    func restore(_ e: TemplateHistoryEntry, cast: CastStore) {
        guard !isBusy, let t = e.videoTemplate else { return }
        template = t
        fill = TemplateFill(cast: e.cast.compactMapValues { id in cast.members.first { $0.id == id } },
                            answers: e.answers)
        lines = e.lines
        shape = e.shareShape
        seed = e.seed
        clip = URL(fileURLWithPath: e.clip)
        captioned = e.captioned.map { URL(fileURLWithPath: $0) }
        gif = e.gif.map { URL(fileURLWithPath: $0) }
        entryId = e.id
        captionNote = nil
        preview = nil
        progress = nil
        linesNote = nil
        phase = .done
        onGallery = false
    }

    /// The same lines on a new seed.
    func anotherTake(app: AppState) {
        let previous = seed
        seed = Int.random(in: 1...999_999)
        if !make(app: app) { seed = previous }
    }

    /// Captions for a finished video that has none: the speech models arrived after it was made.
    func addCaptions(app: AppState) {
        guard let t = template, let clip, phase == .done, captioned == nil else { return }
        phase = .captioning
        start { [self] in
            await burnCaptions(clip, t, top: fill.text("top"), app: app)
            guard !Task.isCancelled else { return }
            if let captioned, let entryId { history.setCaptioned(entryId, captioned.path) }
            phase = .done
        }
    }

    private func burnCaptions(_ clip: URL, _ t: MediaTemplate, top: String, app: AppState) async {
        let speech = t.captions && CaptionModels.of(app.localModels, engine: app.serverOptions.transcribeEngine) == .ready
        let burned = await Self.caption(clip, speech: speech, top: top, app: app)
        guard !Task.isCancelled else { return }
        captioned = burned
        captionNote = speech && captioned == nil && top.isEmpty
            ? L10n.text("No captions: no speech was heard in this video.") : nil
    }

    func cancel() {
        job?.cancel()
        switch phase {
        case .writing: phase = .filling
        case .rendering: phase = .review
        // Captions added to a saved video go back to it; a first render was never saved.
        case .captioning: phase = history.entries.contains { $0.id == entryId } ? .done : .review
        default: break
        }
    }

    /// Advanced Editor: hands what the template would make to its full pane,
    /// for anyone who wants every knob.
    func openInEditor(app: AppState) {
        guard let t = template else { return }
        let peer: String? = if case .onPeer(let id) = Self.readiness(t, shape: shape, app: app) { id } else { nil }
        switch t.kind {
        case .video:
            let request = Self.request(t, fill: fill, lines: lines, shape: shape, seed: seed, cast: app.cast, peer: peer)
            var s = VideoGenSettings.load()
            s.modelId = LanPick.persisted(lanModel: peer, presetId: request.model.id)
            s.prompt = request.prompt
            s.refImagePaths = request.refImagePaths
            s.seed = request.seed
            s.numFrames = request.numFrames
            s.steps = request.steps
            s.resolutionId = ResolutionOption.custom.id
            s.customWidth = request.width
            s.customHeight = request.height
            s.storyboardMode = false
            s.save()
            app.showCreate(.video)
        case .image:
            Self.imageSettings(t, fill: fill, shape: shape, seed: seed, cast: app.cast, peer: peer).save()
            app.showCreate(.image)
        case .audio:
            Self.musicSettings(t, fill: fill, lines: lines, seed: seed, peer: peer).save()
            UserDefaults.standard.set(AudioGenView.Tab.music.rawValue, forKey: "audioGenTab")
            app.showCreate(.audio)
        }
    }

    static func imageSettings(_ t: MediaTemplate, fill: TemplateFill, shape: ShareShape, seed: Int, cast: CastStore,
                              peer: String?) -> ImageGenSettings {
        let r = imageRequest(t, fill: fill, shape: shape, seed: seed, cast: cast, peer: peer)
        var s = ImageGenSettings.load()
        s.modelId = LanPick.persisted(lanModel: peer, presetId: r.model.id)
        s.prompt = r.prompt
        s.sourcePath = r.initImagePath
        s.refPaths = []
        s.editMode = true
        s.seed = seed
        s.resolutionId = ResolutionOption.custom.id
        s.customWidth = r.width
        s.customHeight = r.height
        return s
    }

    static func musicSettings(_ t: MediaTemplate, fill: TemplateFill, lines: [String: String], seed: Int,
                              peer: String?) -> MusicGenSettings {
        let r = songRequest(t, fill: fill, lines: lines, seed: seed, peer: peer)
        var s = MusicGenSettings.load()
        s.modelId = LanPick.persisted(lanModel: peer, presetId: r.model.id)
        s.prompt = r.prompt
        s.lyrics = r.lyrics
        s.durationSeconds = r.durationSeconds
        s.autoLength = false
        s.instrumental = false
        s.task = .text2music
        s.seed = seed
        return s
    }

    static func imageRequest(_ t: MediaTemplate, fill: TemplateFill, shape: ShareShape, seed: Int, cast: CastStore,
                             peer: String?) -> ImageGenRequest {
        let m = MediaTemplate.imageModel
        let size = shape.imageSize
        var r = ImageGenRequest(model: m, prompt: t.imagePrompt(fill), seed: seed, width: size.width, height: size.height,
                                steps: m.qualityProfiles[m.defaultQuality]?.steps ?? 8)
        r.initImagePath = t.cast(fill).first.map { cast.photoURL($0).path }
        r.editMode = true
        r.lanModelId = peer
        return r
    }

    static func songRequest(_ t: MediaTemplate, fill: TemplateFill, lines: [String: String], seed: Int,
                            peer: String?) -> MusicGenRequest {
        var r = MusicGenRequest(model: MediaTemplate.songModel, prompt: t.songStyle(fill), lyrics: lines["lyrics"] ?? "")
        r.durationSeconds = t.seconds
        r.seed = seed
        r.lanModelId = peer
        return r
    }

    /// The picture with its meme text on, or nil when there is none to add.
    private static func memeText(_ picture: URL, top: String, bottom: String) -> URL? {
        guard !top.isEmpty || !bottom.isEmpty else { return nil }
        let out = picture.deletingPathExtension().appendingPathExtension("meme.png")
        return (try? CaptionBurner.burn(image: picture, top: top, bottom: bottom, to: out)) != nil ? out : nil
    }

    static func request(_ t: MediaTemplate, fill: TemplateFill, lines: [String: String], shape: ShareShape,
                        seed: Int, cast: CastStore, peer: String?) -> VideoGenRequest {
        var r = VideoGenRequest(model: MediaTemplate.videoModel, prompt: t.document(fill, lines: lines), seed: seed,
                                width: shape.width, height: shape.height,
                                numFrames: TemplateText.frames(seconds: t.seconds), fps: 24,
                                mode: .oneStage, steps: MediaTemplate.videoModel.fixedSteps ?? 8, cfgScale: 1.0)
        r.refImagePaths = t.cast(fill).map { cast.photoURL($0).path }
        r.livePreview = true
        r.lanModelId = peer
        return r
    }

    /// The clip with its words burned in (captions of what is said when
    /// `speech`, meme text across the top when `top` is set), or nil when there
    /// is nothing to burn or a step fails: the clip is the result, the text a bonus.
    private static func caption(_ clip: URL, speech: Bool, top: String, app: AppState) async -> URL? {
        var words: [CaptionWord] = []
        if speech, let model = VoiceASR.modelPath(app.localModels, engine: app.serverOptions.transcribeEngine),
           let wav = try? await CaptionBurner.speechWAV(from: clip),
           let data = try? await TrayTranscript.post(wav, model: model, server: app.server.baseURL,
                                                     format: "verbose_json", words: true) {
            words = CaptionPlan.words(fromVerboseJSON: data)
        }
        guard !words.isEmpty || !top.isEmpty else { return nil }
        let out = clip.deletingPathExtension().appendingPathExtension("captioned.mp4")
        do {
            try await CaptionBurner.burn(video: clip, words: words, top: top, to: out)
            return out
        } catch {
            return nil
        }
    }
}
