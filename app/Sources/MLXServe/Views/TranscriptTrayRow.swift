import AppKit
import CoreMediaIO
import SwiftUI

/// The tray's transcript switch: on records the microphone (and the desktop
/// audio when ticked) to disk, off transcribes it with the local Qwen3-ASR,
/// its speakers named, and opens it as Markdown under the notes a chat model
/// writes. It can also start when an app turns the camera on, stop after a
/// silence, and pick up a recording a quit or a crash left behind.
@MainActor
final class TrayTranscript: ObservableObject {
    static let shared = TrayTranscript()
    nonisolated static let asrRepo = "mlx-community/Qwen3-ASR-0.6B-8bit"
    /// What a transcript needs, by model type, and the repo the tray fetches for each.
    nonisolated static let packs = [(type: "qwen3_asr", repo: asrRepo), (type: "qwen3_forced_aligner", repo: DownloadManager.alignerRepo), (type: "nemotron_diarization", repo: DownloadManager.diarizerRepo)]
    nonisolated static let folder = URL(fileURLWithPath: NSString(string: "~/.mlx-serve/transcripts").expandingTildeInPath)
    /// The audio of recordings not yet transcribed, written as it arrives.
    nonisolated static let recordings = folder.appendingPathComponent(".recording")

    /// `model` is the ASR chosen when recording began, so a change to the model list cannot strand the stop.
    enum Phase: Equatable { case idle, starting, recording(since: Date, model: String), transcribing, note(String) }
    @Published private(set) var phase = Phase.idle
    /// Not observed: its meter would redraw the tray many times a second.
    private let recorder = AudioRecorder()
    private let desktopAudio = DesktopAudio()
    private weak var app: AppState?
    private var watch: Timer?
    /// The recording in progress: its name (its start time) and its files.
    private var stamp = ""
    private var mic: RecordingFile?
    private var desk: RecordingFile?
    private var cameraWasOn = false
    /// Recordings taken up this run, so one that keeps failing is tried once.
    private var tried: Set<String> = []

    /// Also record what the Mac plays, from the moment it is ticked; remembered.
    @Published var desktop = UserDefaults.standard.bool(forKey: "transcriptDesktopAudio") {
        didSet {
            UserDefaults.standard.set(desktop, forKey: "transcriptDesktopAudio")
            if case .recording(let since, _) = phase { desktop ? startDesktop(since) : desktopAudio.stop() }
        }
    }

    var isRecording: Bool {
        if case .recording = phase { return true }
        return false
    }

    /// Hook up to the app: the server, the models and the settings the
    /// automatic start, stop and recovery read every few seconds.
    func attach(_ app: AppState) {
        self.app = app
        guard watch == nil else { return }
        watch = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            Task { @MainActor in await self?.tick() }
        }
    }

    /// The ASR to transcribe with, once the server runs and every pack is whole.
    private var readyASR: String? {
        guard let app, app.server.status == .running, Self.missingRepos(app.localModels).isEmpty else { return nil }
        return app.localModels.first { $0.modelType == "qwen3_asr" && $0.defect == nil && !$0.isDownloading }?.path
    }

    private func tick() async {
        guard let app else { return }
        let options = app.serverOptions
        switch phase {
        case .recording:
            let heard = max(mic?.lastHeard ?? .distantPast, desk?.lastHeard ?? .distantPast)
            guard Self.silentTooLong(since: heard, now: Date(), minutes: options.transcriptAutoStopMinutes) else { return }
            await stop()
            if phase == .idle { phase = .note(L10n.format("Stopped after %d minutes of silence", options.transcriptAutoStopMinutes)) }
        case .idle, .note:
            guard let asr = readyASR else { return }
            if let left = Self.leftovers(in: Self.recordings).first(where: { !tried.contains($0) }) {
                await finish(left, model: asr)
                return
            }
            // A call that turns the camera on starts one; a stop while it stays on does not restart it.
            let camera = options.transcriptAutoStartCamera && Self.cameraInUse()
            defer { cameraWasOn = camera }
            if camera && !cameraWasOn { Task { await start() } }
        default:
            break
        }
    }

    func start() async {
        guard let app, let model = readyASR else { return }
        // A second click while the microphone prompt is up must not start a second recording.
        switch phase { case .idle, .note: phase = .starting; default: return }
        guard await AudioRecorder.requestPermission() else { phase = .note("Microphone access is off"); return }
        // Voice mode would hear, and answer, what is being transcribed.
        for voice in [app.voice, app.chatVoice] where voice.isActive { voice.end() }
        let since = Date()
        stamp = Self.stamp(of: since)
        do {
            let file = try RecordingFile(url: Self.recordings.appendingPathComponent(stamp + ".mic.pcm"))
            mic = file
            recorder.sink = { file.write($0) }
            try recorder.start()
            phase = .recording(since: since, model: model)
            if desktop { startDesktop(since) }
        } catch {
            mic?.close()
            if let url = mic?.url { try? FileManager.default.removeItem(at: url) }
            mic = nil
            phase = .note(error.localizedDescription)
        }
    }

    /// A tap that will not start (no System Audio Recording access) unticks the box.
    private func startDesktop(_ since: Date) {
        do {
            let file = try desk ?? RecordingFile(url: Self.recordings.appendingPathComponent(stamp + ".desktop.pcm"))
            desk = file
            try desktopAudio.start(at: Int(Date().timeIntervalSince(since) * AudioRecorder.targetSampleRate), into: file)
        } catch {
            desktop = false
        }
    }

    func stop() async {
        guard case .recording(_, let model) = phase else { return }
        desktopAudio.stop()
        _ = recorder.stop()
        recorder.sink = nil
        mic?.close()
        desk?.close()
        (mic, desk) = (nil, nil)
        await finish(stamp, model: model)
    }

    /// Transcribe the recording `stamp` from its files and open the result.
    private func finish(_ stamp: String, model: String) async {
        guard let app else { return }
        tried.insert(stamp)
        phase = .transcribing
        let notesModel = Self.notesModel(setting: app.serverOptions.transcriptNotesModel, chatModel: app.selectedModelPath, models: app.localModels)
        do {
            let saved = try await Self.save(stamp, model: model, server: app.server.baseURL, notesModel: notesModel)
            guard let file = saved.file else { phase = .note("Nothing was heard"); return }
            NSWorkspace.shared.open(file)
            phase = saved.notesFailure.map { .note(L10n.format("Saved without notes: %@", $0)) } ?? .idle
        } catch {
            phase = .note(error.localizedDescription)
        }
    }

    /// Off the main actor: the recording `stamp` read back from disk, transcribed
    /// with its speakers, its notes written, the Markdown saved, and its audio
    /// files removed. A failed transcription keeps the audio as `<stamp>.wav`.
    /// No file when nothing was heard.
    nonisolated static func save(_ stamp: String, model: String, server: String, notesModel: String?) async throws -> (file: URL?, notesFailure: String?) {
        let parts = ["mic", "desktop"].map { recordings.appendingPathComponent("\(stamp).\($0).pcm") }
        let removeParts = { for part in parts { try? FileManager.default.removeItem(at: part) } }
        let mic = RecordingFile.read(parts[0]), desktop = RecordingFile.read(parts[1])
        guard !mic.isEmpty else { removeParts(); return (nil, nil) }
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let segments: [(start: Double, text: String, speaker: String?)]
        do {
            segments = try await transcribeRecording(mic, desktop, model: model, server: server, speakers: true)
        } catch {
            let wav = AudioReference.wavData(fromMonoFloat: mix(mic, desktop), sampleRate: Int(AudioRecorder.targetSampleRate))
            if (try? wav.write(to: folder.appendingPathComponent(stamp + ".wav"))) != nil { removeParts() }
            throw error
        }
        guard !segments.isEmpty else { removeParts(); return (nil, nil) }
        var notes: String?
        var notesFailure: String?
        if let notesModel {
            do { notes = try await self.notes(for: segments, model: notesModel, server: server) } catch { notesFailure = error.localizedDescription }
        }
        let file = folder.appendingPathComponent(stamp + ".md")
        try markdown(title: title(of: stamp), notes: notes, segments: segments).write(to: file, atomically: true, encoding: .utf8)
        removeParts()
        return (file, notesFailure)
    }

    /// One request for the whole recording (the server cuts a long one), the
    /// speakers named when there is more than one.
    nonisolated static func transcribeRecording(_ mic: [Float], _ desktop: [Float], model: String, server: String, speakers: Bool) async throws -> [(start: Double, text: String, speaker: String?)] {
        let rate = Int(AudioRecorder.targetSampleRate)
        // Models downloaded since the server started (this ASR, its aligner and diarizer) register first.
        var rescan = URLRequest(url: URL(string: server + "/v1/models/rescan")!)
        rescan.httpMethod = "POST"
        _ = try? await URLSession.shared.data(for: rescan)
        let wav = AudioReference.wavData(fromMonoFloat: mix(mic, desktop), sampleRate: rate)
        let heard = try await transcribe(wav, model: model, server: server, speakers: speakers).filter { !$0.text.isEmpty }
        let letters = heard.compactMap(\.speaker)
        let names = Set(letters).count > 1 ? speakerNames(letters, you: onlyOnMic(heard, mic: mic, desktop: desktop, rate: rate)) : [:]
        return heard.map { ($0.start, $0.text, $0.speaker.flatMap { names[$0] }) }
    }

    /// The chat model that writes the notes: the one chosen in Settings, else
    /// the chat model in use; nil when notes are off or there is none.
    nonisolated static func notesModel(setting: String, chatModel: String, models: [LocalModel]) -> String? {
        guard setting != ServerOptions.notesOff else { return nil }
        let usable = { (path: String) in models.contains { $0.path == path && $0.isChatPickable } }
        if usable(setting) { return setting }
        return usable(chatModel) ? chatModel : nil
    }

    private nonisolated static let notesPrompt = """
        You turn a meeting transcript into notes. Write in the language of the transcript, in Markdown, in \
        this order: a first line `# ` with a short title for the meeting; `## Summary` with a few sentences; \
        `## Decisions` as a list; `## Action items` as a checklist (`- [ ] who: what`). Leave out a section \
        with nothing in it. Use only what the transcript says, and call speakers by their labels or by a \
        name the conversation gives them.
        """

    /// The transcript as the notes model reads it: one line a segment, with its time and speaker.
    nonisolated static func notesInput(_ segments: [(start: Double, text: String, speaker: String?)]) -> String {
        segments.map { "[\(clock($0.start))] " + ($0.speaker.map { "\($0): " } ?? "") + $0.text }.joined(separator: "\n")
    }

    /// The notes `model` writes for a transcript: Markdown led by a `# title`.
    nonisolated static func notes(for segments: [(start: Double, text: String, speaker: String?)], model: String, server: String) async throws -> String {
        let body: [String: Any] = [
            "model": model, "max_tokens": 2048, "temperature": 0.3, "enable_thinking": false,
            "messages": [["role": "system", "content": notesPrompt], ["role": "user", "content": notesInput(segments)]],
        ]
        var request = URLRequest(url: URL(string: server + "/v1/chat/completions")!, timeoutInterval: 3600)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw serverError(data, "The notes could not be written") }
        struct Reply: Decodable {
            struct Choice: Decodable { struct Message: Decodable { let content: String? }; let message: Message }
            let choices: [Choice]
        }
        let notes = (try JSONDecoder().decode(Reply.self, from: data).choices.first?.message.content ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !notes.isEmpty else { throw serverError(Data(), "The notes could not be written") }
        return notes
    }

    nonisolated static func silentTooLong(since heard: Date, now: Date, minutes: Int) -> Bool {
        minutes > 0 && now.timeIntervalSince(heard) >= Double(minutes * 60)
    }

    /// A recording's name: when it started, safe in a file name.
    nonisolated static func stamp(of date: Date) -> String {
        let format = DateFormatter()
        format.locale = Locale(identifier: "en_US_POSIX")
        format.dateFormat = "yyyy-MM-dd HH.mm.ss"
        return format.string(from: date)
    }

    nonisolated static func title(of stamp: String) -> String {
        stamp.replacingOccurrences(of: ".", with: ":")
    }

    /// The recordings in `dir` still waiting to be transcribed, oldest first.
    nonisolated static func leftovers(in dir: URL) -> [String] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []
        return names.filter { $0.hasSuffix(".mic.pcm") }.map { String($0.dropLast(".mic.pcm".count)) }.sorted()
    }

    /// True while any app has a camera running (Core Media IO; no camera permission needed).
    nonisolated static func cameraInUse() -> Bool {
        var address = CMIOObjectPropertyAddress(mSelector: CMIOObjectPropertySelector(kCMIOHardwarePropertyDevices),
                                                mScope: CMIOObjectPropertyScope(kCMIOObjectPropertyScopeGlobal),
                                                mElement: CMIOObjectPropertyElement(kCMIOObjectPropertyElementMain))
        let system = CMIOObjectID(kCMIOObjectSystemObject)
        var size: UInt32 = 0
        guard CMIOObjectGetPropertyDataSize(system, &address, 0, nil, &size) == 0, size > 0 else { return false }
        var devices = [CMIOObjectID](repeating: 0, count: Int(size) / MemoryLayout<CMIOObjectID>.size)
        guard CMIOObjectGetPropertyData(system, &address, 0, nil, size, &size, &devices) == 0 else { return false }
        address.mSelector = CMIOObjectPropertySelector(kCMIODevicePropertyDeviceIsRunningSomewhere)
        return devices.contains { device in
            var running: UInt32 = 0
            var used = UInt32(MemoryLayout<UInt32>.size)
            return CMIOObjectGetPropertyData(device, &address, 0, nil, used, &used, &running) == 0 && running != 0
        }
    }

    /// The repos to fetch before a transcript can start: none once every pack is
    /// on disk, whole and not still arriving. Without the ASR only it, since its
    /// aligner and diarizer follow it.
    nonisolated static func missingRepos(_ models: [LocalModel]) -> [String] {
        let missing = packs.filter { pack in !models.contains { $0.modelType == pack.type && $0.defect == nil && !$0.isDownloading } }
        return missing.first?.type == "qwen3_asr" ? [asrRepo] : missing.map(\.repo)
    }

    /// The microphone with the desktop audio on top, for as long as the microphone ran.
    nonisolated static func mix(_ mic: [Float], _ desktop: [Float]) -> [Float] {
        desktop.isEmpty ? mic : mic.indices.map { mic[$0] + ($0 < desktop.count ? desktop[$0] : 0) }
    }

    /// The speaker heard on the microphone and not in the desktop audio: meeting
    /// apps never play your own voice back, so that voice is yours. Nil without
    /// desktop audio, or when more than one voice is only on the microphone.
    nonisolated static func onlyOnMic(_ segments: [Segment], mic: [Float], desktop: [Float], rate: Int) -> String? {
        guard !desktop.isEmpty else { return nil }
        var loudness: [String: (mic: Float, desktop: Float)] = [:]
        for s in segments {
            guard let who = s.speaker else { continue }
            let lo = min(mic.count, Int(s.start * Double(rate))), hi = min(mic.count, max(lo, Int(s.end * Double(rate))))
            var sum = loudness[who] ?? (0, 0)
            for i in lo..<hi {
                sum.mic += abs(mic[i])
                if i < desktop.count { sum.desktop += abs(desktop[i]) }
            }
            loudness[who] = sum
        }
        let mine = loudness.filter { $0.value.mic > 0 && $0.value.desktop < $0.value.mic / 2 }
        return mine.count == 1 ? mine.first?.key : nil
    }

    /// The names the file gives the server's speaker letters: "You" for `you`,
    /// the others lettered again in the order they first speak.
    nonisolated static func speakerNames(_ letters: [String], you: String?) -> [String: String] {
        var names: [String: String] = [:]
        var next: UInt8 = 65
        for letter in letters where names[letter] == nil {
            if letter == you {
                names[letter] = L10n.text("You")
            } else {
                names[letter] = L10n.format("Speaker %@", String(UnicodeScalar(next)))
                next += 1
            }
        }
        return names
    }

    /// The transcript file: when it started, then the text, each segment led by
    /// its time in the recording when there is more than one, and by its
    /// speaker where the speaker changes. Notes go above it, under their own
    /// title when they have one.
    nonisolated static func markdown(title: String, notes: String? = nil, segments: [(start: Double, text: String, speaker: String?)]) -> String {
        var paragraphs: [String] = []
        var last: String?
        for s in segments {
            let who = s.speaker.flatMap { $0 == last ? nil : " \($0):" } ?? ""
            last = s.speaker ?? last
            paragraphs.append("**\(clock(s.start))\(who)** \(s.text)")
        }
        let transcript = segments.count == 1 ? segments[0].text : paragraphs.joined(separator: "\n\n")
        guard let notes else { return "# \(title)\n\n\(transcript)\n" }
        let titled = notes.hasPrefix("# ")
        let head = titled ? "\(notes.prefix { $0 != "\n" })\n\n\(title)" : "# \(title)"
        let rest = titled ? notes.drop { $0 != "\n" }.trimmingCharacters(in: .whitespacesAndNewlines) : notes
        return "\(head)\n\n\(rest)\n\n## \(L10n.text("Transcript"))\n\n\(transcript)\n"
    }

    nonisolated static func clock(_ seconds: Double) -> String {
        let s = Int(seconds)
        return s < 3600 ? String(format: "%02d:%02d", s / 60, s % 60) : String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60)
    }

    /// A `verbose_json` or `diarized_json` segment; only the second names a speaker.
    struct Segment: Decodable { let start: Double; let end: Double; let text: String; let speaker: String? }
    private struct Verbose: Decodable { let segments: [Segment] }
    private struct Failure: Decodable {
        struct Detail: Decodable { let message: String }
        let error: Detail
    }

    /// The server's own error message, else `fallback`.
    private nonisolated static func serverError(_ data: Data, _ fallback: String) -> NSError {
        let message = (try? JSONDecoder().decode(Failure.self, from: data))?.error.message ?? fallback
        return NSError(domain: "Transcript", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    /// One request: what the server heard, timed from the start of `wav`, with
    /// its speakers when asked.
    nonisolated static func transcribe(_ wav: Data, model: String, server: String, speakers: Bool) async throws -> [Segment] {
        let data = try await post(wav, model: model, server: server, format: speakers ? "diarized_json" : "verbose_json")
        return try JSONDecoder().decode(Verbose.self, from: data).segments
    }

    /// `POST /v1/audio/transcriptions` with `wav` as the file; the body of a 200, else the server's own error.
    nonisolated static func post(_ wav: Data, model: String, server: String, format: String) async throws -> Data {
        let boundary = UUID().uuidString
        var body = Data()
        for (name, value) in [("model", model), ("response_format", format)] {
            body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(value)\r\n".utf8))
        }
        body.append(Data("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"recording.wav\"\r\nContent-Type: audio/wav\r\n\r\n".utf8))
        body.append(wav)
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))
        // An hour of meeting is minutes of work before the first byte comes back.
        var request = URLRequest(url: URL(string: server + "/v1/audio/transcriptions")!, timeoutInterval: 3600)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await URLSession.shared.upload(for: request, from: body)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw serverError(data, "Transcription failed") }
        return data
    }
}

/// A third switch beside Voice and the Quick Launcher; its folder button opens the saved transcripts.
struct TranscriptTrayRow: View {
    @EnvironmentObject var appState: AppState
    @EnvironmentObject var downloads: DownloadManager
    @ObservedObject private var transcript = TrayTranscript.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            row
            if recording {
                Toggle("Also record desktop audio", isOn: $transcript.desktop)
                    .toggleStyle(.checkbox)
                    .font(.app(.caption))
                    .padding(.leading, 26)
            }
        }
    }

    private var row: some View {
        let missing = TrayTranscript.missingRepos(appState.localModels)
        let downloading = TrayTranscript.packs.contains { downloads.downloads[$0.repo]?.status == .downloading }
        return TrayFeatureRow(icon: "record.circle", title: "Transcript", subtitle: subtitle(ready: missing.isEmpty, downloading: downloading), isOn: recording, tint: .red) {
            Button {
                try? FileManager.default.createDirectory(at: TrayTranscript.folder, withIntermediateDirectories: true)
                NSWorkspace.shared.open(TrayTranscript.folder)
            } label: {
                Image(systemName: "folder")
            }
            .buttonStyle(.plain)
            .help("Open the transcripts folder")
            if missing.isEmpty || recording {
                Toggle("", isOn: Binding(get: { recording }, set: { on in
                    Task {
                        if on { await transcript.start() } else { await transcript.stop() }
                    }
                }))
                .labelsHidden()
                .toggleStyle(.switch)
                .controlSize(.small)
                // Stopping always works; starting needs the server.
                .disabled(transcript.phase == .transcribing || transcript.phase == .starting || !recording && appState.server.status != .running)
            } else {
                // The progress shows in the Downloads tab, which also closes the tray.
                Button("Download") {
                    for repo in missing { downloads.start(repoId: repo) { appState.refreshModels() } }
                    appState.showModels(.downloads)
                }
                .controlSize(.small)
                .disabled(downloading)
            }
        }
    }

    private var recording: Bool {
        if case .recording = transcript.phase { return true }
        return false
    }

    private func subtitle(ready: Bool, downloading: Bool) -> String {
        switch transcript.phase {
        case .idle, .starting: return ready ? "Record, then open as Markdown" : downloading ? "Downloading speech models…" : "Needs speech models"
        case .recording(let since, _): return L10n.format("Recording since %@", since.formatted(date: .omitted, time: .shortened))
        case .transcribing: return "Transcribing…"
        case .note(let note): return note
        }
    }
}
