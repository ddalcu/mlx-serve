import AVFoundation
import XCTest
@testable import MLXCore

/// The tray transcript: the mix, who is speaking, the file it writes.
final class TrayTranscriptTests: XCTestCase {
    /// Desktop audio joins where it was ticked and stops with the microphone.
    func testDesktopAudioIsAddedOnTopOfTheMicrophone() {
        XCTAssertEqual(TrayTranscript.mix([0.1, 0.2, 0.3, 0.4], [0, 0.5]), [0.1, 0.7, 0.3, 0.4])
        XCTAssertEqual(TrayTranscript.mix([0.1, 0.2], [0, 0.5, 0.9]), [0.1, 0.7])
        XCTAssertEqual(TrayTranscript.mix([0.1], []), [0.1])
    }

    func testTheFileIsItsStartThenTimedParagraphs() {
        XCTAssertEqual(TrayTranscript.markdown(title: "2026-10-13 18:40:05", segments: [(0, "Hello there.", nil), (65.5, "Bye.", nil), (3725, "Late.", nil)]),
                       "# 2026-10-13 18:40:05\n\n**00:00** Hello there.\n\n**01:05** Bye.\n\n**1:02:05** Late.\n")
        XCTAssertEqual(TrayTranscript.markdown(title: "t", segments: [(0, "Only one.", nil)]), "# t\n\nOnly one.\n")
    }

    /// A paragraph names its speaker when the speaker changes.
    func testSpeakersAreNamedWhereTheyChange() {
        let segments: [(start: Double, text: String, speaker: String?)] = [(0, "Shall we start?", "Speaker A"), (4, "The plan first.", "Speaker A"), (9, "Yes.", "You"), (12, "Good.", "Speaker A")]
        XCTAssertEqual(TrayTranscript.markdown(title: "t", segments: segments),
                       "# t\n\n**00:00 Speaker A:** Shall we start?\n\n**00:04** The plan first.\n\n**00:09 You:** Yes.\n\n**00:12 Speaker A:** Good.\n")
    }

    /// Meeting apps never play your own voice back: the voice on the microphone and not in the desktop audio is yours.
    func testTheVoiceOnlyOnTheMicrophoneIsYou() {
        let rate = 10
        let segments = [TrayTranscript.Segment(start: 0, end: 1, text: "Hi.", speaker: "A"), .init(start: 1, end: 2, text: "Hello.", speaker: "B"), .init(start: 2, end: 3, text: "Bye.", speaker: "A")]
        // A is the call (loud on the desktop, its echo on the mic); B is only on the mic.
        let mic = [Float](repeating: 0.1, count: 10) + [Float](repeating: 0.5, count: 10) + [Float](repeating: 0.1, count: 10)
        let desktop = [Float](repeating: 0.6, count: 10) + [Float](repeating: 0, count: 10) + [Float](repeating: 0.6, count: 10)
        XCTAssertEqual(TrayTranscript.onlyOnMic(segments, mic: mic, desktop: desktop, rate: rate), "B")
        XCTAssertNil(TrayTranscript.onlyOnMic(segments, mic: mic, desktop: [], rate: rate))
        // Two voices only on the mic (people in the room): no telling which is you.
        XCTAssertNil(TrayTranscript.onlyOnMic(segments, mic: mic, desktop: [Float](repeating: 0, count: 30), rate: rate))
    }

    /// The switch waits for all three packs, whole and not mid-transfer; Download fetches what is missing.
    func testTheSwitchWaitsForTheASRTheAlignerAndTheDiarizer() {
        func pack(_ type: String, defect: ModelDefect? = nil, downloading: Bool = false) -> LocalModel {
            var m = LocalModel(id: type, name: type, path: "/tmp/\(type)", sizeFormatted: "1 GB", modelType: type, source: .mlxServe, kind: .base)
            m.defect = defect
            m.isDownloading = downloading
            return m
        }
        let asr = pack("qwen3_asr"), aligner = pack("qwen3_forced_aligner"), diarizer = pack("nemotron_diarization")
        // Without the ASR only it is fetched: the aligner and the diarizer follow it.
        XCTAssertEqual(TrayTranscript.missingRepos([]), [TrayTranscript.asrRepo])
        XCTAssertEqual(TrayTranscript.missingRepos([pack("qwen3_asr", defect: .interruptedDownload), aligner, diarizer]), [TrayTranscript.asrRepo])
        XCTAssertEqual(TrayTranscript.missingRepos([asr]), [DownloadManager.alignerRepo, DownloadManager.diarizerRepo])
        XCTAssertEqual(TrayTranscript.missingRepos([asr, aligner, pack("nemotron_diarization", downloading: true)]), [DownloadManager.diarizerRepo])
        XCTAssertEqual(TrayTranscript.missingRepos([asr, aligner, diarizer]), [])
    }

    /// Audio goes to disk as it arrives: it reads back as written, and a later start leaves silence before it.
    func testARecordingFileReadsBackWhatWasWrittenWithSilenceInItsGaps() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString).appendingPathComponent("a.pcm")
        defer { try? FileManager.default.removeItem(at: url.deletingLastPathComponent()) }
        let file = try RecordingFile(url: url)
        [Float(0.5), -0.25].withUnsafeBufferPointer { file.write($0) }
        [Float](repeating: 0.1, count: 2).withUnsafeBufferPointer { file.write($0, at: 5) }
        file.close()
        let back = RecordingFile.read(url)
        XCTAssertEqual(back.count, 7)
        XCTAssertEqual(back[0], 0.5, accuracy: 1e-4)
        XCTAssertEqual(back[1], -0.25, accuracy: 1e-4)
        XCTAssertEqual(Array(back[2..<5]), [0, 0, 0])
        XCTAssertEqual(back[6], 0.1, accuracy: 1e-4)
    }

    /// Only an audible stretch moves the moment something was last heard.
    func testSilenceIsNotHearingSomething() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".pcm")
        defer { try? FileManager.default.removeItem(at: url) }
        let file = try RecordingFile(url: url)
        let start = file.lastHeard
        [Float](repeating: 0.001, count: 1600).withUnsafeBufferPointer { file.write($0) }
        XCTAssertEqual(file.lastHeard, start)
        [Float](repeating: 0.2, count: 1600).withUnsafeBufferPointer { file.write($0) }
        XCTAssertGreaterThan(file.lastHeard, start)
    }

    func testARecordingStopsOnlyAfterTheWholeSilence() {
        let heard = Date()
        XCTAssertFalse(TrayTranscript.silentTooLong(since: heard, now: heard.addingTimeInterval(599), minutes: 10))
        XCTAssertTrue(TrayTranscript.silentTooLong(since: heard, now: heard.addingTimeInterval(600), minutes: 10))
        XCTAssertFalse(TrayTranscript.silentTooLong(since: heard, now: heard.addingTimeInterval(86_400), minutes: 0), "0 is never")
    }

    func testTheNotesModelFollowsTheSetting() {
        func chat(_ path: String) -> LocalModel {
            LocalModel(id: path, name: path, path: path, sizeFormatted: "1 GB", modelType: "qwen3", source: .mlxServe, kind: .base)
        }
        let models = [chat("/m/chat"), chat("/m/other")]
        XCTAssertEqual(TrayTranscript.notesModel(setting: "", chatModel: "/m/chat", models: models), "/m/chat")
        XCTAssertEqual(TrayTranscript.notesModel(setting: "/m/other", chatModel: "/m/chat", models: models), "/m/other")
        XCTAssertNil(TrayTranscript.notesModel(setting: ServerOptions.notesOff, chatModel: "/m/chat", models: models))
        // A chosen model that is gone falls back to the chat model in use; with none, no notes.
        XCTAssertEqual(TrayTranscript.notesModel(setting: "/m/gone", chatModel: "/m/chat", models: models), "/m/chat")
        XCTAssertNil(TrayTranscript.notesModel(setting: "", chatModel: "", models: models))
    }

    func testTheNotesGoAboveTheTranscript() {
        let segments: [(start: Double, text: String, speaker: String?)] = [(0, "Hi.", "Speaker A"), (3, "Yes.", "You")]
        XCTAssertEqual(TrayTranscript.markdown(title: "2026-10-13 21:59:00", notes: "# Release plan\n\n## Summary\nShip on Thursday.", segments: segments),
                       "# Release plan\n\n2026-10-13 21:59:00\n\n## Summary\nShip on Thursday.\n\n## Transcript\n\n**00:00 Speaker A:** Hi.\n\n**00:03 You:** Yes.\n")
        // Notes without a title of their own keep the recording's start as the title.
        XCTAssertEqual(TrayTranscript.markdown(title: "t", notes: "## Summary\nShort.", segments: segments),
                       "# t\n\n## Summary\nShort.\n\n## Transcript\n\n**00:00 Speaker A:** Hi.\n\n**00:03 You:** Yes.\n")
    }

    func testTheNotesModelReadsTheTranscriptWithTimesAndSpeakers() {
        XCTAssertEqual(TrayTranscript.notesInput([(0, "Hi.", "Speaker A"), (65, "Yes.", "You"), (70, "Alone.", nil)]),
                       "[00:00] Speaker A: Hi.\n[01:05] You: Yes.\n[01:10] Alone.")
    }

    /// A quit or a crash leaves a recording's files behind; they are found oldest first and keep their start time.
    func testRecordingsLeftBehindAreFoundOldestFirst() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        for name in ["2026-10-13 21.59.00.mic.pcm", "2026-10-12 08.00.00.mic.pcm", "2026-10-12 08.00.00.desktop.pcm", "notes.md"] {
            FileManager.default.createFile(atPath: dir.appendingPathComponent(name).path, contents: Data())
        }
        XCTAssertEqual(TrayTranscript.leftovers(in: dir), ["2026-10-12 08.00.00", "2026-10-13 21.59.00"])
        XCTAssertEqual(TrayTranscript.title(of: "2026-10-12 08.00.00"), "2026-10-12 08:00:00")
        XCTAssertEqual(TrayTranscript.leftovers(in: dir.appendingPathComponent("missing")), [])
    }

    func testTheOtherSpeakersAreLetteredInTheOrderTheyFirstSpeak() {
        XCTAssertEqual(TrayTranscript.speakerNames(["B", "A", "B", "C"], you: "A"), ["B": "Speaker A", "A": "You", "C": "Speaker B"])
        XCTAssertEqual(TrayTranscript.speakerNames(["A", "B"], you: nil), ["A": "Speaker A", "B": "Speaker B"])
    }

    /// Live: TRANSCRIPT_SERVER=http://127.0.0.1:<port> TRANSCRIPT_MODEL=<Qwen3-ASR dir> TRANSCRIPT_WAV=<16 kHz mono wav>.
    /// The ASR may land on disk after the server started: the stop registers it first.
    func testARecordingRoundTripsThroughTheServer() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let server = env["TRANSCRIPT_SERVER"], let model = env["TRANSCRIPT_MODEL"], let wav = env["TRANSCRIPT_WAV"] else {
            throw XCTSkip("set TRANSCRIPT_SERVER, TRANSCRIPT_MODEL and TRANSCRIPT_WAV")
        }
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: wav))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let mic = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
        let segments = try await TrayTranscript.transcribeRecording(mic, [], model: model, server: server, speakers: false)
        XCTAssertTrue(segments.map(\.text).joined(separator: " ").lowercased().contains("fox"), "\(segments)")
    }

    /// Live, on a server with the aligner and the diarizer: TRANSCRIPT_CONVERSATION=<16 kHz mono wav of several voices>.
    func testAConversationComesBackWithItsSpeakersNamed() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let server = env["TRANSCRIPT_SERVER"], let model = env["TRANSCRIPT_MODEL"], let wav = env["TRANSCRIPT_CONVERSATION"] else {
            throw XCTSkip("set TRANSCRIPT_SERVER, TRANSCRIPT_MODEL and TRANSCRIPT_CONVERSATION")
        }
        let file = try AVAudioFile(forReading: URL(fileURLWithPath: wav))
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        let mic = Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
        let segments = try await TrayTranscript.transcribeRecording(mic, [], model: model, server: server, speakers: true)
        let names = segments.compactMap(\.speaker)
        XCTAssertEqual(names.first, "Speaker A", "\(segments)")
        XCTAssertGreaterThan(Set(names).count, 1, "\(segments)")
        // Live, with a chat model on that server: TRANSCRIPT_NOTES_MODEL=<its path>.
        if let notesModel = env["TRANSCRIPT_NOTES_MODEL"] {
            let notes = try await TrayTranscript.notes(for: segments, model: notesModel, server: server)
            XCTAssertTrue(notes.hasPrefix("# ") && notes.contains("## "), notes)
        }
    }
}
