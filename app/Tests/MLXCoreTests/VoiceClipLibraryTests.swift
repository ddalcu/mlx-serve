import XCTest
@testable import MLXCore

/// Uploading a voice used to mean OVERWRITING the one global clip
/// (`voice-clips/voice-clone.wav`), so a second upload silently cost you the
/// first. Agents each pick their own voice, so clips are now a small library
/// under the same folder: named files you can point several agents at, and come
/// back to later.
final class VoiceClipLibraryTests: XCTestCase {

    private func tempDir() throws -> String {
        let d = (NSTemporaryDirectory() as NSString).appendingPathComponent("vcl-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: d, withIntermediateDirectories: true)
        return d
    }

    private func touch(_ dir: String, _ name: String) throws {
        try Data("RIFF".utf8).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent(name)))
    }

    // MARK: - Naming

    func testBasenameKeepsTheSourceNameButMakesItAFilename() {
        XCTAssertEqual(VoiceClipLibrary.basename(for: "Morgan Freeman.mp3"), "Morgan Freeman")
        XCTAssertEqual(VoiceClipLibrary.basename(for: "my/voice:take 2.wav"), "my-voice-take 2")
        XCTAssertEqual(VoiceClipLibrary.basename(for: "   .wav"), "Voice clip",
                       "a nameless file still gets a usable label")
    }

    func testBasenameNeverCollidesWithTheGlobalSettingsClip() {
        // The global clip lives in the same folder; a library entry that took its
        // filename would have the Settings picker and an agent fighting over one
        // file, and re-recording in Settings would overwrite the agent's voice.
        XCTAssertNotEqual(VoiceClipLibrary.basename(for: "voice-clone.wav"), "voice-clone")
    }

    func testUniqueBasenameDeduplicates() {
        XCTAssertEqual(VoiceClipLibrary.uniqueBasename("morgan", existing: []), "morgan")
        XCTAssertEqual(VoiceClipLibrary.uniqueBasename("morgan", existing: ["morgan"]), "morgan-2")
        XCTAssertEqual(VoiceClipLibrary.uniqueBasename("morgan", existing: ["morgan", "morgan-2"]), "morgan-3")
        XCTAssertEqual(VoiceClipLibrary.uniqueBasename("morgan", existing: ["other"]), "morgan")
    }

    // MARK: - Listing

    func testListsWavClipsAndExcludesTheGlobalOne() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        try touch(dir, "morgan.wav")
        try touch(dir, "alice.wav")
        try touch(dir, "voice-clone.wav")          // the Settings clip
        try touch(dir, "notes.txt")

        let clips = VoiceClipLibrary.clips(in: dir)
        XCTAssertEqual(clips.map(\.name), ["alice", "morgan"], "sorted, and non-audio ignored")
        XCTAssertFalse(clips.contains { $0.path.hasSuffix("voice-clone.wav") },
                       "the global clip is shown as its own entry, not duplicated in the library")
        XCTAssertTrue(clips.allSatisfy { $0.path.hasPrefix(dir) })
    }

    func testAMissingFolderIsAnEmptyLibraryRatherThanAnError() {
        XCTAssertEqual(VoiceClipLibrary.clips(in: "/nope/not/here").count, 0)
    }

    // MARK: - Install

    func testInstallStoresTheNormalizedClipUnderItsOwnName() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        // Stand in for AudioReference's converter: the real one writes a 24 kHz
        // mono wav to a temp file, which is all install() cares about.
        let normalized = (dir as NSString).appendingPathComponent("tmp-normalized.wav")
        try Data("RIFF-normalized".utf8).write(to: URL(fileURLWithPath: normalized))

        let clip = try VoiceClipLibrary.install(URL(fileURLWithPath: "/somewhere/Morgan Freeman.mp3"),
                                               in: dir,
                                               normalize: { _ in URL(fileURLWithPath: normalized) })
        XCTAssertEqual(clip.name, "Morgan Freeman")
        XCTAssertEqual(clip.path, (dir as NSString).appendingPathComponent("Morgan Freeman.wav"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: clip.path))
        XCTAssertEqual(try String(contentsOfFile: clip.path, encoding: .utf8), "RIFF-normalized")
    }

    func testASecondUploadOfTheSameNameKeepsBothClips() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let a = (dir as NSString).appendingPathComponent("tmp-a.wav")
        let b = (dir as NSString).appendingPathComponent("tmp-b.wav")
        try Data("first".utf8).write(to: URL(fileURLWithPath: a))
        try Data("second".utf8).write(to: URL(fileURLWithPath: b))

        let first = try VoiceClipLibrary.install(URL(fileURLWithPath: "/x/morgan.wav"), in: dir,
                                                 normalize: { _ in URL(fileURLWithPath: a) })
        let second = try VoiceClipLibrary.install(URL(fileURLWithPath: "/x/morgan.wav"), in: dir,
                                                  normalize: { _ in URL(fileURLWithPath: b) })
        XCTAssertNotEqual(first.path, second.path, "the earlier clip is never overwritten")
        XCTAssertEqual(second.name, "morgan-2")
        XCTAssertEqual(try String(contentsOfFile: first.path, encoding: .utf8), "first")
        XCTAssertTrue(VoiceClipLibrary.clips(in: dir).map(\.name).contains("morgan"))
        XCTAssertTrue(VoiceClipLibrary.clips(in: dir).map(\.name).contains("morgan-2"))
    }

    func testInstallCreatesTheFolderOnFirstUse() throws {
        let root = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: root) }
        let dir = (root as NSString).appendingPathComponent("voice-clips")
        let src = (root as NSString).appendingPathComponent("tmp.wav")
        try Data("x".utf8).write(to: URL(fileURLWithPath: src))

        let clip = try VoiceClipLibrary.install(URL(fileURLWithPath: "/x/a.wav"), in: dir,
                                                normalize: { _ in URL(fileURLWithPath: src) })
        XCTAssertTrue(FileManager.default.fileExists(atPath: clip.path))
    }

    func testInstallPropagatesANormalizationFailure() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        struct Boom: Error {}
        XCTAssertThrowsError(try VoiceClipLibrary.install(URL(fileURLWithPath: "/x/a.aiff"), in: dir,
                                                          normalize: { _ in throw Boom() }),
                             "an unreadable clip must surface, not save an empty file")
        XCTAssertTrue(VoiceClipLibrary.clips(in: dir).isEmpty)
    }

    // MARK: - Picking a stored clip for an agent

    func testAStoredClipBecomesTheAgentsVoice() {
        let clip = VoiceClipLibrary.Clip(name: "morgan", path: "/clips/morgan.wav")
        var agent = Agent(name: "A", brief: "", systemPrompt: "p")
        agent.voice = .clone(clip.path)
        XCTAssertEqual(agent.resolvedVoice, .clone("/clips/morgan.wav"))
        XCTAssertEqual(ActiveAgentVoice.neuralVoice(agent: agent.resolvedVoice, options: ServerOptions()),
                       .clone(clipPath: "/clips/morgan.wav"),
                       "an uploaded clip speaks through the same clone path as the Settings one")
    }
}
