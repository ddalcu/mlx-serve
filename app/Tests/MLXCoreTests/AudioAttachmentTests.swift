import XCTest
@testable import MLXCore

/// An audio clip the user attaches lives beside the pictures in
/// `~/.mlx-serve/attachments/`, as a 16 kHz mono 16-bit WAV, and the history
/// carries its path. The float32 PCM used to ride `chat-history.json` as
/// base64: 64 kB per second of speech, re-read and re-written on every save.
final class AudioAttachmentTests: XCTestCase {

    private func tempRoot() throws -> String {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-core-audio-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }

    /// Float32-LE bytes for a few samples.
    private func pcm(_ samples: [Float]) -> Data {
        var d = Data()
        for s in samples { withUnsafeBytes(of: s.bitPattern.littleEndian) { d.append(contentsOf: $0) } }
        return d
    }

    private func samples(_ pcm: Data) -> [Float] {
        stride(from: 0, to: pcm.count, by: 4).map { i in
            Float(bitPattern: UInt32(littleEndian: pcm.subdata(in: i..<(i + 4)).withUnsafeBytes { $0.load(as: UInt32.self) }))
        }
    }

    // MARK: - The file

    /// What lands on disk is what the model is sent: the round trip through
    /// 16-bit samples is the "reduced" in reduced WAV, and it is taken ONCE, on
    /// send, so a regenerate after a relaunch hands the model the same bytes.
    func testAClipRoundTripsThroughTheFileWithinSixteenBitPrecision() {
        let original: [Float] = [0, 0.5, -0.5, 0.999, -1, 0.123456]
        let wav = AudioClipFile.encode(pcm: pcm(original))
        let back = samples(AudioClipFile.decode(wav: wav) ?? Data())
        XCTAssertEqual(back.count, original.count)
        for (a, b) in zip(original, back) {
            XCTAssertEqual(a, b, accuracy: 1.0 / 32767.0)
        }
    }

    func testTheFileIsACanonicalSixteenKilohertzMonoWav() {
        let wav = AudioClipFile.encode(pcm: pcm([0.25, -0.25]))
        XCTAssertEqual(String(decoding: wav.prefix(4), as: UTF8.self), "RIFF")
        XCTAssertEqual(String(decoding: wav[8..<12], as: UTF8.self), "WAVE")
        XCTAssertEqual(wav.count, 44 + 2 * 2)
        // Channels at 22, sample rate at 24, bits at 34.
        XCTAssertEqual(wav[22], 1)
        XCTAssertEqual(UInt32(wav[24]) | UInt32(wav[25]) << 8 | UInt32(wav[26]) << 16, 16_000)
        XCTAssertEqual(wav[34], 16)
    }

    /// Only our own shape is read back. Anything else is a file that landed in
    /// the folder some other way, and answering samples from it would be a
    /// guess handed to the model.
    func testAFileThatIsNotOurWavDecodesToNothing() {
        XCTAssertNil(AudioClipFile.decode(wav: Data("not a wave".utf8)))
        XCTAssertNil(AudioClipFile.decode(wav: Data()))
        var stereo = AudioClipFile.encode(pcm: pcm([0.1, 0.2]))
        stereo[22] = 2
        XCTAssertNil(AudioClipFile.decode(wav: stereo))
    }

    /// A `Data` slice keeps its parent's indices; the parser must not care.
    func testASliceDecodesLikeTheWholeFile() {
        let wav = AudioClipFile.encode(pcm: pcm([0.25, -0.25]))
        let padded = Data([0xAA, 0xBB]) + wav
        XCTAssertEqual(AudioClipFile.decode(wav: padded[2...]), AudioClipFile.decode(wav: wav))
    }

    /// The one round trip, on send: the file holds what the model is sent,
    /// and the clip's own samples are the file's from then on.
    func testAStoredClipCarriesTheFilesSamplesAndItsPath() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }
        let clip = ChatAudio(name: "note.m4a", pcm: pcm([0.3, -0.7, 0.123456]))

        let stored = AudioClipFile.stored(clip, root: root)

        let path = try XCTUnwrap(stored.path)
        XCTAssertEqual(path, root + "/\(clip.id.uuidString)_note.wav")
        let onDisk = try XCTUnwrap(FileManager.default.contents(atPath: path))
        XCTAssertEqual(stored.pcm, AudioClipFile.decode(wav: onDisk))
        XCTAssertNotEqual(stored.pcm, clip.pcm, "the 16-bit rounding is taken here, once")
        XCTAssertEqual(stored.id, clip.id)
    }

    // MARK: - The history

    func testEncodingCarriesThePathAndNotTheSamples() throws {
        var clip = ChatAudio(name: "note.m4a", pcm: pcm([0.1, 0.2, 0.3]))
        clip.path = "/tmp/x.wav"
        let json = try JSONSerialization.jsonObject(with: try JSONEncoder().encode(clip)) as? [String: Any]
        XCTAssertEqual(json?["path"] as? String, "/tmp/x.wav")
        XCTAssertEqual(json?["name"] as? String, "note.m4a")
        XCTAssertNil(json?["pcm"], "samples belong in the file, not the history")
    }

    func testDecodingReadsTheSamplesBackFromTheFile() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }
        let path = root + "/clip.wav"
        try AudioClipFile.encode(pcm: pcm([0.5, -0.5])).write(to: URL(fileURLWithPath: path))

        let json = """
        {"id":"\(UUID().uuidString)","name":"clip.m4a","path":"\(path)"}
        """.data(using: .utf8)!
        let clip = try JSONDecoder().decode(ChatAudio.self, from: json)
        XCTAssertEqual(clip.path, path)
        XCTAssertEqual(clip.sampleCount, 2)
        XCTAssertEqual(samples(clip.pcm)[0], 0.5, accuracy: 1.0 / 32767.0)
    }

    /// A history from before this change still carries `pcm`; one written
    /// after it names a file that may since be gone. Both DECODE: a throw here
    /// is `loadChatHistory`'s `?? []`, and that empties every conversation.
    func testAHistoryWrittenBeforeAndAMissingFileBothStillDecode() throws {
        let legacy = """
        {"id":"\(UUID().uuidString)","name":"old.m4a","pcm":"AAAAAA=="}
        """.data(using: .utf8)!
        let old = try JSONDecoder().decode(ChatAudio.self, from: legacy)
        XCTAssertNil(old.path)
        XCTAssertEqual(old.sampleCount, 0, "the clip is gone, and the transcript says so")

        let missing = """
        {"id":"\(UUID().uuidString)","name":"gone.m4a","path":"/nonexistent/gone.wav"}
        """.data(using: .utf8)!
        let clip = try JSONDecoder().decode(ChatAudio.self, from: missing)
        XCTAssertEqual(clip.path, "/nonexistent/gone.wav")
        XCTAssertEqual(clip.sampleCount, 0)
    }

    // MARK: - The wire

    /// A sample-less clip must not become an `input_audio` block with an empty
    /// payload: that tells the model there is a recording and hands it nothing.
    func testAClipWithNoSamplesIsNotSent() {
        var gone = ChatAudio(name: "gone.m4a", pcm: Data())
        gone.path = "/gone.wav"
        let blocks = MultimodalContent.build(
            text: "what did I say",
            images: [],
            audio: [gone, ChatAudio(name: "here.m4a", pcm: pcm([0.1]))])
        XCTAssertEqual(blocks.filter { $0["type"] as? String == "input_audio" }.count, 1)
    }

    // MARK: - Deleting

    /// Same sweep as the pictures: a clip's file goes with its message or its
    /// chat, and never while a fork still names it.
    func testDeletingAMessageRemovesItsOwnClip() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        var doomed = ChatMessage(role: .user, content: "listen")
        var clip = ChatAudio(name: "a.m4a", pcm: Data())
        clip.path = root + "/a.wav"
        doomed.audio = [clip]

        var fork = ChatSession(title: "fork")
        fork.messages = [doomed]

        XCTAssertEqual(AttachmentStore.removablePaths(orphanedBy: [doomed], in: [], root: root),
                       [root + "/a.wav"])
        XCTAssertEqual(AttachmentStore.removablePaths(orphanedBy: [doomed], in: [fork], root: root),
                       [], "the fork still plays it")
    }

    func testDeletingAConversationRemovesItsClips() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(atPath: root) }

        var m = ChatMessage(role: .user, content: "listen")
        var clip = ChatAudio(name: "a.m4a", pcm: Data())
        clip.path = root + "/a.wav"
        m.audio = [clip]
        var doomed = ChatSession(title: "t")
        doomed.messages = [m]

        XCTAssertEqual(AttachmentStore.removablePaths(deleting: [doomed.id], in: [doomed], root: root),
                       [root + "/a.wav"])
    }

    // MARK: - Naming

    /// `<uuid>_<name>.wav`, the picture's own scheme: the extension describes
    /// the bytes stored, never the name the clip arrived under.
    func testTheFileIsNamedForTheRecordAndTheStoredFormat() {
        let id = UUID()
        let name = AttachmentStore.filename(id: id, name: "Voice memo.m4a", ext: "wav")
        XCTAssertEqual(name, "\(id.uuidString)_Voice memo.wav")
    }
}
