import XCTest
import AppKit
@testable import MLXCore

/// The four `generate_*` agent tools: dispatch through
/// `AgentEngine.executeToolCall` (injected closure vs. unavailable), the pure
/// caption/base64 split for images, the caption/path split for tracks and clips,
/// and PNG→JPEG transcoding for inline display.
final class AgentMediaInlineTests: XCTestCase {

    private func toolCall(_ name: String, _ args: [String: String]) -> APIClient.ToolCall {
        APIClient.ToolCall(id: "1", name: name, arguments: args, rawArguments: "")
    }

    // MARK: - Dispatch

    @MainActor
    func testGenerateImageCallsInjectedClosure() async {
        var wd: String? = nil
        let sentinel = "SENTINEL:caption\ndata:image/jpeg;base64,Zm9v"
        let r = await AgentEngine.executeToolCall(
            toolCall("generate_image", ["prompt": "a red fox"]),
            workingDirectory: &wd, repetition: AgentEngine.RepetitionTracker(),
            iteration: 0, agentMemory: AgentMemory(),
            generateMedia: { kind, args in
                XCTAssertEqual(kind, .image)
                XCTAssertEqual(args["prompt"], "a red fox")
                return sentinel
            })
        XCTAssertEqual(r.name, "generate_image")
        XCTAssertEqual(r.output, sentinel)
    }

    @MainActor
    func testEveryMediaToolRoutesToItsOwnKindWithArgumentsIntact() async {
        // One seam for four tools, so the tool→kind mapping is the whole
        // dispatch: get it wrong and a music request renders a video.
        let calls: [(String, [String: String], MediaKind)] = [
            ("generate_image",  ["prompt": "a fox"], .image),
            ("generate_speech", ["text": "hello", "speed": "1.5"], .speech),
            ("generate_music",  ["prompt": "lo-fi", "duration_seconds": "30"], .music),
            ("generate_video",  ["prompt": "clouds", "seconds": "2"], .video),
        ]
        for (name, args, expected) in calls {
            var wd: String? = nil
            var seen: (MediaKind, [String: String])? = nil
            let r = await AgentEngine.executeToolCall(
                toolCall(name, args),
                workingDirectory: &wd, repetition: AgentEngine.RepetitionTracker(),
                iteration: 0, agentMemory: AgentMemory(),
                generateMedia: { kind, a in seen = (kind, a); return "ok" })
            XCTAssertEqual(r.name, name)
            XCTAssertEqual(seen?.0, expected, "\(name) routed to the wrong kind")
            XCTAssertEqual(seen?.1, args, "\(name) must pass its arguments through untouched")
        }
    }

    @MainActor
    func testMediaToolsWithoutClosureAreUnavailable() async {
        for name in ["generate_image", "generate_speech", "generate_music", "generate_video"] {
            var wd: String? = nil
            let r = await AgentEngine.executeToolCall(
                toolCall(name, ["prompt": "x", "text": "x"]),
                workingDirectory: &wd, repetition: AgentEngine.RepetitionTracker(),
                iteration: 0, agentMemory: AgentMemory())
            XCTAssertTrue(r.output.contains("isn't available in this context"), "\(name): \(r.output)")
        }
    }

    @MainActor
    func testMediaToolsAreGatedByTheAgentsCapabilities() async {
        // They're dispatched ahead of the handler registry, so the capability
        // gate has to sit ahead of THEM.
        var wd: String? = nil
        var fired = false
        let r = await AgentEngine.executeToolCall(
            toolCall("generate_music", ["prompt": "lo-fi"]),
            workingDirectory: &wd, repetition: AgentEngine.RepetitionTracker(),
            iteration: 0, agentMemory: AgentMemory(),
            generateMedia: { _, _ in fired = true; return "made it" },
            allowedTools: [.readFile])
        XCTAssertFalse(fired, "a gated media tool must be refused BEFORE it loads a model")
        XCTAssertTrue(r.output.contains("generate_music"), r.output)
    }

    // MARK: - splitMediaRef (tracks and clips ride a PATH, not bytes)

    func testSplitMediaRefRecoversCaptionAndReference() {
        let path = "/Users/x/.mlx-serve/generations/music/2026-07-28/track.wav"
        let output = "Generated a 30s track for: lo-fi. Saved to \(path).\n"
            + AgentMediaInline.mediaRefLine(kind: .audio, path: path)
        let (caption, ref) = AgentMediaInline.splitMediaRef(output, prompt: "lo-fi")
        XCTAssertEqual(caption, "Generated a 30s track for: lo-fi. Saved to \(path).")
        XCTAssertEqual(ref, ChatMediaRef(kind: .audio, path: path, prompt: "lo-fi"))
        XCTAssertFalse(caption.contains(AgentMediaInline.mediaRefMarker),
                       "the marker must not survive into the model-facing caption")
    }

    /// A generated image carries BOTH markers — the JPEG bytes the transcript
    /// displays and the PNG path its Reveal-in-Finder button opens. Ordering is
    /// load-bearing: the ref line sits between the caption and the data URI, so
    /// `splitMediaRef` gets a clean caption and a ref that stops at its own
    /// newline, while `splitInlineImage` still finds the payload. Put the ref
    /// AFTER the base64 and the caption swallows the whole data URI.
    func testAGeneratedImageCarriesBothItsBytesAndItsPath() {
        let path = "/Users/x/.mlx-serve/generations/images/2026-07-28/fox.png"
        let payload = Data("pretend-jpeg".utf8)
        let output = "Generated a 1024×1024 image for: a fox. Saved to \(path).\n"
            + AgentMediaInline.mediaRefLine(kind: .image, path: path) + "\n"
            + AgentMediaInline.jpegDataURIMarker + payload.base64EncodedString()

        let (caption, ref) = AgentMediaInline.splitMediaRef(output, prompt: "a fox")
        XCTAssertEqual(caption, "Generated a 1024×1024 image for: a fox. Saved to \(path).")
        XCTAssertEqual(ref, ChatMediaRef(kind: .image, path: path, prompt: "a fox"))
        XCTAssertFalse(caption.contains("base64"), "the model must never see the payload")

        let (_, jpeg) = AgentMediaInline.splitInlineImage(output)
        XCTAssertEqual(jpeg, payload, "the ref line must not break the payload split")
    }

    func testSplitMediaRefKeepsAPathContainingColons() {
        // The split is ONE colon: a path is user data and may hold more.
        let path = "/tmp/odd:name/clip:2.mp4"
        let (_, ref) = AgentMediaInline.splitMediaRef(
            "caption\n" + AgentMediaInline.mediaRefLine(kind: .video, path: path), prompt: "p")
        XCTAssertEqual(ref?.path, path)
        XCTAssertEqual(ref?.kind, .video)
    }

    func testSplitMediaRefWithNoMarkerOrABadKindYieldsNoReference() {
        let (caption, ref) = AgentMediaInline.splitMediaRef("just a sentence", prompt: "p")
        XCTAssertEqual(caption, "just a sentence")
        XCTAssertNil(ref)
        // A kind this build doesn't know leaves the caption intact rather than
        // attaching a player for something we can't render.
        let (c2, r2) = AgentMediaInline.splitMediaRef("cap\nmlx-serve-media:hologram:/x/y", prompt: "p")
        XCTAssertEqual(c2, "cap")
        XCTAssertNil(r2)
        // So does an empty path.
        XCTAssertNil(AgentMediaInline.splitMediaRef("cap\nmlx-serve-media:audio:", prompt: "p").ref)
    }

    func testMediaRefRoundTripsThroughTheTranscript() throws {
        var msg = ChatMessage(role: .assistant, content: "")
        msg.media = [ChatMediaRef(kind: .video, path: "/x/y.mp4", prompt: "clouds")]
        let back = try JSONDecoder().decode(ChatMessage.self, from: JSONEncoder().encode(msg))
        XCTAssertEqual(back.media, msg.media)
    }

    func testMessagesSavedBeforeMediaExistedDecodeWithNone() throws {
        let json = #"""
        {"id":"\#(UUID().uuidString)","role":"assistant","content":"hi",
         "isStreaming":false,"timestamp":0}
        """#
        let back = try JSONDecoder().decode(ChatMessage.self, from: Data(json.utf8))
        XCTAssertNil(back.media)
    }

    // MARK: - splitInlineImage

    func testSplitInlineImageRecoversCaptionAndJpeg() {
        let payload = Data("not-really-a-jpeg-but-base64-round-trips".utf8)
        let b64 = payload.base64EncodedString()
        let output = "Generated a 1024×1024 image for: red fox. Saved to /x/y.png.\ndata:image/jpeg;base64,\(b64)"
        let (caption, jpeg) = AgentMediaInline.splitInlineImage(output)
        XCTAssertEqual(caption, "Generated a 1024×1024 image for: red fox. Saved to /x/y.png.")
        XCTAssertEqual(jpeg, payload)
        // The base64 must NOT survive into the model-facing caption.
        XCTAssertFalse(caption.contains("base64"))
        XCTAssertFalse(caption.contains(b64))
    }

    func testSplitInlineImageNoMarkerReturnsWholeStringNoImage() {
        let (caption, jpeg) = AgentMediaInline.splitInlineImage("just text, no image here")
        XCTAssertEqual(caption, "just text, no image here")
        XCTAssertNil(jpeg)
    }

    // MARK: - pngFileToJpegDataURI

    func testPngFileToJpegDataURITranscodes() throws {
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: 8, pixelsHigh: 8,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        let png = rep.representation(using: .png, properties: [:])!
        let path = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("agent-media-\(UUID().uuidString).png")
        try png.write(to: URL(fileURLWithPath: path))
        defer { try? FileManager.default.removeItem(atPath: path) }

        let uri = AgentMediaInline.pngFileToJpegDataURI(path)
        XCTAssertNotNil(uri)
        XCTAssertTrue(uri!.hasPrefix("data:image/jpeg;base64,"))
        // Re-split through the same helper → decodes back to a valid bitmap.
        let (_, jpeg) = AgentMediaInline.splitInlineImage(uri!)
        XCTAssertNotNil(jpeg)
        XCTAssertNotNil(NSBitmapImageRep(data: jpeg!), "transcoded payload must be a valid image")
    }

    func testPngFileToJpegDataURIMissingFileReturnsNil() {
        XCTAssertNil(AgentMediaInline.pngFileToJpegDataURI("/nonexistent/\(UUID().uuidString).png"))
    }
}
