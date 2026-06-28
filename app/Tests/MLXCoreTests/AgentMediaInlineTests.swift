import XCTest
import AppKit
@testable import MLXCore

/// The `generate_image` agent tool: dispatch through `AgentEngine.executeToolCall`
/// (injected closure vs. stubs vs. unavailable), the pure caption/base64 split,
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
            generateImage: { prompt in
                XCTAssertEqual(prompt, "a red fox")
                return sentinel
            })
        XCTAssertEqual(r.name, "generate_image")
        XCTAssertEqual(r.output, sentinel)
    }

    @MainActor
    func testGenerateImageWithoutClosureIsUnavailable() async {
        var wd: String? = nil
        let r = await AgentEngine.executeToolCall(
            toolCall("generate_image", ["prompt": "a red fox"]),
            workingDirectory: &wd, repetition: AgentEngine.RepetitionTracker(),
            iteration: 0, agentMemory: AgentMemory())
        XCTAssertTrue(r.output.contains("isn't available in this context"), r.output)
    }

    @MainActor
    func testGenerateAudioAndVideoReturnComingSoon() async {
        var wd: String? = nil
        let audio = await AgentEngine.executeToolCall(
            toolCall("generate_audio", ["text": "hi"]),
            workingDirectory: &wd, repetition: AgentEngine.RepetitionTracker(),
            iteration: 0, agentMemory: AgentMemory())
        XCTAssertEqual(audio.output, AgentPrompt.comingSoonAudio)
        XCTAssertTrue(audio.output.contains("Audio generation window"))

        let video = await AgentEngine.executeToolCall(
            toolCall("generate_video", ["prompt": "clouds"]),
            workingDirectory: &wd, repetition: AgentEngine.RepetitionTracker(),
            iteration: 0, agentMemory: AgentMemory())
        XCTAssertEqual(video.output, AgentPrompt.comingSoonVideo)
        XCTAssertTrue(video.output.contains("Video generation window"))
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
