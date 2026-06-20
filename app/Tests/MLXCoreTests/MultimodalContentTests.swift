import XCTest
@testable import MLXCore

/// Unit tests for the pure multimodal content-block builder that feeds the
/// Gemma 4 12B unified engine (image_url + input_audio + text blocks).
final class MultimodalContentTests: XCTestCase {

    private func blocks(_ any: [[String: Any]]) -> [[String: Any]] { any }

    func testTextOnlyProducesSingleTextBlock() {
        let out = MultimodalContent.build(text: "hello", images: [], audio: [])
        XCTAssertEqual(out.count, 1)
        XCTAssertEqual(out[0]["type"] as? String, "text")
        XCTAssertEqual(out[0]["text"] as? String, "hello")
    }

    func testImageUsesPreprocessedPixelsWhenAvailable() {
        let img = ChatImage(data: Data([0xFF, 0xD8, 0xFF])) // pretend JPEG
        let fakePixels = Data([1, 2, 3, 4])
        let out = MultimodalContent.build(text: "what is this", images: [img], audio: [],
                                          preprocessImage: { _ in fakePixels })
        XCTAssertEqual(out.count, 2) // image + text
        XCTAssertEqual(out[0]["type"] as? String, "image_url")
        let urlDict = out[0]["image_url"] as? [String: Any]
        let url = urlDict?["url"] as? String
        XCTAssertNotNil(url)
        XCTAssertTrue(url!.hasPrefix("data:image/x-mlx-pixels;base64,"))
        XCTAssertTrue(url!.hasSuffix(fakePixels.base64EncodedString()))
        XCTAssertEqual(out[1]["type"] as? String, "text")
    }

    func testImageFallsBackToJpegWhenPreprocessFails() {
        let img = ChatImage(data: Data([0xFF, 0xD8, 0xFF]))
        let out = MultimodalContent.build(text: "", images: [img], audio: [],
                                          preprocessImage: { _ in nil })
        XCTAssertEqual(out.count, 1) // just the image (text empty)
        let url = (out[0]["image_url"] as? [String: Any])?["url"] as? String
        XCTAssertEqual(url, img.base64URL)
        XCTAssertTrue(url!.hasPrefix("data:image/jpeg;base64,"))
    }

    func testServerPreprocessSendsRawJpegEvenWhenPreprocessorAvailable() {
        // Qwen3-VL: the Gemma-square `x-mlx-pixels` format doesn't apply; the app
        // must send the raw image so the server runs Qwen smart_resize + patchify,
        // even though a (Gemma) preprocessor closure would have succeeded.
        let img = ChatImage(data: Data([0xFF, 0xD8, 0xFF]))
        let out = MultimodalContent.build(text: "what is this", images: [img], audio: [],
                                          serverPreprocess: true,
                                          preprocessImage: { _ in Data([1, 2, 3, 4]) })
        XCTAssertEqual(out.count, 2)
        let url = (out[0]["image_url"] as? [String: Any])?["url"] as? String
        XCTAssertEqual(url, img.base64URL)
        XCTAssertTrue(url!.hasPrefix("data:image/jpeg;base64,"))
    }

    func testAudioEmitsInputAudioBlockWithRawPcm() {
        let pcm = Data([0, 0, 0, 0, 0, 0, 0, 0]) // 2 float32 samples
        let clip = ChatAudio(name: "voice.wav", pcm: pcm)
        let out = MultimodalContent.build(text: "transcribe", images: [], audio: [clip])
        XCTAssertEqual(out.count, 2) // audio + text
        XCTAssertEqual(out[0]["type"] as? String, "input_audio")
        let a = out[0]["input_audio"] as? [String: Any]
        XCTAssertEqual(a?["format"] as? String, "mlx_pcm_f32")
        XCTAssertEqual(a?["data"] as? String, pcm.base64EncodedString())
    }

    func testImageThenAudioThenTextOrdering() {
        // The server injects image placeholders before audio placeholders and
        // concatenates [vision ; audio] embeddings in that order; keep the
        // content blocks in the same order for clarity.
        let img = ChatImage(data: Data([1]))
        let clip = ChatAudio(name: "a.wav", pcm: Data([0, 0, 0, 0]))
        let out = MultimodalContent.build(text: "q", images: [img], audio: [clip],
                                          preprocessImage: { _ in Data([9]) })
        XCTAssertEqual(out.map { $0["type"] as? String }, ["image_url", "input_audio", "text"])
    }

    func testChatAudioDurationFromSampleCount() {
        // 16000 float32 samples = 64000 bytes = exactly 1 second at 16 kHz.
        let clip = ChatAudio(name: "x", pcm: Data(count: 16_000 * 4))
        XCTAssertEqual(clip.sampleCount, 16_000)
        XCTAssertEqual(clip.durationSeconds, 1.0, accuracy: 1e-6)
    }
}
