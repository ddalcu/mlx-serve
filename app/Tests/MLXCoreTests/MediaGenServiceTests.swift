import XCTest
@testable import MLXCore

/// Tests for the unified media-generation path: image/audio/video now run
/// through the ONE main `mlx-serve` server (registry-hosted) instead of a
/// dedicated `NativeGenServer` subprocess. Covers the pure response-decode
/// contracts + the load→generate→unload residency default.
@MainActor
final class MediaGenServiceTests: XCTestCase {

    // MARK: - Image response decode (the /v1/images/generations contract)

    func testDecodePngB64ExtractsImage() throws {
        let pngBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x01, 0x02])
        let b64 = pngBytes.base64EncodedString()
        let body = try JSONSerialization.data(withJSONObject: ["data": [["b64_json": b64]]])
        let decoded = ImageGenService.decodePngB64(body)
        XCTAssertEqual(decoded, pngBytes)
    }

    func testDecodePngB64RejectsMalformed() {
        XCTAssertNil(ImageGenService.decodePngB64(Data("not json".utf8)))
        let noData = try! JSONSerialization.data(withJSONObject: ["error": "boom"])
        XCTAssertNil(ImageGenService.decodePngB64(noData))
        let emptyArr = try! JSONSerialization.data(withJSONObject: ["data": []])
        XCTAssertNil(ImageGenService.decodePngB64(emptyArr))
    }

    // MARK: - Video response decode (the /v1/video/generations contract)

    func testDecodeFramesParsesRgb8Body() {
        let frames = 2, w = 2, h = 2
        let rgb = Data(repeating: 7, count: frames * w * h * 3)
        let obj: [String: Any] = [
            "format": "rgb8", "frames": frames, "height": h, "width": w,
            "fps": 24, "data": rgb.base64EncodedString(),
        ]
        let decoded = VideoGenService.decodeFrames(obj)
        XCTAssertEqual(decoded?.frames, frames)
        XCTAssertEqual(decoded?.rgb.count, frames * w * h * 3)
    }

    func testDecodeFramesRejectsSizeMismatch() {
        // rgb byte count must equal frames*h*w*3, else the body is corrupt.
        let obj: [String: Any] = [
            "format": "rgb8", "frames": 2, "height": 2, "width": 2,
            "data": Data(repeating: 1, count: 8).base64EncodedString(),  // wrong size
        ]
        XCTAssertNil(VideoGenService.decodeFrames(obj))
    }

    // MARK: - Model resolution (moved from NativeGenServer to ServerManager)

    func testResolveModelDirMissingRepoIsNil() {
        XCTAssertNil(ServerManager.resolveModelDir(repo: "nonexistent-owner/definitely-not-a-real-model-xyz"))
    }

    // MARK: - Residency default

    func testKeepResidentDefaultsOff() {
        // Decision: load→generate→unload by default; "Keep loaded" is opt-in.
        let img = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 4, guidance: 0.5)
        XCTAssertFalse(img.keepResident)
        let vid = VideoGenRequest(model: .ltx23Q4, prompt: "x", width: 384, height: 256, numFrames: 9, fps: 24, mode: .oneStage, steps: 6, cfgScale: 1.0)
        XCTAssertFalse(vid.keepResident)
        let aud = AudioGenRequest(model: .qwen3TTS06B, text: "x")
        XCTAssertFalse(aud.keepResident)
    }
}
