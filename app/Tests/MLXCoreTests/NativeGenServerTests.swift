import XCTest
@testable import MLXCore

/// Tests for the native-server generation path that replaced the Python venv
/// for audio + image (`NativeGenServer` + the gen services' pure helpers).
@MainActor
final class NativeGenServerTests: XCTestCase {

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

    // MARK: - Gen-server helpers

    @MainActor
    func testFreePortIsInValidRange() {
        let port = NativeGenServer.freePort()
        XCTAssertGreaterThan(port, 1024)
        XCTAssertLessThanOrEqual(port, 65535)
    }

    @MainActor
    func testResolveModelDirMissingRepoIsNil() {
        // A repo that exists in neither ~/.mlx-serve/models nor the HF cache.
        XCTAssertNil(NativeGenServer.resolveModelDir(repo: "nonexistent-owner/definitely-not-a-real-model-xyz"))
    }

    @MainActor
    func testBinaryPathResolves() {
        // Either a real path (bundled / zig-out) or the bare fallback name.
        let p = NativeGenServer.resolveBinaryPath()
        XCTAssertFalse(p.isEmpty)
        XCTAssertTrue(p.hasSuffix("mlx-serve"))
    }
}
