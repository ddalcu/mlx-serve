import XCTest
@testable import MLXCore

/// The single-image-to-3D path (Hunyuan3D 2.1 shape stage) runs through the ONE
/// main `mlx-serve` server, same as image/audio/video. Pins the pure wire
/// contracts: the `/v1/3d/generations` request body, the `{format:"glb",data}`
/// response decode, and the `.glb` output-path shape under `models3dRoot`.
@MainActor
final class Model3DGenServiceTests: XCTestCase {

    // MARK: - Request body (the /v1/3d/generations REQUEST contract)

    func testRequestJsonCarriesImageAndDefaults() {
        let req = Model3DGenRequest(model: .hunyuan3d21_8bit, photoPath: "/tmp/chair.png")
        // The photo is base64-encoded (cutout-on-white) OFF the main actor and
        // handed to requestJson — the wire contract is the encoded string, so
        // the test passes one directly (no file I/O in the pure function).
        let imageB64 = Data([0x89, 0x50, 0x4E, 0x47]).base64EncodedString()
        let json = Model3DGenService.requestJson(for: req, modelName: "m", imageB64: imageB64, seed: 1234)
        XCTAssertEqual(json["model"] as? String, "m")
        XCTAssertEqual(json["image"] as? String, imageB64)
        // Request struct defaults flow through unchanged.
        XCTAssertEqual(json["steps"] as? Int, 30)
        XCTAssertEqual(json["guidance_scale"] as? Double, 5.0)
        XCTAssertEqual(json["octree_resolution"] as? Int, 384)
        XCTAssertEqual(json["seed"] as? Int, 1234)
        // `stream` is injected by APIClient.streamGeneration, NOT requestJson —
        // identical to the image/video/audio panes. Pinning its absence keeps a
        // future edit from redundantly (or wrongly) hardcoding it here.
        XCTAssertNil(json["stream"])
    }

    func testRequestJsonReflectsCustomSettings() {
        var req = Model3DGenRequest(model: .hunyuan3d21_8bit, photoPath: "/tmp/x.jpg")
        req.steps = 42
        req.guidanceScale = 7.5
        req.octreeResolution = 380
        let json = Model3DGenService.requestJson(for: req, modelName: "hy3d", imageB64: "QUJD", seed: 9)
        XCTAssertEqual(json["steps"] as? Int, 42)
        XCTAssertEqual(json["guidance_scale"] as? Double, 7.5)
        XCTAssertEqual(json["octree_resolution"] as? Int, 380)
        XCTAssertEqual(json["seed"] as? Int, 9)
    }

    // MARK: - Response decode (the /v1/3d/generations RESPONSE contract)

    func testDecodeGlbExtractsBytes() throws {
        // glTF binary magic ("glTF") + a little payload — the shape of the bytes
        // the Zig writer emits; the decoder only cares about base64 round-trip.
        let glbBytes = Data([0x67, 0x6C, 0x54, 0x46, 0x02, 0x00, 0x00, 0x00, 0x01, 0x02])
        let obj: [String: Any] = ["format": "glb", "data": glbBytes.base64EncodedString()]
        XCTAssertEqual(Model3DGenService.decodeGlb(obj), glbBytes)
        // Also from a raw Data body (the non-stream response shape).
        let body = try JSONSerialization.data(withJSONObject: ["created": 0, "format": "glb",
                                                               "data": glbBytes.base64EncodedString()])
        XCTAssertEqual(Model3DGenService.decodeGlb(body), glbBytes)
    }

    func testDecodeGlbRejectsMalformed() {
        XCTAssertNil(Model3DGenService.decodeGlb(Data("not json".utf8)))
        // Wrong format tag → nil (never mistake a PNG/error body for a mesh).
        XCTAssertNil(Model3DGenService.decodeGlb(["format": "png", "data": "QUJD"]))
        // Missing data → nil.
        XCTAssertNil(Model3DGenService.decodeGlb(["format": "glb"]))
        // Non-base64 data → nil.
        XCTAssertNil(Model3DGenService.decodeGlb(["format": "glb", "data": "not base64!!!"]))
    }

    // MARK: - Output path shape

    func testMakeOutputPathIsDatedGlbUnderModels3dRoot() {
        let path = Model3DGenService.makeOutputPath(photoPath: "/Users/me/Pictures/My Chair.png")
        XCTAssertTrue(path.hasPrefix(MediaStorage.models3dRoot), "must live under models3dRoot")
        XCTAssertTrue(path.hasSuffix(".glb"), "3D output is a .glb")
        // Slug is derived from the photo's name (non-alphanumerics collapsed).
        XCTAssertTrue((path as NSString).lastPathComponent.contains("my-chair"),
                      "filename should slug the source photo name: \(path)")
    }

    func testMakeOutputPathFallsBackForEmptySlug() {
        // A photo whose name has no alphanumerics still yields a valid filename.
        let path = Model3DGenService.makeOutputPath(photoPath: "/tmp/___.png")
        XCTAssertTrue(path.hasSuffix(".glb"))
        XCTAssertTrue((path as NSString).lastPathComponent.contains("model"))
    }

    // MARK: - Texture (PBR) stage

    func testTextureDefaultsOffAndRidesTheRequestBody() {
        var req = Model3DGenRequest(model: .hunyuan3d21_8bit, photoPath: "/tmp/x.png")
        XCTAssertFalse(req.texture, "texture stage is opt-in")
        let off = Model3DGenService.requestJson(for: req, modelName: "m", imageB64: "QUJD", seed: 1)
        XCTAssertEqual(off["texture"] as? Bool, false)
        req.texture = true
        let on = Model3DGenService.requestJson(for: req, modelName: "m", imageB64: "QUJD", seed: 1)
        XCTAssertEqual(on["texture"] as? Bool, true)
    }

    func testModel3DSettingsPersistTextureToggle() throws {
        var s = Model3DGenSettings()
        XCTAssertFalse(s.texture, "texture default off until the paint port is validated")
        s.texture = true
        let decoded = try JSONDecoder().decode(Model3DGenSettings.self, from: try JSONEncoder().encode(s))
        XCTAssertTrue(decoded.texture)
        // Legacy payloads without the key decode to off.
        let legacy = try JSONDecoder().decode(Model3DGenSettings.self, from: Data("{}".utf8))
        XCTAssertFalse(legacy.texture)
    }

    // MARK: - History shelf (thumbnails + click-to-preview)

    func testThumbnailPathSitsBesideTheGlb() {
        // foo.glb → foo.glb.thumb.png — same dir, non-.glb suffix so the
        // history scan (which filters on ".glb") never lists thumbnails.
        let p = Model3DGenService.thumbnailPath(for: "/a/b/2026-07-04_x.glb")
        XCTAssertEqual(p, "/a/b/2026-07-04_x.glb.thumb.png")
        XCTAssertFalse(p.hasSuffix(".glb"))
    }

    @MainActor
    func testShowHistoryItemSetsCompletedPhaseForExistingFile() throws {
        let dir = NSTemporaryDirectory() + "hy3d-hist-\(UUID().uuidString)"
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let glb = dir + "/one.glb"
        FileManager.default.createFile(atPath: glb, contents: Data("glTF".utf8))

        let svc = Model3DGenService()
        svc.showHistoryItem(glb)
        XCTAssertEqual(svc.phase, .completed(path: glb))
    }

    @MainActor
    func testShowHistoryItemPrunesMissingFileFromRecents() {
        let svc = Model3DGenService()
        let missing = "/nonexistent/\(UUID().uuidString).glb"
        svc.showHistoryItem(missing) // must not crash, must not set .completed
        if case .completed(let p) = svc.phase { XCTAssertNotEqual(p, missing) }
        XCTAssertFalse(svc.recent.contains(missing))
    }

    // MARK: - Residency default

    func testKeepResidentDefaultsOff() {
        // Load → generate → unload by default; "Keep loaded" is opt-in, like the
        // sibling media panes.
        let req = Model3DGenRequest(model: .hunyuan3d21_8bit, photoPath: "/tmp/x.png")
        XCTAssertFalse(req.keepResident)
    }
}
