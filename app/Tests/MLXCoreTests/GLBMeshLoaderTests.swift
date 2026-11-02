import XCTest
import SceneKit
@testable import MLXCore

/// Round-trips the GLB writer's output through the app's SceneKit loader. The
/// fixture (`Fixtures/hy3d_sphere.glb`) is produced by the Zig GLB writer and
/// may not exist yet while the Swift side is built — the test XCTSkips when it's
/// absent so the suite stays green either way.
final class GLBMeshLoaderTests: XCTestCase {

    /// The fixture is NOT a declared SwiftPM resource (the test target ships no
    /// resource bundle), so it's located by a source-relative path off
    /// `#filePath` rather than `Bundle.module`. This finds the real file the
    /// moment the Zig writer drops it in, with no Package.swift change.
    private var fixtureURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/hy3d_sphere.glb")
    }

    func testLoadsFixtureGlbIntoSceneWithGeometry() throws {
        guard FileManager.default.fileExists(atPath: fixtureURL.path) else {
            throw XCTSkip("hy3d_sphere.glb fixture not present yet (produced by the Zig writer)")
        }
        let scene = try XCTUnwrap(GLBMeshLoader.loadScene(url: fixtureURL),
                                  "GLB fixture failed to load into an SCNScene")
        let node = try XCTUnwrap(GLBMeshLoader.firstGeometryNode(in: scene),
                                 "loaded scene has no geometry-bearing node")
        // A real mesh carries a vertex source.
        XCTAssertFalse(node.geometry?.sources(for: .vertex).isEmpty ?? true,
                       "mesh geometry has no vertex source")
    }

    func testMissingFileLoadsNil() {
        let bogus = URL(fileURLWithPath: "/definitely/not/a/mesh-\(UUID().uuidString).glb")
        XCTAssertNil(GLBMeshLoader.loadScene(url: bogus))
    }

    // MARK: - Textured GLB (paint stage)

    private var texturedFixtureURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures/hy3d_textured_quad.glb")
    }

    /// The fixture's material rewritten the way the server writes a finish: a roughness multiplier and a clearcoat.
    private func fixtureWithFinish(roughness: Double, clearcoat: Double) throws -> URL {
        let glb = try Data(contentsOf: texturedFixtureURL)
        func le(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        let jsonLen = Int(glb.subdata(in: 12..<16).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: glb.subdata(in: 20..<(20 + jsonLen))) as? [String: Any])
        var materials = try XCTUnwrap(json["materials"] as? [[String: Any]])
        var pbr = try XCTUnwrap(materials[0]["pbrMetallicRoughness"] as? [String: Any])
        pbr["roughnessFactor"] = roughness
        materials[0]["pbrMetallicRoughness"] = pbr
        materials[0]["extensions"] = ["KHR_materials_clearcoat": ["clearcoatFactor": clearcoat]]
        json["materials"] = materials
        var body = try JSONSerialization.data(withJSONObject: json)
        while body.count % 4 != 0 { body.append(0x20) }
        let rest = glb.subdata(in: (20 + jsonLen)..<glb.count)
        var out = le(0x46546C67) + le(2) + le(UInt32(20 + body.count + rest.count))
        out += le(UInt32(body.count)) + le(0x4E4F534A) + body + rest
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("finish-\(UUID().uuidString).glb")
        try out.write(to: url)
        return url
    }

    func testThePreviewShowsTheFinishTheGlbDeclares() throws {
        guard FileManager.default.fileExists(atPath: texturedFixtureURL.path) else {
            throw XCTSkip("hy3d_textured_quad.glb fixture not present yet (produced by the Zig writer)")
        }
        func material(_ url: URL) throws -> SCNMaterial {
            let scene = try XCTUnwrap(GLBMeshLoader.loadScene(url: url))
            return try XCTUnwrap(GLBMeshLoader.firstGeometryNode(in: scene)?.geometry?.firstMaterial)
        }
        // A GLB with no finish keeps the texture's roughness as it is.
        let plain = try material(texturedFixtureURL)
        XCTAssertEqual(plain.roughness.intensity, 1.0, accuracy: 1e-9)
        // The multiplier scales the roughness texture and the coat shows as SceneKit's clear coat.
        let url = try fixtureWithFinish(roughness: 0.4, clearcoat: 0.8)
        defer { try? FileManager.default.removeItem(at: url) }
        let painted = try material(url)
        XCTAssertEqual(painted.roughness.intensity, 0.4, accuracy: 1e-6)
        XCTAssertEqual(painted.clearCoat.contents as? Double, 0.8)
    }

    func testLoadsTexturedFixtureWithUVsAndDiffuseImage() throws {
        guard FileManager.default.fileExists(atPath: texturedFixtureURL.path) else {
            throw XCTSkip("hy3d_textured_quad.glb fixture not present yet (produced by the Zig writer)")
        }
        let scene = try XCTUnwrap(GLBMeshLoader.loadScene(url: texturedFixtureURL))
        let node = try XCTUnwrap(GLBMeshLoader.firstGeometryNode(in: scene))
        let geo = try XCTUnwrap(node.geometry)
        // TEXCOORD_0 must arrive as a texcoord source with one UV per vertex.
        let uv = try XCTUnwrap(geo.sources(for: .texcoord).first, "no texcoord source")
        let pos = try XCTUnwrap(geo.sources(for: .vertex).first)
        XCTAssertEqual(uv.vectorCount, pos.vectorCount)
        XCTAssertEqual(uv.componentsPerVector, 2)
        // The embedded albedo PNG must land as the diffuse material contents.
        let material = try XCTUnwrap(geo.firstMaterial, "no material on textured mesh")
        XCTAssertNotNil(material.diffuse.contents as? NSImage,
                        "albedo texture did not decode into the diffuse slot")
    }

    // MARK: - Gaussian splat (Idol)

    /// A GLB the way the server's splat writer lays one out: one POINTS primitive, `COLOR_0` as normalized bytes.
    private func splatGlb(points: [(Float, Float, Float)], colors: [[UInt8]]) throws -> URL {
        func le(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        var bin = Data()
        for p in points { for v in [p.0, p.1, p.2] { bin.append(withUnsafeBytes(of: v.bitPattern.littleEndian) { Data($0) }) } }
        let colorOffset = bin.count
        for c in colors { bin.append(contentsOf: c) }
        while bin.count % 4 != 0 { bin.append(0) }
        let json: [String: Any] = [
            "asset": ["version": "2.0"],
            "extensionsUsed": ["KHR_gaussian_splatting"],
            "buffers": [["byteLength": bin.count]],
            "bufferViews": [["buffer": 0, "byteOffset": 0, "byteLength": colorOffset],
                            ["buffer": 0, "byteOffset": colorOffset, "byteLength": colors.count * 4]],
            "accessors": [["bufferView": 0, "componentType": 5126, "count": points.count, "type": "VEC3"],
                          ["bufferView": 1, "componentType": 5121, "normalized": true, "count": colors.count, "type": "VEC4"]],
            "meshes": [["primitives": [["mode": 0, "attributes": ["POSITION": 0, "COLOR_0": 1],
                                        "extensions": ["KHR_gaussian_splatting": ["kernel": "ellipse", "colorSpace": "srgb_rec709_display"]]]]]],
        ]
        var body = try JSONSerialization.data(withJSONObject: json)
        while body.count % 4 != 0 { body.append(0x20) }
        var out = le(0x46546C67) + le(2) + le(UInt32(12 + 8 + body.count + 8 + bin.count))
        out += le(UInt32(body.count)) + le(0x4E4F534A) + body + le(UInt32(bin.count)) + le(0x004E4942) + bin
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("splat-\(UUID().uuidString).glb")
        try out.write(to: url)
        return url
    }

    func testASplatGlbPreviewsAsColouredPointsAndDropsTheNearlyTransparentOnes() throws {
        let url = try splatGlb(points: [(0, 0, 0), (0.1, 0.2, 0.3), (-0.2, 0.1, 0)],
                               colors: [[255, 0, 0, 255], [0, 255, 0, 200], [0, 0, 255, 10]])
        defer { try? FileManager.default.removeItem(at: url) }
        let scene = try XCTUnwrap(GLBMeshLoader.loadScene(url: url))
        let geo = try XCTUnwrap(GLBMeshLoader.firstGeometryNode(in: scene)?.geometry)
        XCTAssertEqual(geo.elements.first?.primitiveType, .point)
        // The third point's opacity is 10/255: left out.
        XCTAssertEqual(try XCTUnwrap(geo.sources(for: .vertex).first).vectorCount, 2)
        let colour = try XCTUnwrap(geo.sources(for: .color).first)
        XCTAssertEqual(colour.vectorCount, 2)
        XCTAssertEqual(colour.componentsPerVector, 4)
    }

    /// A splat GLB with the extension's own attributes (linear opacity and scale, quaternion x y z w, SH0).
    private func fullSplatGlb() throws -> Data {
        func le(_ v: UInt32) -> Data { withUnsafeBytes(of: v.littleEndian) { Data($0) } }
        func f(_ xs: [Float]) -> Data { xs.withUnsafeBytes { Data($0) } }
        var bin = Data()
        var views: [[String: Any]] = []
        var accessors: [[String: Any]] = []
        let attrs: [(String, String, [Float])] = [
            ("POSITION", "VEC3", [0.1, 0.2, 0.3]),
            ("KHR_gaussian_splatting:ROTATION", "VEC4", [0.5, 0, 0, Float(0.75).squareRoot()]),
            ("KHR_gaussian_splatting:SCALE", "VEC3", [0.01, 0.02, 0.03]),
            ("KHR_gaussian_splatting:OPACITY", "SCALAR", [0.5]),
            ("KHR_gaussian_splatting:SH_DEGREE_0_COEF_0", "VEC3", [0.25, -0.5, 1.0]),
        ]
        var map: [String: Int] = [:]
        for (i, a) in attrs.enumerated() {
            views.append(["buffer": 0, "byteOffset": bin.count, "byteLength": a.2.count * 4])
            accessors.append(["bufferView": i, "componentType": 5126, "count": 1, "type": a.1])
            bin.append(f(a.2))
            map[a.0] = i
        }
        let json: [String: Any] = [
            "asset": ["version": "2.0"], "buffers": [["byteLength": bin.count]], "bufferViews": views, "accessors": accessors,
            "meshes": [["primitives": [["mode": 0, "attributes": map, "extensions": ["KHR_gaussian_splatting": ["kernel": "ellipse"]]]]]],
        ]
        var body = try JSONSerialization.data(withJSONObject: json)
        while body.count % 4 != 0 { body.append(0x20) }
        return le(0x46546C67) + le(2) + le(UInt32(12 + 8 + body.count + 8 + bin.count))
            + le(UInt32(body.count)) + le(0x4E4F534A) + body + le(UInt32(bin.count)) + le(0x004E4942) + bin
    }

    func testASplatGlbExportsAsTheStandardPly() throws {
        let ply = try XCTUnwrap(SplatPLYExport.ply(fromGLB: try fullSplatGlb()))
        let end = try XCTUnwrap(ply.range(of: Data("end_header\n".utf8))).upperBound
        let header = String(decoding: ply.prefix(end), as: UTF8.self)
        XCTAssertTrue(header.hasPrefix("ply\nformat binary_little_endian 1.0\nelement vertex 1\n"))
        XCTAssertEqual(header.components(separatedBy: "property float ").count - 1, 17)
        XCTAssertEqual(ply.count, end + 17 * 4)
        let row: [Float] = ply.suffix(from: end).withUnsafeBytes { raw in (0..<17).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) } }
        // glTF y-up back to the reference's PLY axes: (x, y, z) -> (x, -y, -z).
        XCTAssertEqual(row[0], 0.1, accuracy: 1e-6)
        XCTAssertEqual(row[1], -0.2, accuracy: 1e-6)
        XCTAssertEqual(row[2], -0.3, accuracy: 1e-6)
        XCTAssertEqual(row[6], 0.25, accuracy: 1e-6)          // SH0 as stored
        XCTAssertEqual(row[9], 0, accuracy: 1e-6)             // logit(0.5)
        XCTAssertEqual(row[10], logf(0.01), accuracy: 1e-5)   // log scale
        XCTAssertNil(SplatPLYExport.ply(fromGLB: Data([1, 2, 3])))
    }

    func testASplatWithRotationAndScaleBecomesOneOrientedDiscPerGaussian() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("disc-\(UUID().uuidString).glb")
        try fullSplatGlb().write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let geo = try XCTUnwrap(GLBMeshLoader.firstGeometryNode(in: try XCTUnwrap(GLBMeshLoader.loadScene(url: url)))?.geometry)
        XCTAssertEqual(geo.elements.first?.primitiveType, .triangles)
        XCTAssertEqual(try XCTUnwrap(geo.sources(for: .vertex).first).vectorCount, 4, "a quad per Gaussian")
        XCTAssertEqual(geo.elements.first?.primitiveCount, 2)
    }
}
