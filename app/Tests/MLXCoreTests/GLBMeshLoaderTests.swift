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
}
