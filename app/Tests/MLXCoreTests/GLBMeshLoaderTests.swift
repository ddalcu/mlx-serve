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
