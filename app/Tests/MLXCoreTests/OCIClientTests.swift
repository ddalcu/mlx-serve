import XCTest
@testable import MLXCore

/// Pure-logic tests for the Swift OCI image puller that provisions the agent
/// sandbox rootfs: reference parsing, registry auth challenge parsing, manifest
/// index selection, image-config extraction, whiteout handling, and gunzip.
/// No network — the live pull is covered by SandboxSmoke REAL_PROVISION.
final class OCIClientTests: XCTestCase {

    // MARK: image reference parsing (docker conventions)

    func testParseBareOfficialImage() {
        let r = OCIClient.parseImageRef("alpine")
        XCTAssertEqual(r.registry, "registry-1.docker.io")
        XCTAssertEqual(r.repository, "library/alpine", "official images live under library/")
        XCTAssertEqual(r.tag, "latest")
    }

    func testParseUserImageWithTag() {
        let r = OCIClient.parseImageRef("ddalcu/agent-shell:v2")
        XCTAssertEqual(r.registry, "registry-1.docker.io")
        XCTAssertEqual(r.repository, "ddalcu/agent-shell")
        XCTAssertEqual(r.tag, "v2")
    }

    func testParseExplicitRegistry() {
        let r = OCIClient.parseImageRef("ghcr.io/acme/tool:1.0")
        XCTAssertEqual(r.registry, "ghcr.io")
        XCTAssertEqual(r.repository, "acme/tool")
        XCTAssertEqual(r.tag, "1.0")
    }

    func testParseRegistryWithPortDoesNotConfuseTagColon() {
        let r = OCIClient.parseImageRef("localhost:5000/team/img")
        XCTAssertEqual(r.registry, "localhost:5000")
        XCTAssertEqual(r.repository, "team/img")
        XCTAssertEqual(r.tag, "latest")
    }

    func testParseDigestReference() {
        let r = OCIClient.parseImageRef("alpine@sha256:abc123")
        XCTAssertEqual(r.repository, "library/alpine")
        XCTAssertEqual(r.tag, "sha256:abc123", "a digest ref pulls by digest — same GET path as a tag")
    }

    // MARK: WWW-Authenticate challenge (anonymous token dance)

    func testParseWWWAuthenticateBearer() {
        let h = #"Bearer realm="https://auth.docker.io/token",service="registry.docker.io",scope="repository:ddalcu/agent-shell:pull""#
        let c = OCIClient.parseWWWAuthenticate(h)
        XCTAssertEqual(c?.realm, "https://auth.docker.io/token")
        XCTAssertEqual(c?.params["service"], "registry.docker.io")
        XCTAssertEqual(c?.params["scope"], "repository:ddalcu/agent-shell:pull")
    }

    func testParseWWWAuthenticateNonBearerIsNil() {
        XCTAssertNil(OCIClient.parseWWWAuthenticate(#"Basic realm="x""#))
    }

    // MARK: manifest index → platform digest

    func testSelectManifestPicksArm64AndSkipsAttestations() {
        let index = """
        {"schemaVersion":2,"manifests":[
          {"digest":"sha256:amd","platform":{"architecture":"amd64","os":"linux"}},
          {"digest":"sha256:att","platform":{"architecture":"unknown","os":"unknown"}},
          {"digest":"sha256:arm","platform":{"architecture":"arm64","os":"linux"}}
        ]}
        """.data(using: .utf8)!
        XCTAssertEqual(OCIClient.selectManifestDigest(indexJSON: index, arch: "arm64"), "sha256:arm")
        XCTAssertEqual(OCIClient.selectManifestDigest(indexJSON: index, arch: "amd64"), "sha256:amd")
        XCTAssertNil(OCIClient.selectManifestDigest(indexJSON: index, arch: "riscv"),
                     "no match must be nil — pulling the wrong arch boots to an ENOEXEC panic")
    }

    func testSelectManifestOnDirectManifestIsNil() {
        // A direct (non-index) manifest has "layers", not "manifests" — the caller
        // should use it as-is.
        let direct = #"{"schemaVersion":2,"config":{"digest":"sha256:c"},"layers":[]}"#.data(using: .utf8)!
        XCTAssertNil(OCIClient.selectManifestDigest(indexJSON: direct, arch: "arm64"))
    }

    // MARK: image config (env + workdir reach the guest shell)

    func testParseImageConfigExtractsEnvAndWorkingDir() {
        let json = """
        {"architecture":"arm64","config":{
           "Env":["PATH=/usr/local/bin:/usr/bin","PIP_BREAK_SYSTEM_PACKAGES=1"],
           "WorkingDir":"/app","Cmd":["/bin/sh"]}}
        """.data(using: .utf8)!
        let c = OCIClient.parseImageConfig(json)
        XCTAssertEqual(c.env, ["PATH=/usr/local/bin:/usr/bin", "PIP_BREAK_SYSTEM_PACKAGES=1"])
        XCTAssertEqual(c.workingDir, "/app")
    }

    func testParseImageConfigTolerantOfMissingFields() {
        let c = OCIClient.parseImageConfig(#"{"config":{}}"#.data(using: .utf8)!)
        XCTAssertEqual(c.env, [])
        XCTAssertNil(c.workingDir)
    }

    // MARK: layer whiteouts (upper layer deletes a lower layer's path)

    func testApplyWhiteoutsRemovesTargetAndMarker() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("oci-wh-\(UUID().uuidString)")
        let fm = FileManager.default
        try fm.createDirectory(at: dir.appendingPathComponent("etc"), withIntermediateDirectories: true)
        try "old".write(to: dir.appendingPathComponent("etc/removed.conf"), atomically: true, encoding: .utf8)
        try "keep".write(to: dir.appendingPathComponent("etc/kept.conf"), atomically: true, encoding: .utf8)
        try Data().write(to: dir.appendingPathComponent("etc/.wh.removed.conf"))
        defer { try? fm.removeItem(at: dir) }

        try OCIClient.applyWhiteouts(in: dir)

        XCTAssertFalse(fm.fileExists(atPath: dir.appendingPathComponent("etc/removed.conf").path),
                       "the whited-out lower-layer file must be deleted")
        XCTAssertFalse(fm.fileExists(atPath: dir.appendingPathComponent("etc/.wh.removed.conf").path),
                       "the marker itself must not leak into the rootfs")
        XCTAssertTrue(fm.fileExists(atPath: dir.appendingPathComponent("etc/kept.conf").path))
    }

    func testApplyWhiteoutsRemovesWhitedOutDirectoryRecursively() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("oci-wh-\(UUID().uuidString)")
        let fm = FileManager.default
        try fm.createDirectory(at: dir.appendingPathComponent("opt/gone/sub"), withIntermediateDirectories: true)
        try "x".write(to: dir.appendingPathComponent("opt/gone/sub/f"), atomically: true, encoding: .utf8)
        try Data().write(to: dir.appendingPathComponent("opt/.wh.gone"))
        defer { try? fm.removeItem(at: dir) }

        try OCIClient.applyWhiteouts(in: dir)
        XCTAssertFalse(fm.fileExists(atPath: dir.appendingPathComponent("opt/gone").path))
    }

    // MARK: gunzip (kernel asset + fallback for layer blobs)

    func testGunzipDecompressesGzipWithFilenameHeader() throws {
        // gzip blob with an FNAME field ("kernel-test") — exercises header parsing
        // beyond the fixed 10 bytes. Payload: "hello vz sandbox\n" × 3.
        let b64 = "H4sICADxU2UC/2tlcm5lbC10ZXN0AMtIzcnJVyirUihOzEtJyq/gyiAoAABKEvpHMwAAAA=="
        let out = try OCIClient.gunzip(Data(base64Encoded: b64)!)
        XCTAssertEqual(String(decoding: out, as: UTF8.self),
                       String(repeating: "hello vz sandbox\n", count: 3))
    }

    func testGunzipRejectsNonGzipData() {
        XCTAssertThrowsError(try OCIClient.gunzip(Data("not gzip".utf8)))
    }

    // MARK: kernel capability self-heal

    func testKernelVirtiofsCapabilityDetection() {
        // A cached kernel from before the virtiofs config landed must be detected
        // as stale (the app then re-fetches). The marker is the literal fs name
        // the kernel registers.
        XCTAssertTrue(AgentSandbox.kernelHasVirtiofsSupport(Data("...ext4\0virtiofs\0overlay...".utf8)))
        XCTAssertFalse(AgentSandbox.kernelHasVirtiofsSupport(Data("...ext4\0overlay... 9p".utf8)))
    }
}
