import XCTest
import Foundation
@testable import MLXCore

/// Verifies the spawn-time fast-fail path: when an MCP server dies during init (as docker-mcp does
/// when the Docker daemon is down), `startEnabled` must return promptly with a clear error — not
/// hang for 30s+ waiting for an `initialize` reply that will never arrive.
@MainActor
final class MCPDockerSpawnTests: XCTestCase {

    /// Sandbox the mcp.json path so tests never clobber the user's real `~/.mlx-serve/mcp.json`.
    /// Honored by `MCPConfigStore.path` via the `MCP_CONFIG_PATH` env var.
    private var sandboxPath: String!

    override func setUp() async throws {
        let dir = NSTemporaryDirectory().appending("mcp-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        sandboxPath = (dir as NSString).appendingPathComponent("mcp.json")
        setenv("MCP_CONFIG_PATH", sandboxPath, 1)
    }

    override func tearDown() async throws {
        if let p = sandboxPath {
            let dir = (p as NSString).deletingLastPathComponent
            try? FileManager.default.removeItem(atPath: dir)
        }
        unsetenv("MCP_CONFIG_PATH")
    }

    func testStartEnabledFailsFastWhenServerDiesDuringInit() async throws {
        try XCTSkipUnless(BuildFeatures.current.hostShell, "spawns a host script; the App Store build routes MCP into the guest")
        // Hermetic stand-in for docker-mcp's daemon-down behavior: print the failure to stderr and
        // exit non-zero before ever speaking MCP. The real `npx -y docker-mcp` proved too
        // environment-dependent for CI (cold npx cache or a hung child outlives the 30s connect
        // timeout); the code path under test — connectOrFailFast's terminationHandler race — is
        // identical for any stdio server that dies during init.
        let dir = (sandboxPath as NSString).deletingLastPathComponent
        let script = (dir as NSString).appendingPathComponent("dying-mcp-server.sh")
        try """
        #!/bin/sh
        sleep 0.5
        echo 'Error: Docker daemon is not accessible' >&2
        exit 1
        """.write(toFile: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script)

        let manager = MCPManager()
        var config = MCPConfig()
        config.mcpServers["docker"] = MCPServerEntry(
            command: script, args: [], env: nil, disabled: false
        )
        try manager.saveConfig(config)

        // Child dies in ~0.5s; anything approaching the 30s connect timeout means the
        // terminationHandler fast-fail path didn't fire.
        let cap: TimeInterval = 10
        let started = Date()
        await manager.startEnabled()
        let elapsed = Date().timeIntervalSince(started)

        XCTAssertLessThan(elapsed, cap, "startEnabled didn't fail-fast on dead child (took \(elapsed)s, cap \(cap)s)")
        XCTAssertNil(manager.sessions["docker"], "Session should not be present for a server that crashed during init")

        guard let err = manager.startErrors["docker"] else {
            return XCTFail("Expected docker startup error, got nothing. Sessions: \(manager.sessions.keys)")
        }
        let lower = err.lowercased()
        // Must mention either the daemon issue (server's stderr captured) or the fact that the server exited.
        let mentionsRoot = lower.contains("daemon")
            || lower.contains("not accessible")
            || lower.contains("exited")
            || lower.contains("docker")
        XCTAssertTrue(mentionsRoot, "Error message doesn't surface the underlying cause: \(err)")
        print("[test] elapsed=\(String(format: "%.2f", elapsed))s, error=\(String(err.prefix(400)))")
    }

    /// Case A: the user configured an MCP server whose command isn't on PATH at all.
    /// Pre-flight `command -v` check should reject it instantly with a typed `commandNotFound` error.
    func testStartEnabledFailsFastWhenCommandIsMissing() async throws {
        let manager = MCPManager()
        var config = MCPConfig()
        config.mcpServers["bogus"] = MCPServerEntry(
            command: "definitely-not-a-real-binary-xyz", args: [], env: nil, disabled: false
        )
        try manager.saveConfig(config)

        let started = Date()
        await manager.startEnabled()
        let elapsed = Date().timeIntervalSince(started)

        // Pre-flight only spawns a 1-shot zsh; should be sub-second.
        XCTAssertLessThan(elapsed, 5, "Missing-command precheck shouldn't take \(elapsed)s")
        XCTAssertNil(manager.sessions["bogus"])
        guard let err = manager.startErrors["bogus"] else {
            return XCTFail("Expected error for missing command, got nothing")
        }
        XCTAssertTrue(err.contains("not found"), "Error should say 'not found'; got: \(err)")
        XCTAssertTrue(err.contains("definitely-not-a-real-binary-xyz"), "Error should name the missing command; got: \(err)")
        print("[test] missing-command elapsed=\(String(format: "%.2f", elapsed))s, error=\(String(err.prefix(200)))")
    }

    /// Case B: the command exists (npx) but the package it's asked to run doesn't.
    /// npx itself prints "404 Not Found" to stderr and exits — terminationHandler fast-fails for us.
    func testStartEnabledFailsFastWhenNpxPackageIsMissing() async throws {
        #if MAS_BUILD
        throw XCTSkip("host npx spawn is compiled out of the App Store build")
        #else
        guard await HostMCPSpawner().commandExists("npx") else {
            throw XCTSkip("npx not on PATH")
        }
        let manager = MCPManager()
        var config = MCPConfig()
        config.mcpServers["nopkg"] = MCPServerEntry(
            command: "npx", args: ["-y", "@absolutely/no-such-package-9z9z9z@0.0.0"], env: nil, disabled: false
        )
        try manager.saveConfig(config)

        let started = Date()
        await manager.startEnabled()
        let elapsed = Date().timeIntervalSince(started)

        // npm registry check + 404 takes a few seconds at worst on a slow connection.
        XCTAssertLessThan(elapsed, 30, "Missing-package failure shouldn't take \(elapsed)s")
        XCTAssertNil(manager.sessions["nopkg"])
        guard let err = manager.startErrors["nopkg"] else {
            return XCTFail("Expected error for missing npm package, got nothing")
        }
        // The error should reflect either npm's 404 or our exit-early surface.
        let lower = err.lowercased()
        let recognized = lower.contains("exit") || lower.contains("404") || lower.contains("not found") || lower.contains("error")
        XCTAssertTrue(recognized, "Error should surface npm failure; got: \(err)")
        print("[test] missing-package elapsed=\(String(format: "%.2f", elapsed))s, error=\(String(err.prefix(400)))")
        #endif
    }

    /// Regression: when an entry that previously failed to start is later disabled, the stale error
    /// from the failed attempt was lingering and showing up in the inline chat warning even though
    /// the user had toggled it off. `startEnabled` now purges errors for non-enabled entries.
    func testStartEnabledClearsStaleErrorsForDisabledEntries() async throws {
        let manager = MCPManager()
        var config = MCPConfig()
        config.mcpServers["nopkg"] = MCPServerEntry(
            command: "definitely-not-real-xyz", args: [], env: nil, disabled: false
        )
        try manager.saveConfig(config)
        await manager.startEnabled()
        XCTAssertNotNil(manager.startErrors["nopkg"], "Should have an error after first failure")

        // Now disable it (mirrors what the user does in the marketplace toggle).
        var disabled = config
        disabled.mcpServers["nopkg"]?.disabled = true
        try manager.saveConfig(disabled)
        await manager.startEnabled()
        XCTAssertNil(manager.startErrors["nopkg"],
                     "Disabled entries shouldn't keep showing stale errors; got: \(manager.startErrors)")
    }
}
