import Darwin
import XCTest
import System
@testable import MLXCore

final class MCPSpawnerRoutingTests: XCTestCase {

    func testSandboxOffRunsOnTheHost() {
        XCTAssertEqual(MCPSpawnerRouter.destination(sandboxEnabled: false, masBuild: false), .host)
    }

    /// The bug: this used to be `.host`.
    func testSandboxOnRunsInTheGuest() {
        XCTAssertEqual(MCPSpawnerRouter.destination(sandboxEnabled: true, masBuild: false), .guest)
    }

    /// The App Store build has no host arm at all — `HostMCPSpawner` is compiled
    /// out — so the sandbox toggle cannot route a server onto macOS.
    func testMASBuildAlwaysRunsInTheGuestRegardlessOfTheToggle() {
        XCTAssertEqual(MCPSpawnerRouter.destination(sandboxEnabled: false, masBuild: true), .guest)
        XCTAssertEqual(MCPSpawnerRouter.destination(sandboxEnabled: true, masBuild: true), .guest)
    }

    /// `spawner(sandboxEnabled:)` and `destination(...)` must never disagree —
    /// one is what runs, the other is what we test.
    func testConcreteSpawnerMatchesTheDeclaredDestination() {
        for enabled in [true, false] {
            let spawner = MCPSpawnerRouter.spawner(sandboxEnabled: enabled)
            let destination = MCPSpawnerRouter.destination(sandboxEnabled: enabled)
            switch destination {
            case .guest: XCTAssertTrue(spawner is GuestMCPSpawner, "enabled=\(enabled)")
            case .host:
                #if MAS_BUILD
                XCTFail("MAS build must never route to the host")
                #else
                XCTAssertTrue(spawner is HostMCPSpawner, "enabled=\(enabled)")
                #endif
            }
        }
    }

    func testBuildFeaturesGateTheHostSpawner() {
        #if MAS_BUILD
        XCTAssertTrue(BuildFeatures.current.isMAS)
        XCTAssertFalse(BuildFeatures.current.hostShell)
        XCTAssertEqual(BuildFeatures.current.guest.mcpServers, .prebaked)
        #else
        XCTAssertFalse(BuildFeatures.current.isMAS)
        XCTAssertTrue(BuildFeatures.current.hostShell)
        XCTAssertEqual(BuildFeatures.current.guest.mcpServers, .fetched)
        #endif
    }
}

// MARK: - Pre-baked guest bins (guideline 2.5.2)

/// Proven live: `npx -y <package>` in the guest hits registry.npmjs.org even
/// when the package is pre-baked (`npm -g` globals are ignored by npx) — so
/// every MCP start DOWNLOADED CODE, the exact thing guideline 2.5.2 forbids.
/// The pre-baked bins do run offline when invoked directly. `MCPGuestPrebaked`
/// is the translator that turns the npx form into the direct-bin form.
final class MCPGuestPrebakedTests: XCTestCase {

    // MARK: translate

    func testNpxFilesystemTranslatesToTheGuestBinKeepingItsArgs() {
        let got = MCPGuestPrebaked.translate(
            command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem", "/data/project"])
        XCTAssertEqual(got?.command, "mcp-server-filesystem")
        XCTAssertEqual(got?.args, ["/data/project"])
    }

    func testNpxGitHubTranslatesToTheGuestBin() {
        let got = MCPGuestPrebaked.translate(
            command: "npx", args: ["-y", "@modelcontextprotocol/server-github"])
        XCTAssertEqual(got?.command, "mcp-server-github")
        XCTAssertEqual(got?.args, [])
    }

    /// A user may hand-author the direct bin in mcp.json — that IS the pre-baked
    /// form, so it passes through untouched.
    func testDirectGuestBinPassesThroughUnchanged() {
        let got = MCPGuestPrebaked.translate(
            command: "mcp-server-filesystem", args: ["/data"])
        XCTAssertEqual(got?.command, "mcp-server-filesystem")
        XCTAssertEqual(got?.args, ["/data"])
    }

    func testUnknownNpxPackageIsNotPrebaked() {
        XCTAssertNil(MCPGuestPrebaked.translate(command: "npx", args: ["-y", "docker-mcp"]))
    }

    /// A version suffix must not defeat the lookup in either direction: a pinned
    /// pre-baked package still resolves, and a versioned non-pre-baked scoped
    /// package (dbhub@latest) still returns nil.
    func testVersionSuffixIsStrippedBeforeTheLookup() {
        let pinned = MCPGuestPrebaked.translate(
            command: "npx", args: ["-y", "@modelcontextprotocol/server-github@2025.1.0"])
        XCTAssertEqual(pinned?.command, "mcp-server-github")
        XCTAssertNil(MCPGuestPrebaked.translate(
            command: "npx", args: ["-y", "@bytebase/dbhub@latest", "--transport", "stdio"]))
    }

    func testNonNpxCommandsAreNotPrebaked() {
        XCTAssertNil(MCPGuestPrebaked.translate(command: "uvx", args: ["some-server"]))
        XCTAssertNil(MCPGuestPrebaked.translate(command: "python3", args: ["-m", "server"]))
    }

    // MARK: invocation policy

    func testPrebakedOnlyRejectsAnUnknownServerWithTheHonestError() {
        XCTAssertThrowsError(try MCPGuestPrebaked.invocation(
            command: "npx", args: ["-y", "@playwright/mcp@latest"], prebakedOnly: true)) { error in
            guard case MCPSpawnError.notPrebaked = error else {
                return XCTFail("expected .notPrebaked, got \(error)")
            }
            // The message must say what this build CAN run, not "not found on PATH".
            let msg = (error as? LocalizedError)?.errorDescription ?? ""
            XCTAssertTrue(msg.contains("bundled"), msg)
        }
    }

    func testPrebakedOnlyTranslatesAKnownServer() throws {
        let got = try MCPGuestPrebaked.invocation(
            command: "npx", args: ["-y", "@modelcontextprotocol/server-filesystem", "/data"],
            prebakedOnly: true)
        XCTAssertEqual(got.command, "mcp-server-filesystem")
        XCTAssertEqual(got.args, ["/data"])
    }

    /// The DMG build keeps today's behavior for servers outside the table:
    /// npx in the guest is allowed there.
    func testFetchedPolicyPassesAnUnknownServerThroughAsAuthored() throws {
        let got = try MCPGuestPrebaked.invocation(
            command: "npx", args: ["-y", "docker-mcp"], prebakedOnly: false)
        XCTAssertEqual(got.command, "npx")
        XCTAssertEqual(got.args, ["-y", "docker-mcp"])
    }

    /// CLASS INVARIANT: under the pre-baked-only policy NO catalog entry —
    /// raw or materialized with user values — may resolve to an npx invocation.
    /// npx is the code-download vector; if this fails, 2.5.2 is back.
    func testMASDestinationNeverYieldsAnNpxInvocation() throws {
        for entry in MCPCatalog.visibleEntries(prebakedOnly: true) {
            let raw = try MCPGuestPrebaked.invocation(
                command: entry.command, args: entry.args, prebakedOnly: true)
            XCTAssertFalse(raw.command.contains("npx"), entry.id)

            // Materialized with a value for every input, like a real spawn.
            var values: [String: String] = [:]
            for input in entry.inputs { values[input.id] = "test-value" }
            let materialized = entry.materialize(values: values)
            let live = try MCPGuestPrebaked.invocation(
                command: materialized.command ?? "", args: materialized.args ?? [],
                prebakedOnly: true)
            XCTAssertFalse(live.command.contains("npx"), entry.id)
        }
    }

    // MARK: marketplace filter

    func testMarketplaceShowsOnlyPrebakedEntriesUnderMAS() {
        let visible = MCPCatalog.visibleEntries(prebakedOnly: true).map(\.id)
        XCTAssertEqual(Set(visible), ["github", "filesystem"],
                       "only servers whose bins ride the bundled guest image may be offered")
    }

    func testMarketplaceShowsTheFullCatalogOnTheDMGBuild() {
        XCTAssertEqual(MCPCatalog.visibleEntries(prebakedOnly: false).map(\.id),
                       MCPCatalog.entries.map(\.id))
    }

    /// A configured-but-hidden catalog server (e.g. a playwright entry carried
    /// over from a DMG install) must surface in the CUSTOM section under MAS —
    /// invisible config that errors at spawn would be a support trap.
    func testHiddenCatalogEntryIsTreatedAsCustomUnderMAS() {
        XCTAssertNil(MCPCatalog.visibleEntry(for: "playwright", prebakedOnly: true))
        XCTAssertNotNil(MCPCatalog.visibleEntry(for: "playwright", prebakedOnly: false))
        XCTAssertNotNil(MCPCatalog.visibleEntry(for: "filesystem", prebakedOnly: true))
    }

    /// The concrete spawner must follow the same policy source the tests use.
    func testGuestSpawnerPolicyMatchesBuildFeatures() {
        XCTAssertEqual(GuestMCPSpawner().prebakedOnly,
                       BuildFeatures.current.guest.mcpServers == .prebaked)
    }
}

/// Both spawners must build the same invocation, or a server that works on the
/// host silently behaves differently in the guest.
final class MCPCommandLineTests: XCTestCase {

    func testExecReplacesTheShellSoFdsSurvive() {
        let line = MCPCommandLine.shellEscape(command: "npx", args: ["-y", "server"])
        XCTAssertEqual(line, "exec 'npx' '-y' 'server'")
    }

    func testSingleQuotesInTokensCannotEscape() {
        let line = MCPCommandLine.shellEscape(command: "npx", args: ["it's"])
        XCTAssertEqual(line, #"exec 'npx' 'it'"'"'s'"#)
    }

    func testSpacesAndSpecialsAreQuoted() {
        let line = MCPCommandLine.shellEscape(command: "/usr/bin/my server", args: ["$HOME; rm -rf /"])
        XCTAssertEqual(line, "exec '/usr/bin/my server' '$HOME; rm -rf /'")
    }
}

#if !MAS_BUILD
/// The spawner seam carries ONLY per-server env overrides — the guest spawner
/// lays them over the guest image's env, so the manager must not pre-merge the
/// whole macOS environment into them (that clobbered the guest's PATH/HOME).
/// The host spawner therefore owns its own merge; these pin that both halves
/// of the contract hold on the host side.
final class HostMCPSpawnerEnvTests: XCTestCase {

    func testHostSpawnerMergesOverridesOverTheAppEnvironment() async throws {
        let stdio = try await HostMCPSpawner().open(
            command: "env", args: [], cwd: NSTemporaryDirectory(),
            env: ["MCP_SPAWNER_TEST_VAR": "injected"])
        defer { stdio.child.terminate() }

        var out = Data()
        var buf = [UInt8](repeating: 0, count: 16 * 1024)
        let deadline = Date().addingTimeInterval(10)
        while Date() < deadline {
            let n = read(stdio.input.rawValue, &buf, buf.count)
            if n > 0 { out.append(contentsOf: buf[0..<n]); continue }
            if n == 0 { break }
            if errno != EAGAIN && errno != EINTR { break }
        }
        let text = String(decoding: out, as: UTF8.self)
        // The override arrived…
        XCTAssertTrue(text.contains("MCP_SPAWNER_TEST_VAR=injected"), text.prefix(500).description)
        // …and did NOT replace the environment wholesale: the app's own env
        // (PATH at minimum) is still there underneath.
        XCTAssertTrue(text.contains("PATH="), text.prefix(500).description)
    }
}
#endif

/// `Process` and `GuestMCPBridge` both have to satisfy `MCPChild`, because
/// `connectOrFailFast` polls it instead of using `Process.terminationHandler`
/// (a guest server has no such callback).
final class MCPChildContractTests: XCTestCase {

    func testProcessReportsRunningThenItsExitStatus() throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/sh")
        p.arguments = ["-c", "exit 7"]
        try p.run()

        let child: MCPChild = p
        p.waitUntilExit()
        XCTAssertFalse(child.isRunning)
        XCTAssertEqual(child.exitStatus, 7)
    }

    func testRunningProcessHasNoExitStatus() throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/sh")
        p.arguments = ["-c", "sleep 5"]
        try p.run()
        defer { p.terminate() }

        let child: MCPChild = p
        XCTAssertTrue(child.isRunning)
        XCTAssertNil(child.exitStatus, "a live process must not report an exit status")
    }
}

// MARK: - The guest bridge, against the real Zig agent

/// `GuestMCPBridge` adapts the guest's framed exec protocol to the raw
/// descriptor `StdioTransport` drives. Bidirectional, long-lived, and the piece
/// where a framing bug would look like "the MCP server just hangs".
///
///     zig build vz-agent-host
///     VZ_AGENT_HOST_BIN=zig-out/bin/vz-agent-host swift test --filter MCPGuestBridge
final class MCPGuestBridgeInteropTests: XCTestCase {

    private var agent: Process?
    private var socketPath = ""

    override func setUpWithError() throws {
        guard let binary = ProcessInfo.processInfo.environment["VZ_AGENT_HOST_BIN"],
              FileManager.default.isExecutableFile(atPath: binary) else {
            throw XCTSkip("set VZ_AGENT_HOST_BIN (build it with `zig build vz-agent-host`)")
        }
        socketPath = NSTemporaryDirectory() + "vzb-\(UUID().uuidString.prefix(8)).sock"

        let process = Process()
        process.executableURL = URL(fileURLWithPath: binary)
        var env = ProcessInfo.processInfo.environment
        env["VZ_AGENT_UNIX_SOCKET"] = socketPath
        process.environment = env
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        agent = process

        let deadline = Date().addingTimeInterval(5)
        while Date() < deadline, !FileManager.default.fileExists(atPath: socketPath) { usleep(20_000) }
        guard FileManager.default.fileExists(atPath: socketPath) else {
            throw XCTSkip("vz-agent-host never created \(socketPath)")
        }
    }

    override func tearDown() {
        agent?.terminate()
        try? FileManager.default.removeItem(atPath: socketPath)
    }

    /// Connect and hand the agent a request, exactly as `AgentSandbox.openMCPBridge` does.
    private func bridge(running command: String) throws -> GuestMCPBridge {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        addr.sun_len = UInt8(MemoryLayout<sockaddr_un>.size)
        _ = withUnsafeMutablePointer(to: &addr.sun_path) { ptr in
            socketPath.withCString { src in
                strncpy(UnsafeMutableRawPointer(ptr).assumingMemoryBound(to: CChar.self), src, 103)
            }
        }
        let ok = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.connect(fd, $0, socklen_t(MemoryLayout<sockaddr_un>.size))
            }
        }
        guard ok == 0 else { Darwin.close(fd); throw XCTSkip("connect failed") }

        let stream = FileDescriptorStream(fd: fd)
        let request = GuestProtocol.Request(command: command)
        try stream.write(GuestProtocol.frame(.request, request.encode()))
        return try GuestMCPBridge(stream: stream)
    }

    private func read(_ fd: FileDescriptor, upTo count: Int, timeout: TimeInterval = 5) -> Data {
        var out = Data()
        let deadline = Date().addingTimeInterval(timeout)
        var buffer = [UInt8](repeating: 0, count: 4096)
        while out.count < count, Date() < deadline {
            var pfd = pollfd(fd: fd.rawValue, events: Int16(POLLIN), revents: 0)
            if poll(&pfd, 1, 100) <= 0 { continue }
            let n = Darwin.read(fd.rawValue, &buffer, buffer.count)
            if n <= 0 { break }
            out.append(contentsOf: buffer[0..<n])
        }
        return out
    }

    /// stdin → server → stdout, over the socketpair. This is the whole contract
    /// `StdioTransport` relies on.
    func testBridgeShuttlesStdinToStdout() throws {
        let bridge = try self.bridge(running: "cat")
        defer { bridge.terminate() }
        XCTAssertTrue(bridge.isRunning)

        let payload = Data(#"{"jsonrpc":"2.0"}"#.utf8) + Data("\n".utf8)
        _ = payload.withUnsafeBytes { Darwin.write(bridge.transportDescriptor.rawValue, $0.baseAddress, $0.count) }

        let echoed = read(bridge.transportDescriptor, upTo: payload.count)
        XCTAssertEqual(echoed, payload)
    }

    /// The server's stderr must not be interleaved into stdout — JSON-RPC would
    /// choke. It lands in the box `MCPManager` surfaces on failure.
    func testServerStderrIsCapturedNotMixedIntoStdout() throws {
        let bridge = try self.bridge(running: "echo out; echo boom 1>&2; sleep 0.3")
        defer { bridge.terminate() }

        let stdout = read(bridge.transportDescriptor, upTo: 4)
        XCTAssertEqual(String(decoding: stdout, as: UTF8.self), "out\n")

        let deadline = Date().addingTimeInterval(3)
        while bridge.stderr.snapshot().isEmpty, Date() < deadline { usleep(20_000) }
        XCTAssertTrue(bridge.stderr.snapshot().contains("boom"), bridge.stderr.snapshot())
        XCTAssertFalse(String(decoding: stdout, as: UTF8.self).contains("boom"))
    }

    /// A server that dies must flip `isRunning` and report its status, or the
    /// handshake watcher waits out the full 30 s cap.
    func testServerExitIsObservableWithItsStatus() throws {
        let bridge = try self.bridge(running: "exit 3")
        defer { bridge.terminate() }

        let deadline = Date().addingTimeInterval(5)
        while bridge.isRunning, Date() < deadline { usleep(20_000) }

        XCTAssertFalse(bridge.isRunning, "bridge never noticed the server exited")
        XCTAssertEqual(bridge.exitStatus, 3)
    }

    /// Terminating closes the guest stream; the agent's poll sees the hangup and
    /// kills the server. A leaked `sleep 60` in the guest would be invisible.
    func testTerminateStopsALongRunningServer() throws {
        let bridge = try self.bridge(running: "sleep 60")
        XCTAssertTrue(bridge.isRunning)
        bridge.terminate()
        XCTAssertFalse(bridge.isRunning)
    }

    /// EOF on the SDK's side must reach the server as stdin EOF, or `cat` never
    /// exits and `disconnect()` hangs.
    func testClosingTheTransportEndEofsTheServer() throws {
        let bridge = try self.bridge(running: "cat; echo done")
        defer { bridge.terminate() }

        let payload = Data("hi\n".utf8)
        _ = payload.withUnsafeBytes { Darwin.write(bridge.transportDescriptor.rawValue, $0.baseAddress, $0.count) }
        XCTAssertEqual(read(bridge.transportDescriptor, upTo: 3), payload)

        // Half-close: no more stdin. `cat` should EOF and the shell prints "done".
        shutdown(bridge.transportDescriptor.rawValue, SHUT_WR)
        let tail = read(bridge.transportDescriptor, upTo: 5)
        XCTAssertEqual(String(decoding: tail, as: UTF8.self), "done\n")
    }
}
