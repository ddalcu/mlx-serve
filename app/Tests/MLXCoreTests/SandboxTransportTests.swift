import XCTest
@testable import MLXCore

/// The vsock transport has two independent halves — a kernel with
/// `CONFIG_VIRTIO_VSOCKETS`, and the `vz-agent` binary to inject. If either is
/// missing, the guest must boot the legacy console shell rather than hang on a
/// handshake that can never complete.
final class SandboxTransportTests: XCTestCase {

    private func kernel(withVsock: Bool, withVirtiofs: Bool = true) -> Data {
        var data = Data("linux 6.6 arbitrary padding".utf8)
        if withVirtiofs { data.append(Data("virtiofs".utf8)) }
        if withVsock { data.append(Data("virtio_vsock".utf8)) }
        data.append(Data(repeating: 0x41, count: 64))
        return data
    }

    // MARK: transport choice

    func testVsockNeedsBothTheKernelAndTheAgent() {
        XCTAssertEqual(AgentSandbox.chooseTransport(kernelData: kernel(withVsock: true),
                                                    agentBinary: "/tmp/vz-agent"), .vsock)
    }

    func testKernelWithoutVsockFallsBackToTheConsoleShell() {
        XCTAssertEqual(AgentSandbox.chooseTransport(kernelData: kernel(withVsock: false),
                                                    agentBinary: "/tmp/vz-agent"), .legacyConsole)
    }

    func testMissingAgentBinaryFallsBackToTheConsoleShell() {
        XCTAssertEqual(AgentSandbox.chooseTransport(kernelData: kernel(withVsock: true),
                                                    agentBinary: nil), .legacyConsole)
    }

    func testUnreadableKernelFallsBackRatherThanCrashing() {
        XCTAssertEqual(AgentSandbox.chooseTransport(kernelData: nil, agentBinary: "/tmp/vz-agent"),
                       .legacyConsole)
    }

    func testKernelSupportProbes() {
        XCTAssertTrue(AgentSandbox.kernelHasVsockSupport(kernel(withVsock: true)))
        XCTAssertFalse(AgentSandbox.kernelHasVsockSupport(kernel(withVsock: false)))
        XCTAssertTrue(AgentSandbox.kernelHasVirtiofsSupport(kernel(withVsock: true)))
        XCTAssertFalse(AgentSandbox.kernelHasVirtiofsSupport(kernel(withVsock: true, withVirtiofs: false)))
    }

    // MARK: kernel cache is tag-versioned

    /// Bumping the kernel must invalidate every older cache by construction —
    /// not by remembering to add another byte-sniffing probe.
    func testStaleKernelCachesArePrunedIncludingTheUntaggedOne() {
        let existing = ["kernel", "kernel-kernels-v2", "kernel-kernels-v3", "images", "notes.txt"]
        let stale = AgentSandbox.staleKernelNames(existing, tag: "kernels-v3")
        XCTAssertEqual(Set(stale), ["kernel", "kernel-kernels-v2"])
        XCTAssertFalse(stale.contains("images"))
        XCTAssertFalse(stale.contains("notes.txt"))
    }

    func testKernelCacheNameCarriesTheTag() {
        XCTAssertEqual(AgentSandbox.kernelCacheName(tag: "kernels-v3"), "kernel-kernels-v3")
        XCTAssertTrue(AgentSandbox.kernelURL.absoluteString.contains(AgentSandbox.kernelTag))
    }

    // MARK: agent binary discovery

    func testAgentBinaryPathPrefersTheEnvironmentOverride() throws {
        let tmp = NSTemporaryDirectory() + "vz-agent-\(UUID().uuidString)"
        FileManager.default.createFile(atPath: tmp, contents: Data("elf".utf8))
        defer { try? FileManager.default.removeItem(atPath: tmp) }

        XCTAssertEqual(
            AgentSandbox.agentBinaryPath(environment: ["VZ_AGENT_PATH": tmp],
                                         bundleResourceURL: nil, executableURL: nil),
            tmp)
    }

    func testAgentBinaryPathIgnoresAnOverrideThatDoesNotExist() {
        XCTAssertNil(AgentSandbox.agentBinaryPath(environment: ["VZ_AGENT_PATH": "/no/such/file"],
                                                  bundleResourceURL: nil, executableURL: nil))
    }

    func testAgentBinaryPathFindsTheBundledCopy() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("res-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root.appendingPathComponent("guest"),
                                                withIntermediateDirectories: true)
        let binary = root.appendingPathComponent("guest/vz-agent")
        FileManager.default.createFile(atPath: binary.path, contents: Data("elf".utf8))
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertEqual(AgentSandbox.agentBinaryPath(environment: [:],
                                                    bundleResourceURL: root, executableURL: nil),
                       binary.path)
    }
}

/// Under vsock every command is its own `sh -c`, so the Sandbox Terminal's
/// interactive `cd` no longer sticks on its own. The wrapper echoes the shell's
/// final `$PWD`; the host carries it to the next command.
final class SandboxTerminalCwdTests: XCTestCase {

    func testWrapperCdsInAndReportsThePwd() {
        let wrapped = AgentSandbox.wrapUserCommand("ls", guestCwd: "/workspace/sub dir")
        XCTAssertTrue(wrapped.contains("cd '/workspace/sub dir'"), wrapped)
        XCTAssertTrue(wrapped.contains(AgentSandbox.cwdMarker), wrapped)
    }

    /// `printf` would clobber `$?`. The wrapper must stash and re-raise it, or
    /// every command in the terminal reports success.
    func testWrapperPreservesTheExitStatus() {
        let wrapped = AgentSandbox.wrapUserCommand("false", guestCwd: "/workspace")
        XCTAssertTrue(wrapped.contains("__vz_rc=$?"), wrapped)
        XCTAssertTrue(wrapped.contains("exit $__vz_rc"), wrapped)
    }

    func testSplitExtractsTheCwdAndRemovesTheMarkerLine() {
        let output = "file1\nfile2\n\n\(AgentSandbox.cwdMarker)/workspace/deep\n"
        let split = AgentSandbox.splitCwdMarker(output)
        XCTAssertEqual(split.cwd, "/workspace/deep")
        XCTAssertEqual(split.body, "file1\nfile2\n")
        XCTAssertFalse(split.body.contains(AgentSandbox.cwdMarker))
    }

    /// Command output that happens to contain the marker text must not fool the
    /// parser — the real one is always last.
    func testSplitUsesTheLastMarkerNotTheFirst() {
        let output = "echoed \(AgentSandbox.cwdMarker)/decoy\nreal\n\(AgentSandbox.cwdMarker)/workspace\n"
        let split = AgentSandbox.splitCwdMarker(output)
        XCTAssertEqual(split.cwd, "/workspace")
    }

    /// A command that kills its own shell prints no marker. The cwd then simply
    /// doesn't move; the output must survive untouched.
    func testSplitWithoutAMarkerLeavesOutputAlone() {
        let split = AgentSandbox.splitCwdMarker("killed\n")
        XCTAssertNil(split.cwd)
        XCTAssertEqual(split.body, "killed\n")
    }

    func testSplitHandlesEmptyPwd() {
        let split = AgentSandbox.splitCwdMarker("x\n\(AgentSandbox.cwdMarker)\n")
        XCTAssertNil(split.cwd)
    }

    /// The marker rides stdout; stderr arrives on its own channel. Splitting
    /// the MERGED text at the marker dropped everything after it — i.e. all of
    /// stderr, exactly what a failing command's transcript entry needs. The
    /// composer parses stdout alone and re-attaches stderr.
    func testComposePreservesStderrFromAFailingCommand() {
        let out = AgentSandbox.composeUserOutput(
            stdout: "\n\(AgentSandbox.cwdMarker)/workspace\n",
            stderr: "ls: /nope: No such file or directory\n")
        XCTAssertEqual(out.cwd, "/workspace")
        XCTAssertEqual(out.body, "ls: /nope: No such file or directory\n")
    }

    func testComposeAppendsStderrAfterTheStdoutBody() {
        let out = AgentSandbox.composeUserOutput(
            stdout: "built ok\n\n\(AgentSandbox.cwdMarker)/workspace/app\n",
            stderr: "warning: deprecated API\n")
        XCTAssertEqual(out.cwd, "/workspace/app")
        XCTAssertEqual(out.body, "built ok\nwarning: deprecated API\n")
    }

    func testComposeWithoutStderrMatchesSplit() {
        let stdout = "hello\n\n\(AgentSandbox.cwdMarker)/w\n"
        let composed = AgentSandbox.composeUserOutput(stdout: stdout, stderr: "")
        let split = AgentSandbox.splitCwdMarker(stdout)
        XCTAssertEqual(composed.body, split.body)
        XCTAssertEqual(composed.cwd, split.cwd)
    }
}

/// A build with no host shell (the App Store build) cannot have the sandbox
/// "off": `ShellHandler.route` already forces every command into the guest, so
/// a false `isEnabled` would only LIE downstream — the system prompt's
/// execution-environment section would describe a macOS/zsh/brew host the
/// commands never touch, and the Sandbox Terminal + tray section would stay
/// hidden while a guest is actually running. The settings toggle still rules
/// the Developer ID build.
final class SandboxForcedOnTests: XCTestCase {

    func testHostShellBuildHonorsTheToggle() {
        XCTAssertFalse(AgentSandbox.resolveEnabled(requested: false, hostShellAllowed: true))
        XCTAssertTrue(AgentSandbox.resolveEnabled(requested: true, hostShellAllowed: true))
    }

    func testNoHostShellForcesTheSandboxOn() {
        XCTAssertTrue(AgentSandbox.resolveEnabled(requested: false, hostShellAllowed: false))
        XCTAssertTrue(AgentSandbox.resolveEnabled(requested: true, hostShellAllowed: false))
    }

    /// The live singleton must agree with the pure rule after `configure` —
    /// this is the path AppState drives with the persisted settings value, so a
    /// stale `enabled:false` blob from an old container must not stick.
    func testConfigureAppliesTheResolvedValue() {
        let expected = AgentSandbox.resolveEnabled(requested: false)
        let before = AgentSandbox.shared.isEnabled
        AgentSandbox.shared.configure(enabled: false,
                                      baseImage: ServerOptions.SandboxConfig().baseImage)
        XCTAssertEqual(AgentSandbox.shared.isEnabled, expected)
        // Restore whatever the suite started with (other tests read the singleton).
        AgentSandbox.shared.configure(enabled: before,
                                      baseImage: ServerOptions.SandboxConfig().baseImage)
    }
}
