import XCTest
@testable import MLXCore

/// The agent-facing process tools: listProcesses / readProcessOutput /
/// killProcess, plus their schema-validation + name-resolution wiring.
@MainActor
final class ProcessToolDispatchTests: XCTestCase {

    private func waitUntil(_ timeout: Double = 3.0, _ cond: () -> Bool) async {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if cond() { return }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    func testListProcessesListsRunning() {
        let reg = ProcessRegistry(); defer { reg.killAll() }
        let sid = UUID()
        let p = reg.start(command: "sleep 5", workingDirectory: nil, sessionId: sid)
        let out = AgentEngine.processToolOutput(.listProcesses, arguments: [:], registry: reg, sessionId: sid)
        XCTAssertTrue(out.contains(p.handle), out)
        XCTAssertTrue(out.contains("running"), out)
    }

    func testListProcessesEmpty() {
        let reg = ProcessRegistry()
        let out = AgentEngine.processToolOutput(.listProcesses, arguments: [:], registry: reg, sessionId: UUID())
        XCTAssertTrue(out.contains("No background processes"), out)
    }

    func testReadProcessOutputReturnsCaptured() async {
        let reg = ProcessRegistry(); defer { reg.killAll() }
        let p = reg.start(command: "echo hi-there; exec sleep 5", workingDirectory: nil, sessionId: nil)
        await waitUntil { p.output.snapshot().contains("hi-there") }
        let out = AgentEngine.processToolOutput(.readProcessOutput, arguments: ["handle": p.handle],
                                                registry: reg, sessionId: nil)
        XCTAssertTrue(out.contains("hi-there"), out)
    }

    func testReadProcessOutputUnknownHandleErrors() {
        let reg = ProcessRegistry()
        let out = AgentEngine.processToolOutput(.readProcessOutput, arguments: ["handle": "bgX"],
                                                registry: reg, sessionId: nil)
        XCTAssertTrue(out.hasPrefix("Error:"), out)
        XCTAssertTrue(out.contains("bgX"), out)
    }

    func testKillProcessKills() {
        let reg = ProcessRegistry(); defer { reg.killAll() }
        let p = reg.start(command: "sleep 30", workingDirectory: nil, sessionId: nil)
        let out = AgentEngine.processToolOutput(.killProcess, arguments: ["handle": p.handle],
                                                registry: reg, sessionId: nil)
        XCTAssertTrue(out.contains("Killed"), out)
        XCTAssertFalse(reg.isAlive(handle: p.handle))
    }

    func testKillProcessUnknownHandleErrors() {
        let reg = ProcessRegistry()
        let out = AgentEngine.processToolOutput(.killProcess, arguments: ["handle": "bgX"],
                                                registry: reg, sessionId: nil)
        XCTAssertTrue(out.hasPrefix("Error:"), out)
    }

    func testProcessToolsWithoutRegistryAreGraceful() {
        let out = AgentEngine.processToolOutput(.listProcesses, arguments: [:], registry: nil, sessionId: nil)
        XCTAssertTrue(out.hasPrefix("Error:"), out)
    }

    // MARK: - Schema validation + name resolution

    func testMissingRequiredParamsFlagsAbsentHandle() {
        XCTAssertEqual(AgentEngine.missingRequiredParams(for: "killProcess", arguments: [:]), ["handle"])
        XCTAssertEqual(AgentEngine.missingRequiredParams(for: "readProcessOutput", arguments: [:]), ["handle"])
        XCTAssertEqual(AgentEngine.missingRequiredParams(for: "listProcesses", arguments: [:]), [])
    }

    func testCanonicalToolNameResolvesNewTools() {
        XCTAssertEqual(AgentEngine.canonicalToolName("killProcess"), "killProcess")
        XCTAssertEqual(AgentEngine.canonicalToolName(" functions.listProcesses "), "listProcesses")
        XCTAssertEqual(AgentEngine.canonicalToolName("readProcessOutput:"), "readProcessOutput")
        XCTAssertNotNil(AgentToolKind(rawValue: "readProcessOutput"))
        XCTAssertNotNil(AgentToolKind(rawValue: "killProcess"))
        XCTAssertNotNil(AgentToolKind(rawValue: "listProcesses"))
    }

    // MARK: - Salvaging a process tool a weak model emitted as a shell command

    func testFirstBgHandle() {
        XCTAssertEqual(AgentEngine.firstBgHandle(in: "killProcess{handle: \"bg1\"}"), "bg1")
        XCTAssertEqual(AgentEngine.firstBgHandle(in: "stop bg42 now"), "bg42")
        XCTAssertNil(AgentEngine.firstBgHandle(in: "no handle here"))
        XCTAssertNil(AgentEngine.firstBgHandle(in: "bg")) // no digits
    }

    func testProcessToolFromShellCommand() {
        XCTAssertEqual(AgentEngine.processToolFromShellCommand("killProcess{handle: \"bg1\"}")?.tool, .killProcess)
        XCTAssertEqual(AgentEngine.processToolFromShellCommand("killProcess{handle: \"bg1\"}")?.handle, "bg1")
        XCTAssertEqual(AgentEngine.processToolFromShellCommand("killProcess bg2")?.handle, "bg2")
        XCTAssertEqual(AgentEngine.processToolFromShellCommand("readProcessOutput {\"handle\":\"bg3\"}")?.tool, .readProcessOutput)
        XCTAssertEqual(AgentEngine.processToolFromShellCommand("listProcesses")?.tool, .listProcesses)
        XCTAssertNil(AgentEngine.processToolFromShellCommand("listProcesses")?.handle)
        // Real shell commands pass straight through.
        XCTAssertNil(AgentEngine.processToolFromShellCommand("ls -la /tmp"))
        XCTAssertNil(AgentEngine.processToolFromShellCommand("echo killProcess"), "tool name must be the FIRST token")
    }

    /// End-to-end: a `shell` call whose command is really `killProcess{…}` must
    /// re-route to the tool and actually kill the process.
    func testShellMisroutedKillProcessIsSalvaged() async {
        let reg = ProcessRegistry(); defer { reg.killAll() }
        let p = reg.start(command: "sleep 60", workingDirectory: nil, sessionId: nil)
        XCTAssertTrue(reg.isAlive(handle: p.handle))
        let tc = APIClient.ToolCall(id: "1", name: "shell",
                                    arguments: ["command": "killProcess{handle: \"\(p.handle)\"}"],
                                    rawArguments: "")
        var wd: String? = nil
        let r = await AgentEngine.executeToolCall(
            tc, workingDirectory: &wd, repetition: AgentEngine.RepetitionTracker(),
            iteration: 0, agentMemory: AgentMemory(),
            processRegistry: reg, sessionId: nil)
        XCTAssertTrue(r.output.contains("Killed"), r.output)
        XCTAssertFalse(reg.isAlive(handle: p.handle), "misrouted killProcess must still stop the process")
    }
}
