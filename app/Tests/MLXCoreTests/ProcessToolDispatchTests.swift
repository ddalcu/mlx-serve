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
}
