import XCTest
@testable import MLXCore

/// The Tools switch through the REAL turn engine, against a scripted
/// OpenAI-compatible server. Opt-in (`MLX_SERVE_LIVE_TOOLS_GATE=1`) and only
/// under a redirected home (`CFFIXED_USER_HOME`), because `AppState` reads and
/// writes `~/.mlx-serve` and the user defaults. Run the built bundle directly
/// (SwiftPM itself stalls under a redirected home):
/// `swift build --build-tests && MLX_SERVE_LIVE_TOOLS_GATE=1 CFFIXED_USER_HOME=$(mktemp -d)
///  xcrun xctest -XCTest MLXCoreTests.ToolsToggleLiveTests .build/debug/MLXCorePackageTests.xctest`
@MainActor
final class ToolsToggleLiveTests: XCTestCase {

    private var server: Process?
    private var requestLog = ""

    override func setUp() async throws {
        guard ProcessInfo.processInfo.environment["MLX_SERVE_LIVE_TOOLS_GATE"] == "1" else {
            throw XCTSkip("set MLX_SERVE_LIVE_TOOLS_GATE=1 and CFFIXED_USER_HOME to run")
        }
        guard let pw = getpwuid(getuid()),
              NSHomeDirectory() != String(cString: pw.pointee.pw_dir) else {
            throw XCTSkip("needs CFFIXED_USER_HOME: AppState must not touch the real home")
        }
        UserDefaults.standard.set(false, forKey: "autoStartServer")
        _ = NSApplication.shared   // AppState's launch wiring reads NSApp
    }

    override func tearDown() async throws {
        server?.terminate()
        server = nil
    }

    // MARK: - Harness

    /// Replies in order: `["tool": name, "args": [...], "delay": s]` or `["content": text]`.
    private func startServer(script: [[String: Any]]) throws -> UInt16 {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("fake-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let scriptPath = (dir as NSString).appendingPathComponent("script.json")
        requestLog = (dir as NSString).appendingPathComponent("requests.jsonl")
        try JSONSerialization.data(withJSONObject: script).write(to: URL(fileURLWithPath: scriptPath))

        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = ["python3", "-u", "-c", Self.fakeServer, scriptPath, requestLog]
        let out = Pipe()
        p.standardOutput = out
        try p.run()
        server = p
        let line = String(data: out.fileHandleForReading.availableData, encoding: .utf8) ?? ""
        return try XCTUnwrap(UInt16(line.trimmingCharacters(in: .whitespacesAndNewlines)), "fake server port: \(line)")
    }

    /// Tool names each request offered the model, in request order.
    private func offeredTools() throws -> [[String]] {
        let text = try String(contentsOfFile: requestLog, encoding: .utf8)
        return text.split(separator: "\n").map { line in
            let body = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] ?? [:]
            let tools = body["tools"] as? [[String: Any]] ?? []
            return tools.compactMap { ($0["function"] as? [String: Any])?["name"] as? String }
        }
    }

    private func makeSession(appState: AppState, dir: String) -> UUID {
        var s = ChatSession(title: "tools gate")
        s.workingDirectory = dir
        appState.chatSessions.append(s)
        return s.id
    }

    private func config(tools: Bool, mcp: Bool, dir: String) -> ChatTurnEngine.TurnConfig {
        ChatTurnEngine.TurnConfig.from(AgentResolution.resolve(agent: nil, defaults: AppDefaultsSnapshot(
            toolsEnabled: tools, mcpEnabled: mcp, thinkingEnabled: false,
            autoApprove: false, workingDirectory: dir)))
    }

    private func waitUntil(_ timeout: TimeInterval, _ what: String,
                           _ condition: () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(timeout)
        while !condition() {
            if Date() > deadline { XCTFail("timed out waiting for \(what)"); return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    private func toolOutputs(_ appState: AppState, _ id: UUID) -> [String] {
        let msgs = appState.chatSessions.first { $0.id == id }?.messages ?? []
        return msgs.filter { $0.toolCallId != nil }.map(\.content)
    }

    private func workspace() throws -> (dir: String, victim: String) {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("ws-\(UUID().uuidString)")
        let victim = (dir as NSString).appendingPathComponent("test-folder")
        try FileManager.default.createDirectory(atPath: victim, withIntermediateDirectories: true)
        return (dir, victim)
    }

    // MARK: - Cases

    func testToolsOffWithMcpOnRefusesTheShellCall() async throws {
        let (dir, victim) = try workspace()
        let port = try startServer(script: [
            ["tool": "shell", "args": ["command": "rm -rf test-folder"]],
            ["content": "done"],
        ])
        let appState = AppState()
        appState.server.port = port
        appState.server.status = .running   // the fake answers in its place
        let id = makeSession(appState: appState, dir: dir)

        appState.chatEngine.runTurn(sessionId: id, userText: "delete test-folder",
                                    images: nil, audio: nil,
                                    config: config(tools: false, mcp: true, dir: dir),
                                    approval: { _ in true })
        try await waitUntil(20, "the turn to end") { appState.chatEngine.activeTurnSessionIds.isEmpty }

        XCTAssertTrue(FileManager.default.fileExists(atPath: victim), "the folder must survive")
        XCTAssertTrue(toolOutputs(appState, id).contains { $0.contains("was not run") },
                      "\(toolOutputs(appState, id))")
        let offered = try offeredTools()
        XCTAssertEqual(offered.count, 2)
        XCTAssertFalse(offered.joined().contains("shell"), "no request offered shell: \(offered)")
    }

    func testToolsOffMidTurnRefusesTheNextCallAndTheResumedTurn() async throws {
        let (dir, victim) = try workspace()
        let first = (dir as NSString).appendingPathComponent("first.txt")
        let port = try startServer(script: [
            ["tool": "shell", "args": ["command": "touch first.txt"]],
            ["tool": "shell", "args": ["command": "rm -rf test-folder"], "delay": 2],
            ["content": "ok", "delay": 2],
            ["content": "resumed"],
        ])
        let appState = AppState()
        appState.server.port = port
        appState.server.status = .running   // the fake answers in its place
        let id = makeSession(appState: appState, dir: dir)
        let engine = appState.chatEngine

        engine.runTurn(sessionId: id, userText: "make a file then delete test-folder",
                       images: nil, audio: nil,
                       config: config(tools: true, mcp: false, dir: dir),
                       approval: { _ in true })
        try await waitUntil(20, "the first shell call") { FileManager.default.fileExists(atPath: first) }

        // The user flips Tools off while round 2 streams…
        engine.revokeTools(sessionId: id)
        try await waitUntil(20, "the refused call") {
            toolOutputs(appState, id).contains { $0.contains("was not run") }
        }
        // …and types a note while the final answer streams, so it resumes a new turn.
        engine.setSteeringNote("anything else?", for: id)

        try await waitUntil(30, "the resumed turn") {
            engine.activeTurnSessionIds.isEmpty
                && (appState.chatSessions.first { $0.id == id }?.messages.contains { $0.content == "resumed" } ?? false)
        }

        XCTAssertTrue(FileManager.default.fileExists(atPath: victim), "the folder must survive")
        XCTAssertTrue(toolOutputs(appState, id).contains { $0.contains("was not run") },
                      "\(toolOutputs(appState, id))")
        let offered = try offeredTools()
        guard offered.count == 4 else { return XCTFail("expected 4 requests: \(offered)") }
        XCTAssertTrue(offered[0].contains("shell"), "Tools were on for round 1")
        XCTAssertFalse(offered[2].contains("shell"), "round 3 offers no built-ins: \(offered[2])")
        XCTAssertTrue(offered[3].isEmpty, "the resumed turn runs with Tools off: \(offered[3])")
    }

    // MARK: - Fake server

    private static let fakeServer = #"""
import json, sys, time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
script = json.load(open(sys.argv[1])); log = sys.argv[2]; state = {"n": 0}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(b'{"object":"list","data":[]}')
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if not self.path.endswith("/chat/completions"):
            self.send_response(404); self.end_headers(); return
        with open(log, "a") as f: f.write(json.dumps(json.loads(body)) + "\n")
        step = script[min(state["n"], len(script) - 1)]; state["n"] += 1
        time.sleep(step.get("delay", 0))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        def send(obj): self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n"); self.wfile.flush()
        if "tool" in step:
            send({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_%d" % state["n"],
                  "function": {"name": step["tool"], "arguments": json.dumps(step["args"])}}]}}]})
            send({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        else:
            send({"choices": [{"index": 0, "delta": {"content": step["content"]}}]})
            send({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        send({"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()

srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
print(srv.server_address[1], flush=True)
srv.serve_forever()
"""#
}
