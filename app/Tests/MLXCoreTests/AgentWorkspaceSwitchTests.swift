import XCTest
@testable import MLXCore

/// An agent carries its own working directory, so switching agents can move the
/// folder out from under a LIVE sandbox CLI session. That has to behave like the
/// EXPLICIT Settings pick (remount, restarting the pinned sessions in place),
/// not the implicit chat-tab pick that gets declined — and it must never touch
/// the user's global default workspace, which would retarget every other
/// session.
final class AgentWorkspaceSwitchTests: XCTestCase {

    func testNoWorkspaceChangeDoesNothing() {
        XCTAssertEqual(AgentWorkspaceSwitch.decide(from: "/a", to: "/a"), .nothing)
        XCTAssertEqual(AgentWorkspaceSwitch.decide(from: "/a", to: nil), .nothing,
                       "an agent with no folder of its own inherits — nothing to remount")
        XCTAssertEqual(AgentWorkspaceSwitch.decide(from: nil, to: nil), .nothing)
    }

    func testAChangedWorkspaceRemountsAndRestartsPinnedSessions() {
        XCTAssertEqual(AgentWorkspaceSwitch.decide(from: "/a", to: "/b"),
                       .remount(path: "/b", restartPinnedSessions: true))
        XCTAssertEqual(AgentWorkspaceSwitch.decide(from: nil, to: "/b"),
                       .remount(path: "/b", restartPinnedSessions: true))
    }

    func testTheSandboxAgreesThisIsTheRestartingPath() {
        // The decision above feeds `AgentSandbox.noteWorkspaceChanged(_:restartPinnedSessions:)`,
        // whose own rule table must turn that into a teardown-and-restart even
        // with a live pinned session (the pin only declines IMPLICIT switches).
        guard case .remount(let path, let restart) =
                AgentWorkspaceSwitch.decide(from: "/w", to: "/elsewhere") else {
            return XCTFail("expected a remount")
        }
        XCTAssertEqual(AgentSandbox.workspaceChangeAction(
            guestAlive: true, sharedRoot: "/w", newWorkspace: path,
            pinnedLabels: ["pi"], restartPinnedSessions: restart),
            .teardownRestartingSessions)
    }

    func testAnAgentSwitchNeverChangesTheGlobalDefaultWorkspace() throws {
        // Source audit: the global default is a user SETTING. An agent switch
        // that called `setDefaultAgentWorkspace` would retarget every session
        // still on the old default — a side effect an agent pick has no business
        // having, and one nothing in the UI would show.
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for file in ["Sources/MLXServe/Views/AgentsWindow.swift",
                     "Sources/MLXServe/Services/AgentWorkspaceSwitch.swift"] {
            let text = try String(contentsOf: root.appendingPathComponent(file), encoding: .utf8)
            XCTAssertFalse(text.contains("setDefaultAgentWorkspace"),
                           "\(file) must not write the global default workspace")
            XCTAssertFalse(text.contains("setDefaultWorkingDirectory"),
                           "\(file) must not write the global default workspace")
        }
    }
}
