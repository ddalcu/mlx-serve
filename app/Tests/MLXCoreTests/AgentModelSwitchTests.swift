import XCTest
@testable import MLXCore

/// Picking an agent whose model isn't the one loaded has four outcomes, and
/// exactly one of them is a multi-GB surprise the app must never spring on the
/// user. The decision is pure so the two callers (Agents window, tray picker)
/// can't disagree.
final class AgentModelSwitchTests: XCTestCase {

    private func local(_ path: String) -> LocalModel {
        LocalModel(id: path, name: (path as NSString).lastPathComponent, path: path,
                   sizeFormatted: "4 GB", modelType: "qwen3", source: .mlxServe, kind: .base)
    }

    private let downloaded = ["/models/qwen", "/models/gemma"]

    private func decide(_ agent: Agent?, selected: String = "/models/qwen",
                        lan: [String] = ["big-model@studio"]) -> AgentModelSwitch.Decision {
        AgentModelSwitch.decide(modelPath: agent?.modelPath,
                                selectedModelPath: selected,
                                downloadedPaths: downloaded,
                                lanModelIds: lan)
    }

    func testCurrentMeansNoChange() {
        var a = Agent(name: "A", brief: "", systemPrompt: "p")
        a.modelPath = nil
        XCTAssertEqual(decide(a), .noChange, "nil model = \"Current\" — never touches the server")
        XCTAssertEqual(decide(nil), .noChange, "no agent at all is the same story")
    }

    func testAlreadySelectedIsNoChange() {
        var a = Agent(name: "A", brief: "", systemPrompt: "p")
        a.modelPath = "/models/qwen"
        XCTAssertEqual(decide(a), .noChange, "no reload when it's already the selected model")
    }

    func testADownloadedModelLoads() {
        var a = Agent(name: "A", brief: "", systemPrompt: "p")
        a.modelPath = "/models/gemma"
        XCTAssertEqual(decide(a), .load(path: "/models/gemma"))
    }

    func testALanIdPassesThroughWithoutALocalLoad() {
        var a = Agent(name: "A", brief: "", systemPrompt: "p")
        a.modelPath = "big-model@studio"
        XCTAssertEqual(decide(a), .lan(id: "big-model@studio"),
                       "LAN ids are the server's business — no local load, no download check")
    }

    func testAnUnknownLanPeerIsUnavailableRatherThanTreatedAsAPath() {
        var a = Agent(name: "A", brief: "", systemPrompt: "p")
        a.modelPath = "gone@laptop"
        guard case .unavailable(let reason) = decide(a, lan: []) else {
            return XCTFail("an offline peer can't answer — say so")
        }
        XCTAssertTrue(reason.lowercased().contains("network") || reason.lowercased().contains("peer"),
                      "reason should point at the peer: \(reason)")
    }

    func testAModelThatIsNotOnDiskIsUnavailableAndNeverImplicitlyDownloaded() {
        var a = Agent(name: "Chef", brief: "", systemPrompt: "p")
        a.modelPath = "/models/not-here"
        guard case .needsDownload(let path) = decide(a) else {
            return XCTFail("a missing model must ask for a download, never start one")
        }
        XCTAssertEqual(path, "/models/not-here")
    }

    func testSpokenDeclineNamesTheAgentAndTheReason() {
        // A voice switch to an agent whose model isn't downloaded must be
        // declined OUT LOUD — silently answering as whoever was active is the
        // failure the user can't see.
        let line = AgentModelSwitch.spokenDecline(agentName: "Chef",
                                                 decision: .needsDownload(path: "/models/x"))
        XCTAssertNotNil(line)
        XCTAssertTrue(line!.contains("Chef"), line ?? "")
        XCTAssertTrue(line!.lowercased().contains("download"), line ?? "")

        XCTAssertNil(AgentModelSwitch.spokenDecline(agentName: "Chef", decision: .noChange))
        XCTAssertNil(AgentModelSwitch.spokenDecline(agentName: "Chef",
                                                    decision: .load(path: "/models/gemma")))
        XCTAssertNotNil(AgentModelSwitch.spokenDecline(agentName: "Chef",
                                                       decision: .unavailable(reason: "peer offline")))
    }

    func testPickerAvailabilityMirrorsTheDecision() {
        var ok = Agent(name: "OK", brief: "", systemPrompt: "p")
        ok.modelPath = "/models/gemma"
        var missing = Agent(name: "Missing", brief: "", systemPrompt: "p")
        missing.modelPath = "/models/not-here"

        XCTAssertTrue(AgentModelSwitch.isSelectable(decide(ok)))
        XCTAssertFalse(AgentModelSwitch.isSelectable(decide(missing)),
                       "greyed in the picker, with a Download button — not silently selectable")
        XCTAssertTrue(AgentModelSwitch.isSelectable(decide(nil)))
    }

    func testDisplayNameFallsBackToTheLastPathComponent() {
        XCTAssertEqual(AgentModelSwitch.displayName(for: nil, localModels: [local("/models/qwen")]),
                       "Current")
        XCTAssertEqual(AgentModelSwitch.displayName(for: "/models/qwen",
                                                    localModels: [local("/models/qwen")]),
                       "qwen")
        XCTAssertEqual(AgentModelSwitch.displayName(for: "/models/gone", localModels: []),
                       "gone", "a deleted model still reads as its own name, not a blank")
        XCTAssertEqual(AgentModelSwitch.displayName(for: "big@studio", localModels: []),
                       "big@studio", "LAN ids are shown verbatim")
    }
}
