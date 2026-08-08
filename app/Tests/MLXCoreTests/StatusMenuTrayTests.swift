import XCTest
@testable import MLXCore

/// Pins the tray panel's empty-state condition. Live bug: the "no models
/// found" message checked `appState.localModels.isEmpty` while the Picker
/// right below it populated from `localModels.filter { $0.isChatPickable }`
/// — a Mac with only media/drafter downloads has a non-empty `localModels`
/// but nothing the picker can offer, so the message never showed and the
/// user saw a broken empty dropdown instead.
final class StatusMenuTrayTests: XCTestCase {

    private func local(_ name: String, path: String, kind: ModelKind = .base, type: String = "gemma4") -> LocalModel {
        LocalModel(id: path, name: name, path: path, sizeFormatted: "4 GB", modelType: type, source: .mlxServe, kind: kind)
    }

    func testNoModelsAtAllShowsTheMessage() {
        XCTAssertTrue(trayHasNoUsableModels([]))
    }

    /// The regression case: a non-empty list that contains only models the
    /// picker can't offer (media / drafter checkpoints) must still count as
    /// "no usable models".
    func testOnlyNonChatPickableModelsStillShowsTheMessage() {
        let media = local("flux", path: "/m/flux", type: "flux2")
        let drafter = local("gemma-4-e4b-it-assistant-bf16", path: "/m/draft", kind: .drafter, type: "gemma4_assistant")
        XCTAssertTrue(trayHasNoUsableModels([media, drafter]))
    }

    func testARealChatModelHidesTheMessage() {
        let media = local("flux", path: "/m/flux", type: "flux2")
        let chatModel = local("gemma-4-e4b-it-4bit", path: "/m/gemma", type: "gemma4")
        XCTAssertFalse(trayHasNoUsableModels([media, chatModel]))
    }

    // MARK: - Source audit

    /// The Settings gear must render OUTSIDE the empty-state if/else. Live
    /// bug: the gear lived in the populated branch only, so a Mac with no
    /// usable models (fresh install, deleted models dir) had NO route to
    /// Settings from the tray at all.
    func testSettingsGearRendersOutsideTheEmptyStateBranch() throws {
        let source = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe/Views/StatusMenuView.swift")
        let lines = try String(contentsOf: source, encoding: .utf8)
            .components(separatedBy: "\n")
        // The branch reads `if hasNoUsableModels {` — a computed property over
        // the pure `trayHasNoUsableModels(_:lanChatModelCount:)` above, since
        // the redesign needs the same answer in two places.
        let ifIdx = try XCTUnwrap(
            lines.firstIndex { $0.contains("if hasNoUsableModels {") })
        let gearIdx = try XCTUnwrap(
            lines.firstIndex { $0.contains("openSettings()") })
        // Gear before the branch is trivially unconditional; after it, the
        // if/else must have fully closed first (brace depth back to zero).
        guard gearIdx > ifIdx else { return }
        var depth = 0
        for line in lines[ifIdx..<gearIdx] {
            for ch in line where ch == "{" || ch == "}" {
                depth += ch == "{" ? 1 : -1
            }
        }
        XCTAssertEqual(depth, 0, """
            The Settings gear button must be rendered unconditionally, not \
            inside the trayHasNoUsableModels else-branch — with no models \
            downloaded the gear is the only way to reach Settings.
            """)
    }
}
