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
}
