import XCTest
@testable import MLXCore

/// `residentChatModel` falls back to `allModels.first { loaded }` whenever
/// `modelInfo` is nil (ServerManager.swift) — so a `stop()` that cleared
/// `modelInfo` but left `allModels` untouched still let the pill name the
/// model that was resident BEFORE the stop, because the stale entry still
/// read `loaded: true`. A picker switch made while stopped then looked like
/// it might launch either the old or the newly picked model. `stop()` must
/// clear `allModels` too, since its only source is the local server's own
/// `/v1/models` — once the process is down, that snapshot is stale by
/// construction.
@MainActor
final class ServerManagerStopTests: XCTestCase {

    private func residentModel(_ name: String) -> ModelInfo {
        ModelInfo(name: name, quantBits: 4, layers: 0, hiddenSize: 0,
                  vocabSize: 0, contextLength: 0, modelMaxTokens: 0,
                  capabilities: ["chat"], loaded: true)
    }

    func testStopClearsAllModelsSoTheResidentModelIsNotStale() {
        let server = ServerManager()
        server.status = .running
        server.modelInfo = residentModel("old-model")
        server.allModels = [residentModel("old-model")]

        server.stop()

        XCTAssertTrue(server.allModels.isEmpty,
                      "a stale allModels lets residentChatModel keep naming the model resident before stop")
    }

    func testStopMakesChatModelInfoNilRatherThanTheStaleResident() {
        let server = ServerManager()
        server.status = .running
        server.modelInfo = residentModel("old-model")
        server.allModels = [residentModel("old-model")]

        server.stop()

        XCTAssertNil(server.chatModelInfo,
                     "a stopped server has nothing resident to answer chat, so the pill must not fall back to the old model")
    }
}
