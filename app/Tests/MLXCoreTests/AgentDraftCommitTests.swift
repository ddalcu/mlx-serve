import XCTest
@testable import MLXCore

/// Creating or duplicating an agent overwrites the shared editing draft. The
/// commit that protects pending edits only ran on `onChange(of: selectedId)` —
/// and by the time that fires, the draft already holds the NEW agent, so
/// clicking "Create New Agent", a starter, or Duplicate while an edit was
/// pending silently discarded it. `AgentsWorkspaceModel.adopt` commits the
/// outgoing draft BEFORE overwriting; these pin that ordering.
@MainActor
final class AgentDraftCommitTests: XCTestCase {

    private func tempStore() throws -> AgentStore {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("agent-draft-tests-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return AgentStore(rootDir: root)
    }

    func testAdoptingANewAgentCommitsTheOutgoingDraftFirst() throws {
        let store = try tempStore()
        var a = Agent(name: "A", brief: "", systemPrompt: "original")
        store.add(a)
        let model = AgentsWorkspaceModel()
        model.selectedId = a.id
        a.systemPrompt = "edited, not yet saved"
        model.draft = a

        let b = Agent(name: "B", brief: "", systemPrompt: "")
        store.add(b)
        model.adopt(b, committingTo: store, defaultAgentId: nil)

        XCTAssertEqual(store.agent(id: a.id)?.systemPrompt, "edited, not yet saved",
                       "creating a new agent must not discard pending edits to the old one")
        XCTAssertEqual(model.selectedId, b.id)
        XCTAssertEqual(model.draft?.id, b.id)
    }

    func testAdoptWithNoOutgoingDraftJustSelects() throws {
        let store = try tempStore()
        let b = Agent(name: "B", brief: "", systemPrompt: "")
        store.add(b)
        let model = AgentsWorkspaceModel()
        model.adopt(b, committingTo: store, defaultAgentId: nil)
        XCTAssertEqual(model.selectedId, b.id)
        XCTAssertEqual(model.draft?.id, b.id)
    }

    /// The commit inside adopt is the same one Save uses — a colliding wake
    /// phrase is refused there, not silently written.
    func testAdoptStillRefusesACollidingWakePhrase() throws {
        let store = try tempStore()
        var taken = Agent(name: "Taken", brief: "", systemPrompt: "")
        taken.wakePhrase = "hey chef"
        store.add(taken)
        var a = Agent(name: "A", brief: "", systemPrompt: "")
        store.add(a)
        let model = AgentsWorkspaceModel()
        model.selectedId = a.id
        a.wakePhrase = "hey chef"
        model.draft = a

        let b = Agent(name: "B", brief: "", systemPrompt: "")
        store.add(b)
        model.adopt(b, committingTo: store, defaultAgentId: nil)

        XCTAssertNil(store.agent(id: a.id)?.wakePhrase,
                     "a colliding phrase is dropped at commit, the same rule as Save")
        XCTAssertNotNil(model.alert, "…and the refusal is said out loud, not silent")
    }
}
