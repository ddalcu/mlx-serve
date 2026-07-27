import XCTest
@testable import MLXCore

/// The store is one JSON file held whole in memory (same shape as the iPhone's).
/// The interesting property is TOLERANT DECODE in both directions: a build that
/// adds a field must still read an older file, and an older build must still
/// read a newer one — the agents file is the kind of thing a user carries
/// across upgrades and (eventually) between the phone and the Mac.
@MainActor
final class AgentStoreTests: XCTestCase {

    private func tempRoot() throws -> URL {
        let d = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("agents-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }

    private func write(_ json: String, to root: URL) throws {
        try json.data(using: .utf8)!.write(to: root.appendingPathComponent("index.json"))
    }

    func testRoundTripsEveryField() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        var a = Agent(name: "Chef", brief: "cooking help", systemPrompt: "You are a chef.")
        // Whole seconds on purpose: the file is plain ISO-8601 (no fractional
        // seconds) because that's what the iPhone app's decoder accepts, and
        // keeping the two files mutually readable is worth more than sub-second
        // precision on a field only used to sort a list.
        a.createdAt = Date(timeIntervalSince1970: 1_700_000_000)
        a.symbol = "fork.knife"
        a.wakePhrase = "hey chef"
        a.voice = .kokoro("af_bella,af_sky")
        a.voiceID = "af_bella"
        a.modelPath = "/models/qwen"
        a.workingDirectory = "/tmp/recipes"
        a.capabilities = AgentCapabilities(tools: true, mcp: true, web: false,
                                           advancedTools: [.shell, .readFile])
        a.autoApproveTools = true
        a.enableThinking = true
        a.temperature = 0.3
        a.maxTokens = 777

        let store = AgentStore(rootDir: root)
        store.add(a)

        let reloaded = AgentStore(rootDir: root)
        XCTAssertEqual(reloaded.agents.count, 1)
        XCTAssertEqual(reloaded.agents.first, a, "every field survives the round trip")
    }

    func testUpdateAndDelete() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AgentStore(rootDir: root)
        var a = Agent(name: "One", brief: "b", systemPrompt: "p")
        store.add(a)
        a.name = "Two"
        store.update(a)
        XCTAssertEqual(AgentStore(rootDir: root).agents.first?.name, "Two")
        store.delete(id: a.id)
        XCTAssertTrue(AgentStore(rootDir: root).agents.isEmpty)
    }

    func testSortedNewestFirst() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AgentStore(rootDir: root)
        var old = Agent(name: "Old", brief: "b", systemPrompt: "p")
        old.createdAt = Date(timeIntervalSince1970: 1000)
        var new = Agent(name: "New", brief: "b", systemPrompt: "p")
        new.createdAt = Date(timeIntervalSince1970: 2000)
        store.add(old)
        store.add(new)
        XCTAssertEqual(store.sortedAgents.map(\.name), ["New", "Old"])
    }

    // MARK: - Tolerant decode

    func testDecodesAFileMissingEveryMacOnlyKey() throws {
        // Exactly what the iPhone writes: the shared subset and nothing else.
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try write("""
        [{"id":"7B8E4A9C-0000-4000-8000-000000000001","name":"Trip Planner",
          "brief":"plan trips","systemPrompt":"You are a travel planner.",
          "symbol":"airplane","voiceID":"af_heart","canSearchWeb":true,
          "createdAt":"2026-01-02T03:04:05Z"}]
        """, to: root)

        let store = AgentStore(rootDir: root)
        let a = try XCTUnwrap(store.agents.first)
        XCTAssertEqual(a.name, "Trip Planner")
        XCTAssertEqual(a.voiceID, "af_heart")
        XCTAssertNil(a.voice)
        XCTAssertNil(a.modelPath)
        XCTAssertNil(a.workingDirectory)
        XCTAssertNil(a.temperature)
        XCTAssertNil(a.maxTokens)
        XCTAssertNil(a.wakePhrase)
        XCTAssertFalse(a.isBuiltIn)
        XCTAssertFalse(a.capabilities.tools, "absent capabilities decode to the defaults")
        XCTAssertTrue(a.capabilities.web)
        XCTAssertNil(a.capabilities.advancedTools)
    }

    func testIgnoresUnknownFutureKeys() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try write("""
        [{"id":"7B8E4A9C-0000-4000-8000-000000000002","name":"Future",
          "brief":"b","systemPrompt":"p","createdAt":"2026-01-02T03:04:05Z",
          "someFieldFromALaterBuild":{"nested":[1,2,3]},
          "capabilities":{"tools":true,"mcp":false,"web":false,"unknownCap":9}}]
        """, to: root)

        let a = try XCTUnwrap(AgentStore(rootDir: root).agents.first)
        XCTAssertEqual(a.name, "Future")
        XCTAssertTrue(a.capabilities.tools)
        XCTAssertFalse(a.capabilities.web)
    }

    func testUnknownVoiceEngineDegradesToNoVoiceRatherThanLosingTheAgent() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try write("""
        [{"id":"7B8E4A9C-0000-4000-8000-000000000003","name":"Later","brief":"b",
          "systemPrompt":"p","createdAt":"2026-01-02T03:04:05Z",
          "voice":{"engine":"holo","value":"x"}}]
        """, to: root)

        let a = try XCTUnwrap(AgentStore(rootDir: root).agents.first)
        XCTAssertEqual(a.name, "Later")
        XCTAssertNil(a.voice, "an engine this build doesn't know falls back to the app voice")
    }

    func testAMalformedFileLeavesAnEmptyStoreInsteadOfCrashing() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        try write("{ not an array", to: root)
        XCTAssertTrue(AgentStore(rootDir: root).agents.isEmpty)
    }

    func testMissingFileIsAnEmptyStore() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        XCTAssertTrue(AgentStore(rootDir: root).agents.isEmpty)
    }

    // MARK: - Starter agents

    func testStartersAreReadOnlyCodeConstantsAndNotPersisted() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AgentStore(rootDir: root)

        XCTAssertFalse(Agent.starters.isEmpty)
        XCTAssertTrue(Agent.starters.allSatisfy(\.isBuiltIn))
        XCTAssertTrue(Agent.starters.allSatisfy { !$0.systemPrompt.isEmpty })
        XCTAssertEqual(Set(Agent.starters.map(\.id)).count, Agent.starters.count, "stable, distinct ids")
        XCTAssertTrue(store.agents.isEmpty,
                      "starters are constants so improving them reaches existing users — never seeded to disk")
        XCTAssertEqual(store.allAgents.count, Agent.starters.count,
                       "they still show up in the list")
        XCTAssertNotNil(store.agent(id: Agent.starters[0].id),
                        "lookups must find a starter — turn sites only carry an id")
    }

    func testDuplicatingAStarterProducesAnEditableCopy() throws {
        let root = try tempRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = AgentStore(rootDir: root)
        let copy = store.duplicate(Agent.starters[0])
        XCTAssertFalse(copy.isBuiltIn)
        XCTAssertNotEqual(copy.id, Agent.starters[0].id)
        XCTAssertEqual(copy.systemPrompt, Agent.starters[0].systemPrompt)
        XCTAssertEqual(AgentStore(rootDir: root).agents.count, 1, "the copy is persisted")
    }
}
