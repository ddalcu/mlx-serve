import XCTest
@testable import MLXCore

final class AgentPromptTests: XCTestCase {
    // The agent must avoid interactive scaffolders (`npx sv create`, etc.) — in
    // the agent's TTY-less shell they fail/loop. The base prompt must steer it
    // toward non-interactive flags or manual setup.
    func testSystemPromptHasScaffoldingGuidance() {
        let p = AgentPrompt.defaultPromptFile
        XCTAssertTrue(p.contains("Scaffolding"), "base prompt is missing a scaffolding section")
        XCTAssertTrue(p.lowercased().contains("interactive"),
                      "base prompt should warn about interactive commands")
        XCTAssertTrue(p.contains("npm init -y") || p.lowercased().contains("non-interactive"),
                      "base prompt should steer toward non-interactive setup")
    }

    // `system-prompt.md` is now the single editable prompt (seeded with the
    // built-in default), not a separate "additive customizations" stub. These
    // pin the pure resolver that backs `systemPrompt` without touching the
    // user's real ~/.mlx-serve file.
    func testResolvePromptFallsBackToDefaultWhenEmpty() {
        XCTAssertEqual(AgentPrompt.resolvePrompt(fileContent: ""), AgentPrompt.defaultPromptFile)
        XCTAssertEqual(AgentPrompt.resolvePrompt(fileContent: "   \n\t "), AgentPrompt.defaultPromptFile)
    }

    func testResolvePromptMigratesLegacyStubToDefault() {
        // A pre-v26.6.11 install seeded this exact placeholder; under the old
        // append design it was harmless, but as the whole prompt it would strip
        // the agent of its tool/workspace guidance. Must resolve to the default.
        let legacyStub = """
            # Custom Instructions
            Add your project-specific rules, preferences, or personality tweaks here.
            These are appended to the base system prompt.
            """
        XCTAssertEqual(AgentPrompt.resolvePrompt(fileContent: legacyStub),
                       AgentPrompt.defaultPromptFile)
    }

    func testResolvePromptKeepsUserPromptVerbatim() {
        let custom = "# My Prompt\nYou are a terse assistant. No preamble."
        XCTAssertEqual(AgentPrompt.resolvePrompt(fileContent: custom), custom)
        XCTAssertEqual(AgentPrompt.resolvePrompt(fileContent: "\n\n\(custom)\n  "), custom,
                       "surrounding whitespace is trimmed, content preserved")
    }

    // Backgrounding guidance moved from brittle `&`/`kill %1` shell tricks to the
    // managed run_in_background flag + readProcessOutput/killProcess tools.
    func testPromptDropsBrittleBackgroundingGuidance() {
        let p = AgentPrompt.defaultPromptFile
        XCTAssertFalse(p.contains("kill %1"), "brittle `kill %1` guidance must be gone")
        XCTAssertFalse(p.contains("node server.js &"), "brittle `&` backgrounding example must be gone")
        XCTAssertFalse(p.contains("npm run dev &"), "brittle `&` backgrounding example must be gone")
    }

    func testPromptHasRunInBackgroundGuidance() {
        let p = AgentPrompt.defaultPromptFile
        XCTAssertTrue(p.contains("run_in_background"), "prompt should steer toward run_in_background")
        XCTAssertTrue(p.contains("killProcess"), "prompt should mention killProcess")
        XCTAssertTrue(p.contains("readProcessOutput"), "prompt should mention readProcessOutput")
    }
}
