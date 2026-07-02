import XCTest
@testable import MLXCore

final class AgentPromptTests: XCTestCase {
    // The output-budget note must reflect the EFFECTIVE per-response budget —
    // min(max_tokens, ~2/5 of context) — not a flat cap, so a small-RAM /
    // small-context machine gets an honest (smaller) number.
    func testOutputBudgetGuidanceIsContextAware() {
        // Small context caps below a high max_tokens cap: 4096 * 2/5 = 1638.
        let smallCtx = AgentPrompt.outputBudgetGuidance(maxTokens: 16384, contextLength: 4096)
        XCTAssertTrue(smallCtx.contains("1638"), "small context must cap the budget: \(smallCtx)")
        XCTAssertFalse(smallCtx.contains("16384"),
                       "must not advertise the unreachable 16384 cap: \(smallCtx)")

        // Auto (max_tokens <= 0) pegs purely to context: 131072 * 2/5 = 52428.
        let auto = AgentPrompt.outputBudgetGuidance(maxTokens: 0, contextLength: 131072)
        XCTAssertTrue(auto.contains("52428"), "Auto must peg to context: \(auto)")

        // An explicit cap below the context budget still wins.
        let capped = AgentPrompt.outputBudgetGuidance(maxTokens: 4096, contextLength: 131072)
        XCTAssertTrue(capped.contains("4096"), "explicit cap below context budget must win: \(capped)")

        // Always points at append-chunk recovery and warns the work is lost.
        XCTAssertTrue(smallCtx.lowercased().contains("append"), "must point at append chunking")
        XCTAssertTrue(smallCtx.contains("LOST") || smallCtx.lowercased().contains("cut off"),
                      "must warn the work is lost on overflow")
    }

    // The agent must avoid interactive scaffolders (`npx sv create`, etc.) — in
    // the agent's TTY-less shell they fail/loop. The base prompt must steer it
    // toward non-interactive flags or manual setup.
    func testSystemPromptHasScaffoldingGuidance() {
        let p = AgentPrompt.defaultPromptFile
        XCTAssertTrue(p.lowercased().contains("scaffold"), "base prompt is missing scaffolding guidance")
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

    // "Update System Prompt" menu item: enabled only when the on-disk prompt is a
    // real prompt that differs from the latest default.
    func testIsPromptOutdated() {
        // Missing / empty / legacy stub all resolve to the default → not outdated.
        XCTAssertFalse(AgentPrompt.isPromptOutdated(fileContent: nil))
        XCTAssertFalse(AgentPrompt.isPromptOutdated(fileContent: ""))
        XCTAssertFalse(AgentPrompt.isPromptOutdated(fileContent: "   \n  "))
        let legacyStub = "# Custom Instructions\nThese are appended to the base system prompt."
        XCTAssertFalse(AgentPrompt.isPromptOutdated(fileContent: legacyStub))
        // The current default itself → not outdated.
        XCTAssertFalse(AgentPrompt.isPromptOutdated(fileContent: AgentPrompt.defaultPromptFile))
        // A real, differing prompt (old default or a user customization) → outdated.
        XCTAssertTrue(AgentPrompt.isPromptOutdated(fileContent: "# My terse prompt\nNo preamble."))
        XCTAssertTrue(AgentPrompt.isPromptOutdated(fileContent: "You are an old version of the agent prompt."))
    }

    func testPromptBackupFileNameIsStampedAndDistinct() {
        let a = AgentPrompt.promptBackupFileName(stamp: "20260620-101500")
        XCTAssertEqual(a, "system-prompt.backup-20260620-101500.md")
        let b = AgentPrompt.promptBackupFileName(stamp: "20260620-101501")
        XCTAssertNotEqual(a, b, "different stamps must not collide")
    }

    // MARK: - Execution environment (sandbox-aware)

    // The BASE prompt file is user-editable and serves both environments, so it
    // must be OS-neutral; the per-request Execution environment section is what
    // tells the model where shell commands actually run. Without this split, a
    // macOS-flavored prompt sends `brew`/`open` into the Linux guest (and a
    // Linux-flavored one sends `apt-get` at the host).
    func testDefaultPromptIsEnvironmentNeutral() {
        let p = AgentPrompt.defaultPromptFile
        XCTAssertFalse(p.contains("macOS"),
                       "base prompt must be OS-neutral — environment specifics ride the per-request section")
        XCTAssertFalse(p.contains("brew"),
                       "macOS-specific tooling must not be baked into the neutral base prompt")
    }

    func testExecutionEnvironmentSectionLinuxVariant() {
        let s = AgentPrompt.executionEnvironmentSection(sandboxed: true)
        XCTAssertTrue(s.contains("# Execution environment"))
        XCTAssertTrue(s.contains("Linux"))
        XCTAssertTrue(s.contains("/workspace"), "must explain the workspace mount point")
        XCTAssertTrue(s.contains("brew") && s.contains("NOT"),
                      "must warn off macOS-only tooling inside the guest")
        XCTAssertTrue(s.lowercased().contains("network"),
                      "must state the guest's network posture so failed downloads aren't retried forever")
        XCTAssertTrue(s.contains("run_in_background") && s.lowercased().contains("log"),
                      "must explain sandboxed background commands: a bg handle (readProcessOutput/killProcess) plus a guest log")
        XCTAssertTrue(s.contains("readProcessOutput") && s.contains("killProcess"),
                      "sandboxed background handles now poll/kill exactly like the host")
        XCTAssertFalse(s.contains("zsh"))
    }

    func testExecutionEnvironmentSectionMacVariant() {
        let s = AgentPrompt.executionEnvironmentSection(sandboxed: false)
        XCTAssertTrue(s.contains("# Execution environment"))
        XCTAssertTrue(s.contains("Mac"))
        XCTAssertTrue(s.contains("brew"), "host variant restores the macOS tooling hint")
        XCTAssertFalse(s.contains("Linux"))
        XCTAssertFalse(s.contains("/workspace"))
    }

    // The URL a served app is handed back on is ENVIRONMENT-specific: on the
    // host a 0.0.0.0 bind is LAN-reachable at http://<local-ip>:<port>, but in
    // the sandbox only the loopback port map answers — a LAN or guest IP URL
    // is dead. Live 2026-07-02: the base prompt's <local-ip> directive made
    // the agent hand the user the Mac's LAN IP for a sandboxed server. So the
    // base prompt must not hardcode a URL form; each env section states its own.
    func testServedUrlFormRidesTheEnvironmentSectionNotTheBasePrompt() {
        XCTAssertFalse(AgentPrompt.defaultPromptFile.contains("<local-ip>"),
                       "URL form is environment-specific — the base prompt must defer to the env section")
        let sandbox = AgentPrompt.executionEnvironmentSection(sandboxed: true)
        XCTAssertTrue(sandbox.contains("http://localhost:"),
                      "sandbox section must state the mapped localhost URL form")
        XCTAssertTrue(sandbox.contains("NEVER") || sandbox.contains("never"),
                      "sandbox section must explicitly countermand LAN/local-ip URLs")
        let host = AgentPrompt.executionEnvironmentSection(sandboxed: false)
        XCTAssertTrue(host.contains("<local-ip>"),
                      "host section carries the LAN-reachable URL directive (IP from the grounding line)")
    }

    // MARK: - Skill seeding

    private func tempSkillsDir() -> String {
        (NSTemporaryDirectory() as NSString).appendingPathComponent("mlx-skills-\(UUID().uuidString)")
    }

    func testSkillManagerSeedsDefaultReviewSkillOnFirstRun() {
        let dir = tempSkillsDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }

        let mgr = SkillManager(skillsDir: dir)   // dir doesn't exist → first run
        XCTAssertTrue(FileManager.default.fileExists(atPath: (dir as NSString).appendingPathComponent("review.md")),
                      "the example skill is written on first run")

        // It parses, is listed in the always-on index, and its body is injected
        // when a trigger phrase appears.
        let triggered = mgr.matchingSkills(for: "can you review my changes?")
        XCTAssertTrue(triggered.contains("Available skills:"))
        XCTAssertTrue(triggered.contains("review ("), "review skill is indexed")
        XCTAssertTrue(triggered.contains("## Skill: review"), "trigger 'review' injects the body")

        // No trigger → index only, no body.
        let untriggered = mgr.matchingSkills(for: "what files are here")
        XCTAssertFalse(untriggered.contains("## Skill: review"))
    }

    func testSkillManagerDoesNotReSeedAfterUserDeletesExample() throws {
        let dir = tempSkillsDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }

        _ = SkillManager(skillsDir: dir)         // first run seeds review.md
        let reviewPath = (dir as NSString).appendingPathComponent("review.md")
        try FileManager.default.removeItem(atPath: reviewPath)   // user deletes it

        _ = SkillManager(skillsDir: dir)         // dir still exists → must NOT re-seed
        XCTAssertFalse(FileManager.default.fileExists(atPath: reviewPath),
                       "deleting the example sticks once the skills dir exists")
    }
}
