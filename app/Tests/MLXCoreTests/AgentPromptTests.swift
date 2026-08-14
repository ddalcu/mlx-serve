import XCTest
@testable import MLXCore

final class AgentPromptTests: XCTestCase {
    // The output-budget section is a SCARCITY warning — it exists to stop
    // work being lost past a tight cap. It must appear ONLY when the
    // effective budget (min(max_tokens, ~2/5 of context)) is genuinely tight
    // (< ~12K tokens: a single one-shot file write measured live runs
    // 8–10.7K). Roomy machines get NO section at all — an honest "you have
    // ~419430 tokens per response" reads as an invitation to one-shot a
    // whole website in one 5-minute tool call (live 2026-07-03, Qwen3.6-27B);
    // the chunking convention lives in the writeFile tool description.
    func testOutputBudgetGuidanceOnlyAppearsWhenBudgetIsTight() {
        // Roomy: huge ctx slider, default 16K cap, or Auto on a 131K model.
        XCTAssertEqual(AgentPrompt.outputBudgetGuidance(maxTokens: 0, contextLength: 1_048_576), "")
        XCTAssertEqual(AgentPrompt.outputBudgetGuidance(maxTokens: 16384, contextLength: 131_072), "")
        XCTAssertEqual(AgentPrompt.outputBudgetGuidance(maxTokens: 0, contextLength: 131_072), "")

        // Tight via small context: 4096 * 2/5 = 1638 — warning appears,
        // with the honest context-derived numbers, never the 16384 cap.
        let smallCtx = AgentPrompt.outputBudgetGuidance(maxTokens: 16384, contextLength: 4096)
        XCTAssertTrue(smallCtx.contains("1638"), "small context must cap the budget: \(smallCtx)")
        XCTAssertFalse(smallCtx.contains("16384"),
                       "must not advertise the unreachable 16384 cap: \(smallCtx)")
        XCTAssertTrue(smallCtx.contains("~81 lines (~819 tokens)"),
                      "chunk advice must stay context-derived: \(smallCtx)")

        // Tight via a user-lowered max_tokens on a roomy model.
        let capped = AgentPrompt.outputBudgetGuidance(maxTokens: 4096, contextLength: 131_072)
        XCTAssertTrue(capped.contains("4096"), "explicit low cap must warn: \(capped)")

        // The warning always points at append-chunk recovery and the loss risk.
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

    // MARK: - Selected music engine

    // The `generate_music` schema is STATIC, but the two engines take
    // different inputs — Music 3 fails a call with no lyrics and 400s the
    // bpm/key/meter fields, ACE-Step reads all of them. The model only ever
    // sees the schema, so the selected engine's contract rides the prompt,
    // DERIVED from the preset's own flags (a third engine gets a correct line
    // without editing prose).
    func testMusicEngineNoteFollowsThePresetsOwnContract() {
        for model in MusicModelPreset.all {
            let note = AgentPrompt.musicEngineNote(model)
            XCTAssertTrue(note.contains(model.name), "the note names the selected model")
            XCTAssertEqual(note.contains("bpm"), model.supportsMusicalMeta,
                           "bpm/keyscale/meter are advertised exactly when the engine reads them: \(model.name)")
            XCTAssertEqual(note.contains("requires `lyrics`"), model.requiresLyrics,
                           "the lyric requirement is stated exactly when it exists: \(model.name)")
        }
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

    // MARK: - Skill trigger word-boundary matching (issue #92)

    private func makeSkillsDir(files: [String: String]) throws -> String {
        let dir = tempSkillsDir()
        // Create the dir BEFORE SkillManager sees it so first-run seeding
        // doesn't drop the example review.md into the fixture.
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        for (name, content) in files {
            try content.write(toFile: (dir as NSString).appendingPathComponent(name),
                              atomically: true, encoding: .utf8)
        }
        return dir
    }

    private func skillFile(name: String, trigger: String) -> String {
        """
        ---
        name: \(name)
        description: test skill \(name)
        trigger: \(trigger)
        ---
        BODY-\(name)
        """
    }

    // A trigger is a whole-word/phrase match, never a substring scan (issue
    // #92): "ui" fired inside "build"/"guide", "review" inside "preview" —
    // and every false hit injects the skill's ENTIRE body into the system
    // prompt, a silent per-request context tax on small-context local models.
    func testSkillTriggersMatchWholeWordsNotSubstrings() throws {
        let dir = try makeSkillsDir(files: [
            "ui.md": skillFile(name: "ui-helper", trigger: "ui"),
            "review.md": skillFile(name: "reviewer", trigger: "review"),
            "phrase.md": skillFile(name: "phrase-only", trigger: "code review"),
            "plan.md": skillFile(name: "planner", trigger: "/plan"),
            "deps.md": skillFile(name: "deps", trigger: "requirements.txt"),
        ])
        defer { try? FileManager.default.removeItem(atPath: dir) }
        let mgr = SkillManager(skillsDir: dir)

        // Fragments inside unrelated words must NOT fire.
        XCTAssertFalse(mgr.matchingSkills(for: "build the project").contains("## Skill: ui-helper"),
                       "'ui' must not fire inside 'build'")
        XCTAssertFalse(mgr.matchingSkills(for: "please guide me").contains("## Skill: ui-helper"),
                       "'ui' must not fire inside 'guide'")
        XCTAssertFalse(mgr.matchingSkills(for: "preview the deployment").contains("## Skill: reviewer"),
                       "'review' must not fire inside 'preview'")
        XCTAssertFalse(mgr.matchingSkills(for: "scan this barcode reviews page").contains("## Skill: phrase-only"),
                       "'code review' must not fire inside 'barcode reviews'")

        // Whole words and phrases still fire — mid-message, at string edges,
        // next to punctuation, and case-insensitively.
        XCTAssertTrue(mgr.matchingSkills(for: "build a ui").contains("## Skill: ui-helper"))
        XCTAssertTrue(mgr.matchingSkills(for: "ui layout, please").contains("## Skill: ui-helper"))
        XCTAssertTrue(mgr.matchingSkills(for: "Review my changes").contains("## Skill: reviewer"))
        XCTAssertTrue(mgr.matchingSkills(for: "do a code review please").contains("## Skill: phrase-only"))
        XCTAssertTrue(mgr.matchingSkills(for: "run /plan now").contains("## Skill: planner"))
        XCTAssertTrue(mgr.matchingSkills(for: "open requirements.txt").contains("## Skill: deps"))
    }

    // A built-in added AFTER first run has to reach installs whose skills dir
    // already exists — the old seeding gate was directory existence, so every
    // future built-in would have shipped only to brand-new installs. Seeding
    // is per-file against a ledger, so a deleted built-in still stays deleted
    // and a built-in the user never had is not resurrected by the migration.
    func testNewBuiltinSkillReachesAnExistingSkillsDir() throws {
        let fm = FileManager.default
        let dir = tempSkillsDir()
        defer { try? fm.removeItem(atPath: dir) }
        func path(_ f: String) -> String { (dir as NSString).appendingPathComponent(f) }

        // Pre-ledger install: the dir is there, the user already deleted the
        // review example.
        try fm.createDirectory(atPath: dir, withIntermediateDirectories: true)

        let mgr = SkillManager(skillsDir: dir)
        XCTAssertTrue(fm.fileExists(atPath: path("music3.md")),
                      "a built-in added after first run still ships to an existing skills dir")
        XCTAssertTrue(mgr.matchingSkills(for: "write me a song about the sea").contains("## Skill: music3"),
                      "the seeded file parses and its trigger fires")
        XCTAssertFalse(fm.fileExists(atPath: path("review.md")),
                       "migration must not resurrect a built-in the user deleted before the ledger existed")

        try fm.removeItem(atPath: path("music3.md"))
        _ = SkillManager(skillsDir: dir)
        XCTAssertFalse(fm.fileExists(atPath: path("music3.md")),
                       "deleting a seeded built-in sticks")
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
