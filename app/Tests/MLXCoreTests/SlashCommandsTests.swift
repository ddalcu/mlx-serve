import XCTest
@testable import MLXCore

/// The composer's "/" menu, and the `/name` invocation it writes.
final class SlashCommandsTests: XCTestCase {

    private let skills = [
        SkillSummary(name: "music3", description: "Write a caption plus lyrics"),
        SkillSummary(name: "review", description: "Review the current changes"),
        SkillSummary(name: "mixdown", description: "Mastering notes"),
    ]

    // The menu opens on a leading "/" and closes the moment the command is
    // chosen — a space (or a newline) ends the token. A "/" anywhere else is
    // a path or a date, never a command.
    func testMenuOpensOnlyOnALeadingSlashTokenStillBeingTyped() {
        XCTAssertEqual(SlashCommands.query(in: "/"), "")
        XCTAssertEqual(SlashCommands.query(in: "/mus"), "mus")
        XCTAssertNil(SlashCommands.query(in: ""))
        XCTAssertNil(SlashCommands.query(in: "hello"))
        XCTAssertNil(SlashCommands.query(in: "read src/main.zig"), "a path is not a command")
        XCTAssertNil(SlashCommands.query(in: "/music3 make me a song"), "the command is chosen")
        XCTAssertNil(SlashCommands.query(in: "/music3\nsecond line"))
        XCTAssertNil(SlashCommands.query(in: " /music3"), "leading space means it is prose")
    }

    func testMatchesArePrefixFirstAndCaseInsensitive() {
        XCTAssertEqual(SlashCommands.matches(query: "", in: skills).map(\.name),
                       ["music3", "review", "mixdown"], "an empty query lists everything")
        XCTAssertEqual(SlashCommands.matches(query: "MU", in: skills).map(\.name), ["music3"])
        // "i" hits mus(i)c3 / rev(i)ew / m(i)xdown only as a substring; the
        // prefix hits come first when both exist.
        XCTAssertEqual(SlashCommands.matches(query: "m", in: skills).map(\.name), ["music3", "mixdown"])
        XCTAssertTrue(SlashCommands.matches(query: "zzz", in: skills).isEmpty)
    }

    func testAcceptingReplacesTheHalfTypedTokenAndLeavesTheCaretPastIt() {
        XCTAssertEqual(SlashCommands.accept("music3", in: "/mus"), "/music3 ")
        XCTAssertEqual(SlashCommands.accept("music3", in: "/"), "/music3 ")
    }

    // The typed `/name` is what INVOKES the skill — its triggers are bypassed
    // entirely, which is the whole point of typing it.
    func testInvokedSkillNameIsTheLeadingTokenOnly() {
        XCTAssertEqual(SlashCommands.invokedSkillName(in: "/music3 make me a song about dogs"), "music3")
        XCTAssertEqual(SlashCommands.invokedSkillName(in: "/Music3"), "music3")
        XCTAssertNil(SlashCommands.invokedSkillName(in: "/"))
        XCTAssertNil(SlashCommands.invokedSkillName(in: "make me a song"))
    }

    // A skill invoked by name has to reach the model in EVERY turn path.
    // There are two — the agent loop and plain chat — and plain chat builds
    // its own system message from scratch, so the injection is a second
    // construction site that a reader of `runAgentLoop` alone would miss
    // (live: /music3 with Tools off answered from the model's own head).
    func testBothTurnPathsConsumeAnInvokedSkill() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let text = try String(contentsOf: root.appendingPathComponent("Sources/MLXServe/Services/ChatTurnEngine.swift"),
                              encoding: .utf8)
        for path in ["private func runPlainTurn", "private func runAgentLoop"] {
            guard let start = text.range(of: path) else {
                return XCTFail("\(path) is gone — re-point this scan")
            }
            // Bound the slice to THIS function — the next declaration ends it,
            // or the scan reads the neighbour's injection and always passes.
            let after = text[start.upperBound...]
            let end = after.range(of: "\n    private func ") ?? after.range(of: "\n    func ")
            let body = end.map { String(after[..<$0.lowerBound]) } ?? String(after)
            XCTAssertTrue(body.contains("invokedSkill(for:") || body.contains("matchingSkills(for:"),
                          "\(path) must consume a skill the user invoked by name")
        }
    }

    func testTypingASkillNameInjectsItsBodyThoughNoTriggerMatches() throws {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("mlx-slash-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(atPath: dir) }
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        try """
        ---
        name: deps
        description: dependency notes
        trigger: requirements.txt
        ---
        BODY-deps
        """.write(toFile: (dir as NSString).appendingPathComponent("deps.md"),
                  atomically: true, encoding: .utf8)
        let mgr = SkillManager(skillsDir: dir)

        XCTAssertFalse(mgr.matchingSkills(for: "what is in here").contains("## Skill: deps"),
                       "no trigger, no body — unchanged")
        XCTAssertTrue(mgr.matchingSkills(for: "/deps what is in here").contains("## Skill: deps"),
                      "typing the name invokes it regardless of triggers")
        XCTAssertTrue(mgr.summaries.contains { $0.name == "deps" }, "the menu can list it")
    }
}
