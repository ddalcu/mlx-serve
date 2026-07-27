import XCTest
@testable import MLXCore

/// Agents answered at length because a written persona covers role, expertise,
/// tone and priorities and says nothing about LENGTH — and outside voice mode
/// nothing else does either (plain chat has no system message, the tool-loop
/// prompt is about tools).
///
/// The fix is NOT a setting. Brevity is written INTO the prompt when a model
/// writes it, so it's visible in the editor and the user can reword or delete it
/// like any other sentence. Nothing is hidden, nothing to configure, and prompts
/// the user wrote themselves are left alone.
final class AgentBrevityTests: XCTestCase {

    func testAWrittenPromptGainsABrevityLineWhenTheModelDidNotGiveOne() {
        let parsed = AgentWriter.parse("""
            NAME: Code Reviewer
            PROMPT: You review Swift code. You flag real defects and ignore style.
            """, brief: "review my code")
        let prompt = AgentWriter.concise(try! XCTUnwrap(parsed)).systemPrompt
        XCTAssertTrue(prompt.hasPrefix("You review Swift code."), "the model's words come first")
        XCTAssertTrue(prompt.contains(AgentWriter.brevityLine),
                      "an AI-written prompt says how long to answer: \(prompt)")
    }

    func testItIsNotAddedTwiceWhenTheModelAlreadyAskedForBrevity() {
        // The instructions ask for it, so a good model provides its own — a second
        // sentence saying the same thing is prompt bloat on every turn.
        let parsed = AgentWriter.parse("""
            NAME: Chef
            PROMPT: You are a home cook. Keep answers concise and lead with the answer.
            """, brief: "cooking")
        let prompt = AgentWriter.concise(try! XCTUnwrap(parsed)).systemPrompt
        XCTAssertFalse(prompt.contains(AgentWriter.brevityLine))
        XCTAssertTrue(prompt.contains("concise"))
    }

    func testTheDetectionCoversTheWaysAModelPhrasesIt() {
        for phrasing in ["Answer concisely.", "Be brief.", "Keep replies short.",
                         "Use as few words as possible.", "Avoid being verbose."] {
            XCTAssertTrue(AgentWriter.mentionsBrevity(phrasing), phrasing)
        }
        XCTAssertFalse(AgentWriter.mentionsBrevity("You are a warm, thorough travel planner."))
    }

    func testTheFallbackDraftIsConciseToo() {
        // The model refused or there was none — the user's own words become the
        // prompt, and they still shouldn't produce essays.
        let draft = AgentWriter.concise(AgentWriter.fallbackDraft(brief: "a blunt code reviewer for Swift"))
        XCTAssertTrue(draft.systemPrompt.contains(AgentWriter.brevityLine))
    }

    func testTheBrevityLineAlwaysSurvivesTheLengthCap() {
        // A model that fills the 1200-char budget must not push the brevity line
        // off the end — the prose is trimmed to make room for it.
        let long = String(repeating: "You are helpful. ", count: 200)
        let draft = AgentWriter.concise(
            try! XCTUnwrap(AgentWriter.parse("NAME: Big\nPROMPT: \(long)", brief: "b")))
        XCTAssertLessThanOrEqual(draft.systemPrompt.count, AgentWriter.maxPromptCharacters)
        XCTAssertTrue(draft.systemPrompt.hasSuffix(AgentWriter.brevityLine),
                      "…and it's the last thing in the prompt: \(draft.systemPrompt.suffix(120))")
    }

    func testAHandWrittenPromptIsNeverRewritten() {
        // Only the AI-write path adds anything. Typing your own prompt — or
        // editing a generated one — is left exactly as typed.
        var a = Agent(name: "Mine", brief: "", systemPrompt: "You are terse.")
        a.verbosityFieldDoesNotExist()
        let prefix = AgentResolution.resolve(agent: a, defaults: AppDefaultsSnapshot()).systemPromptPrefix
        XCTAssertEqual(prefix, "You are terse.\n\n", "no clause bolted on at resolution time")
    }

    func testTheParserItselfIsUntouchedSoThePortStillTransplants() {
        // `parse` and `fallbackDraft` are the shared iPhone code (their tests are
        // ported verbatim); the macOS-only step is applied by the composer.
        let parsed = AgentWriter.parse("NAME: X\nPROMPT: You do a thing.", brief: "b")
        XCTAssertEqual(parsed?.systemPrompt, "You do a thing.")
        XCTAssertFalse(AgentWriter.fallbackDraft(brief: "b").systemPrompt.contains(AgentWriter.brevityLine))
    }

    func testTheWriterAsksTheModelForBrevityFirst() {
        let i = AgentWriter.instructions.lowercased()
        XCTAssertTrue(i.contains("concise") || i.contains("brief"),
                      "the appended line is the backstop, not the primary mechanism")
    }

    func testStartersSayHowLongToAnswer() {
        for starter in Agent.starters {
            XCTAssertTrue(AgentWriter.mentionsBrevity(starter.systemPrompt),
                          "\(starter.name)'s prompt should set the length expectation")
        }
    }
}

/// Compile-time reminder that the knob was removed on purpose: an agent has no
/// verbosity setting, only prompt text.
private extension Agent {
    func verbosityFieldDoesNotExist() {}
}
