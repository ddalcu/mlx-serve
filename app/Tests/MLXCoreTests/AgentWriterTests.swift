import XCTest
@testable import MLXCore

/// Ported verbatim from the iPhone app (MLXChatTests/AgentTests) along with
/// `AgentWriter` itself — the pure half of "describe an agent, get a system
/// prompt". Every case below is a shape a small local model actually produces,
/// and none of them may cost the user what they typed.
final class AgentWriterTests: XCTestCase {

    // MARK: - parse

    func testParsesTheTaggedFormat() {
        let reply = """
        NAME: Recipe Helper
        PROMPT: You are a home cook's assistant. You suggest recipes from what \
        the user already has, and you keep steps short.
        """
        let draft = AgentWriter.parse(reply, brief: "help me cook")
        XCTAssertEqual(draft?.name, "Recipe Helper")
        XCTAssertTrue(draft?.systemPrompt.hasPrefix("You are a home cook's assistant.") == true)
    }

    func testParsesAMultiLinePromptAfterTheTag() {
        let reply = """
        NAME: Code Reviewer
        PROMPT: You review code.

        You only flag real defects, never style.
        """
        let draft = AgentWriter.parse(reply, brief: "review my code")
        XCTAssertEqual(draft?.name, "Code Reviewer")
        XCTAssertTrue(draft?.systemPrompt.contains("only flag real defects") == true,
                      "everything after PROMPT: belongs to the prompt")
    }

    func testSurvivesMarkdownFlourishAroundTheTags() {
        // Small models love bolding the tag.
        let reply = """
        **NAME:** Trip Planner
        **PROMPT:** You plan trips that fit a budget.
        """
        let draft = AgentWriter.parse(reply, brief: "plan trips")
        XCTAssertEqual(draft?.name, "Trip Planner")
        XCTAssertEqual(draft?.systemPrompt, "You plan trips that fit a budget.")
    }

    func testUntaggedProseStillBecomesAnAgent() {
        // No tags at all — the reply IS the prompt, named from the brief.
        let draft = AgentWriter.parse(
            "You are a patient maths tutor for a 12-year-old.",
            brief: "a patient maths tutor for my kid"
        )
        XCTAssertEqual(draft?.systemPrompt, "You are a patient maths tutor for a 12-year-old.")
        XCTAssertEqual(draft?.name, "a patient maths tutor")
    }

    func testFencedOutputIsUnwrapped() {
        let reply = """
        ```
        NAME: Fitness Coach
        PROMPT: You build training plans.
        ```
        """
        let draft = AgentWriter.parse(reply, brief: "coach me")
        XCTAssertEqual(draft?.name, "Fitness Coach")
        XCTAssertEqual(draft?.systemPrompt, "You build training plans.")
    }

    func testEmptyReplyYieldsNoDraft() {
        XCTAssertNil(AgentWriter.parse("", brief: "something"))
        XCTAssertNil(AgentWriter.parse("   \n  ", brief: "something"))
    }

    // MARK: - cleaners

    func testPromptIsCappedOnASentenceBoundary() {
        let long = String(repeating: "You are helpful. ", count: 200)
        let cleaned = AgentWriter.cleanPrompt(long)
        XCTAssertLessThanOrEqual(cleaned.count, AgentWriter.maxPromptCharacters)
        XCTAssertTrue(cleaned.hasSuffix("."), "a system prompt must not trail off mid-word")
    }

    func testWrappingQuotesAndHeadingsAreStripped() {
        XCTAssertEqual(AgentWriter.cleanPrompt("\"You are a helpful bot.\""), "You are a helpful bot.")
        XCTAssertEqual(AgentWriter.cleanPrompt("## You are a helpful bot."), "You are a helpful bot.")
    }

    // MARK: - fallbacks

    func testAFailedWriteStillProducesAUsableAgent() {
        // A model that refuses or babbles must not cost the user what they typed.
        let draft = AgentWriter.fallbackDraft(brief: "a blunt code reviewer for Swift")
        XCTAssertEqual(draft.name, "a blunt code reviewer")
        XCTAssertTrue(draft.systemPrompt.contains("a blunt code reviewer for Swift"))
    }

    func testFallbackNameNeverEmpty() {
        XCTAssertEqual(AgentWriter.fallbackName(brief: "   "), "New Agent")
    }

    // MARK: - symbol

    func testSymbolIsGuessedFromTheAgentsOwnWords() {
        XCTAssertEqual(AgentSymbol.pick(for: "a blunt code reviewer"),
                       "chevron.left.forwardslash.chevron.right")
        XCTAssertEqual(AgentSymbol.pick(for: "helps me plan a trip to Rome"), "airplane")
        XCTAssertEqual(AgentSymbol.pick(for: "something entirely unlike anything"), "sparkles")
    }

    // MARK: - macOS additions

    func testInstructionsAskForTheTwoLineFormatTheParserReads() {
        // The parser is tolerant, but the instructions and the parse must still
        // agree on the tags — a reworded prompt that drops them silently turns
        // every write into the untagged fallback.
        XCTAssertTrue(AgentWriter.instructions.contains("NAME:"))
        XCTAssertTrue(AgentWriter.instructions.contains("PROMPT:"))
        XCTAssertTrue(AgentWriter.request(brief: " a chef ").contains("a chef"))
    }

    func testDraftBecomesAnAgentWithAGuessedSymbol() {
        let draft = AgentWriter.Draft(name: "Sous Chef", systemPrompt: "You are a chef.")
        let agent = Agent(draft: draft, brief: "help me cook dinner")
        XCTAssertEqual(agent.name, "Sous Chef")
        XCTAssertEqual(agent.systemPrompt, "You are a chef.")
        XCTAssertEqual(agent.brief, "help me cook dinner")
        XCTAssertEqual(agent.symbol, "fork.knife")
        XCTAssertFalse(agent.isBuiltIn)
    }
}
