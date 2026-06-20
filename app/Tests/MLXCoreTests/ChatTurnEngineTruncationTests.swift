import XCTest
@testable import MLXCore

/// Pins the agent loop's truncation-vs-ghost classification for a tool call
/// that produced NO parsed calls.
///
/// Regression context (live JFK-novel capture, 2026-06-20): a model dumped a
/// 19k-char file into one Hermes `<function=writeFile>` call and hit the token
/// cap mid-content. The server dropped the truncated call, so the loop saw
/// empty tool calls + an opener in content and fired the *ghost* nudge ("call
/// it with proper JSON" — useless, the JSON was fine) instead of the
/// *truncation* nudge (chunk + append). `hasUnclosedToolCallOpener` +
/// `maxTokensHit` is the discriminator the loop uses to route correctly.
final class ChatTurnEngineTruncationTests: XCTestCase {

    func testTruncatedHermesOpenerIsDetected() {
        let content = "<tool_call>\n<function=writeFile>\n<parameter=content>\n# THE NOVEL\n\nChapter 1. It was a long story"
        XCTAssertTrue(ChatTurnEngine.hasUnclosedToolCallOpener(content),
                      "an unclosed <function=> opener is a truncation")
    }

    func testTruncatedToolCallWrapperIsDetected() {
        let content = "<tool_call>\n{\"name\":\"writeFile\",\"arguments\":{\"content\":\"lots and lots of text"
        XCTAssertTrue(ChatTurnEngine.hasUnclosedToolCallOpener(content),
                      "<tool_call> with no </tool_call> is a truncation")
    }

    func testTruncatedGemma4OpenerIsDetected() {
        let content = "<|tool_call>call:writeFile{\"content\":\"# THE NOVEL\nlots of prose with no closer"
        XCTAssertTrue(ChatTurnEngine.hasUnclosedToolCallOpener(content),
                      "Gemma 4 <|tool_call> with no <tool_call|> is a truncation")
    }

    func testFullyClosedCallIsNotTreatedAsTruncation() {
        // A complete-but-unparseable (genuinely malformed) call keeps its close
        // tags — that stays on the ghost path, not the truncation path.
        let content = "<tool_call>\n<function=writeFile>\n<parameter=content>hi</parameter>\n</function>\n</tool_call>"
        XCTAssertFalse(ChatTurnEngine.hasUnclosedToolCallOpener(content),
                       "a closed call is not a truncation")
    }

    func testPlainProseIsNotTruncation() {
        XCTAssertFalse(ChatTurnEngine.hasUnclosedToolCallOpener("Here is a summary of what I did."))
    }

    func testTruncationNudgeSteersTowardChunkedAppend() {
        // The nudge must tell the model to chunk via append — not retry the
        // same one-shot write or switch to an equally-capped heredoc.
        let nudge = ChatTurnEngine.truncatedToolCallNudge
        XCTAssertTrue(nudge.contains("append:\"true\""), "nudge must steer toward append")
        XCTAssertTrue(nudge.contains("chunk"), "nudge must mention chunking")
        XCTAssertTrue(nudge.contains("NOT executed"), "nudge must state the call did not run")
        XCTAssertFalse(nudge.contains("proper JSON"), "must not be the ghost/malformed-JSON nudge")
    }
}
