import XCTest
@testable import MLXCore

/// Pins the policy that decides when the agent loop surfaces the "output
/// truncated" notice.
///
/// Regression context: the notice used to be appended inside the per-iteration
/// stream loop, so a multi-step agent turn that hit the output cap on more than
/// one iteration stacked a notice per iteration ("⚠️ Output truncated…" twice in
/// a row). The fix shows it at most once, only at the turn boundary, and never
/// on an iteration that silently retries a truncated tool call — captured by the
/// pure `TruncationNotice.shouldShow(...)` policy here.
final class TruncationNoticeTests: XCTestCase {

    func testTextMentionsTheCapAndGuidance() {
        let t = TruncationNotice.text(cause: .maxTokens, maxTokens: 16384)
        XCTAssertTrue(t.contains("16384"), "notice should name the cap that was hit")
        XCTAssertTrue(t.lowercased().contains("truncated"))
        // Steers the user toward the fix (shorter steps / raise the cap).
        XCTAssertTrue(t.lowercased().contains("smaller") || t.contains("Settings"))
    }

    func testShownOnceAtTurnEnd() {
        XCTAssertTrue(TruncationNotice.shouldShow(maxTokensHit: true, turnEnding: true, willRetry: false))
    }

    func testNotShownMidTurn() {
        // Intermediate iteration (more tool calls coming) — no notice, so a
        // long multi-step turn can't stack one per round.
        XCTAssertFalse(TruncationNotice.shouldShow(maxTokensHit: true, turnEnding: false, willRetry: false))
    }

    func testNotShownWhenSilentlyRecovering() {
        // A truncated tool call triggers a silent retry-with-nudge; the user
        // shouldn't also see a scary truncation banner for it.
        XCTAssertFalse(TruncationNotice.shouldShow(maxTokensHit: true, turnEnding: true, willRetry: true))
    }

    func testNotShownWhenNothingWasTruncated() {
        XCTAssertFalse(TruncationNotice.shouldShow(maxTokensHit: false, turnEnding: true, willRetry: false))
    }

    // MARK: - Loop cuts (live 2026-08-05, under pi)

    func testLoopNoticeNamesNoCapBecauseNoCapWasHit() {
        let t = TruncationNotice.text(cause: .repetitionLoop, maxTokens: 16384)
        // The whole incident was a message naming an output limit neither side
        // set — the session had two thirds of its context free and pi sends no
        // max_tokens at all. Naming a cap here sends people to raise a setting
        // that was never the problem.
        XCTAssertFalse(t.contains("16384"))
        XCTAssertFalse(t.lowercased().contains("max tokens"))
        XCTAssertTrue(t.lowercased().contains("repeating"))
    }

    func testARepetitionLoopEndsTheTurnAndNothingElseDoes() {
        // The loop's text is already in the transcript (a streamed delta cannot
        // be retracted, so the server's trim never reaches a streaming client).
        // Continuing would send it back as history and the model resumes it.
        XCTAssertTrue(TruncationNotice.endsTurn(cause: .repetitionLoop))
        // A max_tokens cut keeps every existing recovery path — the reply was
        // fine and simply ran out of room.
        XCTAssertFalse(TruncationNotice.endsTurn(cause: .maxTokens))
        XCTAssertFalse(TruncationNotice.endsTurn(cause: nil))
    }

    func testCauseComesFromTheServersSiblingFieldAndDegradesToMaxTokens() {
        // Preserve max_tokens handling for backends without loop details.
        XCTAssertEqual(APIClient.truncationCause(fromChoice: ["finish_reason": "length"]), .maxTokens)
        // An unknown cause is still a truncation, not a loop.
        XCTAssertEqual(
            APIClient.truncationCause(fromChoice: [
                "finish_reason": "length",
                "finish_details": ["type": "something_new"],
            ]),
            .maxTokens
        )
        // An ordinary stop, including one with unknown details, needs no notice.
        XCTAssertNil(APIClient.truncationCause(fromChoice: ["finish_reason": "stop"]))
        XCTAssertNil(APIClient.truncationCause(fromChoice: [
            "finish_reason": "stop",
            "finish_details": ["type": "something_new"],
        ]))
        XCTAssertNil(APIClient.truncationCause(fromChoice: nil))
        XCTAssertNil(APIClient.truncationCause(fromChoice: [:]))
    }

    func testExplicitLoopDetailsShowTheNoticeAndEndTheTurnRegardlessOfFinishReason() {
        // Current and legacy server contracts, plus an omitted finish reason:
        // the explicit cause is sufficient to identify a repetition-loop cut.
        for reason: String? in ["stop", "length", "tool_calls", nil] {
            var choice: [String: Any] = ["finish_details": ["type": "repetition_loop"]]
            if let reason { choice["finish_reason"] = reason }
            let cause = APIClient.truncationCause(fromChoice: choice)
            XCTAssertEqual(cause, .repetitionLoop)
            XCTAssertTrue(TruncationNotice.endsTurn(cause: cause))
        }
    }

    func testLoopCutsSuppressAllStreamToolRecoveryPaths() async throws {
        for reason in ["stop", "length", "tool_calls"] {
            for structured in [false, true] {
                let delta: [String: Any] = structured
                    ? ["tool_calls": [["index": 0, "id": "call_1", "function": ["name": "writeFile", "arguments": "{\"path\":\"unsafe\"}"]]]]
                    : ["content": "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"unsafe\"}}</tool_call>"]
                let events = try await streamEvents(delta: delta, reason: reason, loopCut: true)
                XCTAssertTrue(events.contains { if case .truncated(.repetitionLoop) = $0 { return true }; return false })
                XCTAssertFalse(events.contains { if case .toolCalls = $0 { return true }; return false })
            }
        }
    }

    func testOrdinaryStreamToolRecoveryStillWorks() async throws {
        let structured: [String: Any] = ["tool_calls": [["index": 0, "id": "call_1", "function": ["name": "writeFile", "arguments": "{\"path\":\"safe\"}"]]]]
        let content: [String: Any] = ["content": "<tool_call>{\"name\":\"writeFile\",\"arguments\":{\"path\":\"safe\"}}</tool_call>"]
        for (delta, reason) in [(structured, "tool_calls"), (structured, "length"), (structured, "stop"), (content, "stop")] {
            let events = try await streamEvents(delta: delta, reason: reason, loopCut: false)
            XCTAssertTrue(events.contains { if case .toolCalls = $0 { return true }; return false })
        }
    }

    private func streamEvents(delta: [String: Any], reason: String, loopCut: Bool) async throws -> [SSEEvent] {
        var finish: [String: Any] = ["finish_reason": reason]
        if loopCut { finish["finish_details"] = ["type": "repetition_loop"] }
        let chunks: [[String: Any]] = [
            ["choices": [["delta": delta]]],
            ["choices": [finish]],
        ]
        let lines = try chunks.map { "data: " + String(decoding: try JSONSerialization.data(withJSONObject: $0), as: UTF8.self) }
        let input = AsyncStream<String> { continuation in
            for line in lines { continuation.yield(line) }
            continuation.yield("data: [DONE]")
            continuation.finish()
        }
        let output = AsyncThrowingStream<SSEEvent, Error>.makeStream()
        try await APIClient.consumeChatLines(input, continuation: output.continuation)
        var events: [SSEEvent] = []
        for try await event in output.stream { events.append(event) }
        return events
    }

    // MARK: - The notice is DATA, never content (2026-08-11)

    // Regression context: the banner was appended INTO `message.content`
    // (`updateLastMessage(content:)`), so both history builders sent it back
    // as assistant prose on every later turn — the model reads its own
    // warning. The live capture's chat re-sent "⚠️ Stopped — the model
    // started repeating itself…" forever. The notice now rides
    // `ChatMessage.truncationNotice`; content stays clean by construction,
    // and legacy sessions get the banner STRIPPED at history build.

    func testNoticeIsAFieldAndContentStaysClean() {
        var msg = ChatMessage(role: .assistant, content: "clean answer")
        msg.truncationNotice = TruncationNotice.Notice(cause: .repetitionLoop, maxTokens: 0)
        XCTAssertEqual(msg.content, "clean answer")
        XCTAssertTrue(msg.truncationNotice!.text.lowercased().contains("repeating"))
        // The cap variant still names its cap.
        let cap = TruncationNotice.Notice(cause: .maxTokens, maxTokens: 16384)
        XCTAssertTrue(cap.text.contains("16384"))
    }

    func testStrippedRemovesBothLegacyBannersAndLeavesProseAlone() {
        let loop = "answer" + TruncationNotice.text(cause: .repetitionLoop, maxTokens: 0)
        XCTAssertEqual(TruncationNotice.stripped(from: loop), "answer")
        // The cap number varies per session — any number must strip.
        let cap = "done" + TruncationNotice.text(cause: .maxTokens, maxTokens: 12345)
        XCTAssertEqual(TruncationNotice.stripped(from: cap), "done")
        XCTAssertEqual(TruncationNotice.stripped(from: "plain answer"), "plain answer")
    }

    func testPlainHistoryCarriesNeitherTheFieldNorTheLegacyBanner() {
        var msg = ChatMessage(
            role: .assistant,
            content: "answer" + TruncationNotice.text(cause: .repetitionLoop, maxTokens: 0))
        msg.truncationNotice = TruncationNotice.Notice(cause: .repetitionLoop, maxTokens: 0)
        let d = ChatTurnEngine.plainHistoryDict(msg)
        XCTAssertEqual(d["content"] as? String, "answer")
        XCTAssertNil(d["truncationNotice"])
    }

    @MainActor
    func testAgentHistoryStripsTheLegacyBanner() {
        let banner = TruncationNotice.text(cause: .maxTokens, maxTokens: 4096)
        let msgs = [
            ChatMessage(role: .user, content: "do the thing"),
            ChatMessage(role: .assistant, content: "did it" + banner),
        ]
        let history = AgentEngine.buildAgentHistory(
            messages: msgs, contextLength: 32768, maxTokens: 4096)
        let joined = history.compactMap { $0["content"] as? String }.joined(separator: "\n")
        XCTAssertTrue(joined.contains("did it"))
        XCTAssertFalse(joined.contains("⚠️"))
        XCTAssertFalse(joined.lowercased().contains("truncated"))
    }

    func testNoticeDecodeIsTolerant() throws {
        var msg = ChatMessage(role: .assistant, content: "hi")
        msg.truncationNotice = TruncationNotice.Notice(cause: .maxTokens, maxTokens: 512)
        let data = try JSONEncoder().encode(msg)
        let back = try JSONDecoder().decode(ChatMessage.self, from: data)
        XCTAssertEqual(back.truncationNotice, TruncationNotice.Notice(cause: .maxTokens, maxTokens: 512))

        // A cause this build doesn't know must not fail the whole message.
        var json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        json["truncationNotice"] = ["cause": "something_new", "maxTokens": 1]
        let unknown = try JSONDecoder().decode(
            ChatMessage.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(unknown.truncationNotice)
        XCTAssertEqual(unknown.content, "hi")

        // Absent forever on messages saved before the field existed.
        json["truncationNotice"] = nil
        let old = try JSONDecoder().decode(
            ChatMessage.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(old.truncationNotice)
    }
}
