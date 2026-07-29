import XCTest
@testable import MLXCore

/// Turns a failed turn into the card the transcript shows.
///
/// Context overflow is the one failure a user can actually fix, so it gets its
/// own shape with the server's real counts rather than being flattened into
/// `[Error: …]` text. Everything else stays a generic notice — inventing a
/// specific diagnosis from an unrecognized error is worse than showing what the
/// server said.
final class ChatErrorNoticeTests: XCTestCase {

    // MARK: - Context overflow

    func testParsesBothCountsFromTheServerBody() {
        // The detail is the raw HTTP body, so the phrase arrives wrapped in JSON.
        let body = #"{"error":{"message":"Prompt exceeds maximum context length: 4108 tokens requested, 4096 available","type":"invalid_request_error","code":400}}"#
        let notice = ChatErrorNotice.from(APIError.badStatus(code: 400, detail: body))
        XCTAssertEqual(notice.kind, .contextOverflow)
        XCTAssertEqual(notice.neededTokens, 4108)
        XCTAssertEqual(notice.contextLength, 4096)
    }

    func testOlderServerWithoutCountsStillReadsAsOverflow() {
        // A server built before the counts were added sends the bare sentence.
        // The card must still appear — only its numbers are unknown.
        let notice = ChatErrorNotice.from(
            APIError.badStatus(code: 400, detail: "Prompt exceeds maximum context length"))
        XCTAssertEqual(notice.kind, .contextOverflow)
        XCTAssertNil(notice.neededTokens)
        XCTAssertNil(notice.contextLength)
    }

    func testAlternateOverflowPhrasingsAreRecognized() {
        // Third-party-compatible wordings we already keyed on before.
        for detail in ["context length exceeded", "Prompt too long", "maximum context reached"] {
            XCTAssertEqual(ChatErrorNotice.from(APIError.badStatus(code: 400, detail: detail)).kind,
                           .contextOverflow, "missed \(detail.debugDescription)")
        }
    }

    func testUnrelatedErrorsAreNotDiagnosedAsOverflow() {
        // Claiming "you ran out of context" on an unrelated 500 sends the user
        // to change a setting that was never the problem.
        let notice = ChatErrorNotice.from(APIError.badStatus(code: 500, detail: "model failed to load"))
        XCTAssertEqual(notice.kind, .generic)
        XCTAssertNil(notice.neededTokens)
    }

    func testNonApiErrorsBecomeGenericNotices() {
        let notice = ChatErrorNotice.from(URLError(.timedOut))
        XCTAssertEqual(notice.kind, .generic)
        XCTAssertFalse(notice.message.isEmpty, "a notice with no text is a blank card")
    }

    func testMessageIsNeverEmpty() {
        // The card renders `message` as its body; an empty string is a card
        // that says nothing at all.
        for detail in ["", "   ", "Prompt exceeds maximum context length"] {
            XCTAssertFalse(ChatErrorNotice.from(APIError.badStatus(code: 400, detail: detail))
                .message.trimmingCharacters(in: .whitespaces).isEmpty)
        }
    }

    // MARK: - Card copy

    func testOverflowHeadlineAndDetailUseTheRealCounts() {
        let notice = ChatErrorNotice(kind: .contextOverflow, message: "x",
                                     neededTokens: 4108, contextLength: 4096)
        XCTAssertEqual(notice.headline, "Model ran out of context size")
        XCTAssertEqual(notice.detail,
                       "This request needed 4,108 tokens, but the model's context window holds only 4,096.")
    }

    func testOverflowDetailFallsBackWhenCountsAreUnknown() {
        let notice = ChatErrorNotice(kind: .contextOverflow, message: "Prompt exceeds maximum context length",
                                     neededTokens: nil, contextLength: nil)
        XCTAssertEqual(notice.headline, "Model ran out of context size")
        XCTAssertFalse(notice.detail.contains("nil"))
        XCTAssertFalse(notice.detail.isEmpty)
    }

    func testGenericNoticeShowsTheServerText() {
        let notice = ChatErrorNotice(kind: .generic, message: "HTTP 500 from mlx-serve",
                                     neededTokens: nil, contextLength: nil)
        XCTAssertEqual(notice.headline, "Something went wrong")
        XCTAssertEqual(notice.detail, "HTTP 500 from mlx-serve")
    }

    func testOnlyOverflowOffersTheContextAction() {
        // The "Increase Context Size" button must not appear on errors it
        // cannot fix.
        XCTAssertTrue(ChatErrorNotice(kind: .contextOverflow, message: "x",
                                      neededTokens: nil, contextLength: nil).offersContextAction)
        XCTAssertFalse(ChatErrorNotice(kind: .generic, message: "x",
                                       neededTokens: nil, contextLength: nil).offersContextAction)
    }

    // MARK: - Persistence

    func testNoticeSurvivesAChatHistoryRoundTrip() throws {
        // Chat history is reloaded on launch; a notice that decodes back to nil
        // turns a rendered card into a blank message row.
        var msg = ChatMessage(role: .assistant, content: "")
        msg.errorNotice = ChatErrorNotice(kind: .contextOverflow, message: "m",
                                          neededTokens: 10, contextLength: 8)
        let data = try JSONEncoder().encode(msg)
        let back = try JSONDecoder().decode(ChatMessage.self, from: data)
        XCTAssertEqual(back.errorNotice, msg.errorNotice)
    }

    func testHistoryWrittenByAnOlderBuildStillDecodes() throws {
        // The field is new; every message already on disk lacks it.
        let json = #"{"id":"\#(UUID().uuidString)","role":"assistant","content":"hi","isStreaming":false,"timestamp":0,"isAgentSummary":false}"#
        let back = try JSONDecoder().decode(ChatMessage.self, from: Data(json.utf8))
        XCTAssertNil(back.errorNotice)
        XCTAssertEqual(back.content, "hi")
    }
}
