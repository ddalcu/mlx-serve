import XCTest
@testable import MLXCore

/// The reasoning block's header, once the reasoning is over.
final class ThinkingDurationTests: XCTestCase {

    /// Nothing measured yet, or nothing worth reporting: a reply from a history
    /// written before the duration existed has no number, and neither has one
    /// that is still being written.
    func testWithoutADurationItIsJustThinking() {
        XCTAssertEqual(ThinkingDuration.label(seconds: nil), "Thinking")
        XCTAssertEqual(ThinkingDuration.label(seconds: 0), "Thinking")
        XCTAssertEqual(ThinkingDuration.label(seconds: 0.4), "Thinking",
                       "under a second is not a wait anyone noticed")
    }

    func testSecondsReadAsSeconds() {
        XCTAssertEqual(ThinkingDuration.label(seconds: 1), "Thinking took 1 second")
        XCTAssertEqual(ThinkingDuration.label(seconds: 6.4), "Thinking took 6 seconds")
        XCTAssertEqual(ThinkingDuration.label(seconds: 59), "Thinking took 59 seconds")
    }

    /// "127 seconds" is a number you have to do arithmetic on before it means
    /// anything.
    func testPastAMinuteItSplits() {
        XCTAssertEqual(ThinkingDuration.label(seconds: 60), "Thinking took 1 minute")
        XCTAssertEqual(ThinkingDuration.label(seconds: 61), "Thinking took 1 minute 1 second")
        XCTAssertEqual(ThinkingDuration.label(seconds: 367), "Thinking took 6 minutes 7 seconds")
        XCTAssertEqual(ThinkingDuration.label(seconds: 120), "Thinking took 2 minutes")
    }

    /// The seconds are dropped on the minute, not rendered as "2 minutes 0
    /// seconds".
    func testAWholeMinuteDropsTheSeconds() {
        XCTAssertFalse(ThinkingDuration.label(seconds: 180).contains("0 second"))
    }

    /// The duration is measured once, while the turn runs, and never again -
    /// so if it does not survive a save it is gone for good and every reopened
    /// conversation reads "Thinking". `ChatMessage` decodes by hand, which is
    /// exactly where a new key gets forgotten on one side.
    func testTheDurationSurvivesASaveAndReload() throws {
        var message = ChatMessage(role: .assistant, content: "answer",
                                  reasoningContent: "let me think")
        message.thinkingSeconds = 367

        let data = try JSONEncoder().encode(message)
        XCTAssertTrue(String(data: data, encoding: .utf8)?.contains("thinkingSeconds") == true,
                      "the key never reached the file")

        let reloaded = try JSONDecoder().decode(ChatMessage.self, from: data)
        XCTAssertEqual(reloaded.thinkingSeconds, 367)
        XCTAssertEqual(ThinkingDuration.label(seconds: reloaded.thinkingSeconds),
                       "Thinking took 6 minutes 7 seconds")
    }

    /// A history written before this field existed still decodes, and its
    /// reasoning blocks simply carry no duration.
    func testAMessageWithoutTheFieldStillDecodes() throws {
        let json = """
        {"id":"\(UUID().uuidString)","role":"assistant","content":"hi",
         "isStreaming":false,"timestamp":0,"reasoningContent":"hmm"}
        """.data(using: .utf8)!
        let message = try JSONDecoder().decode(ChatMessage.self, from: json)
        XCTAssertNil(message.thinkingSeconds)
        XCTAssertEqual(ThinkingDuration.label(seconds: message.thinkingSeconds), "Thinking")
    }
}
