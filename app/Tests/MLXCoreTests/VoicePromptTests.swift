import XCTest
@testable import MLXCore

/// Pins the voice/agent system-prompt grounding: the assistant is told who it is
/// (Loki), what the current date/time is, that it must speak briefly without
/// Markdown/URLs, and that it must not read tool calls aloud.
final class VoicePromptTests: XCTestCase {

    /// 2026-06-04 15:20 UTC — mid-day so the year is stable regardless of the
    /// test machine's timezone.
    private static let fixedNow = Date(timeIntervalSince1970: 1_780_586_400)

    // MARK: SystemGrounding

    func testDateTimeLineStatesTheCurrentMoment() {
        let line = SystemGrounding.dateTimeLine(now: Self.fixedNow)
        XCTAssertTrue(line.contains("current date and time is"))
        XCTAssertTrue(line.contains("2026"))                 // year is TZ-independent here
        XCTAssertTrue(line.lowercased().contains("do not guess"))
    }

    // MARK: VoicePrompt

    func testVoiceSystemPromptIsGroundedAndIdentified() {
        let p = VoicePrompt.systemPrompt(now: Self.fixedNow)
        XCTAssertTrue(p.contains("Loki"))                    // knows its name
        XCTAssertTrue(p.contains("2026"))                    // knows the date
        XCTAssertTrue(p.contains("current date and time is"))
    }

    func testSpeakingStyleKeepsTheOriginalSpokenConstraints() {
        let p = VoicePrompt.speakingStyle.lowercased()
        XCTAssertTrue(p.contains("url"))       // don't read URLs
        XCTAssertTrue(p.contains("markdown"))  // no markdown
        XCTAssertTrue(p.contains("brief") || p.contains("conversational"))
    }

    func testVoicePromptForbidsReadingToolCallsAloud() {
        // The exact symptom from the bug report: speaking "shell command date".
        XCTAssertTrue(VoicePrompt.speakingStyle.lowercased().contains("tool"))
        XCTAssertTrue(VoicePrompt.speakingStyle.contains("shell command date"))
    }

    func testDecorateAddsVoiceIdentityWithoutDuplicatingTheDate() {
        // The agent base already carries the date line; decorate must add the
        // voice identity/style but NOT a second date line.
        let base = "You are an agent. Use tools."
        let decorated = VoicePrompt.decorate(base)
        XCTAssertTrue(decorated.hasPrefix(base), "agent prompt must come first")
        XCTAssertTrue(decorated.contains("Loki"))
        XCTAssertFalse(decorated.contains("current date and time is"))
        XCTAssertTrue(decorated.contains(VoicePrompt.speakingStyle))
    }

    func testDecorateAloneFallsBackToSpeakingStyle() {
        XCTAssertEqual(VoicePrompt.decorate(nil), VoicePrompt.speakingStyle)
        XCTAssertEqual(VoicePrompt.decorate(""), VoicePrompt.speakingStyle)
    }

    /// A custom wake phrase renames the assistant everywhere in the voice
    /// prompt — identity, the quoted wake phrase — with no trace of the
    /// default name. Raw user formatting is tolerated.
    func testIdentityFollowsTheCustomWakePhrase() {
        let p = VoicePrompt.systemPrompt(now: Self.fixedNow, phrase: "Hey, JARVIS!")
        XCTAssertTrue(p.contains("You are Jarvis"))
        XCTAssertTrue(p.contains("\"Hey Jarvis\""))
        XCTAssertFalse(p.contains("Loki"))

        let d = VoicePrompt.decorate("You are an agent.", phrase: "hey jarvis")
        XCTAssertTrue(d.contains("Jarvis"))
        XCTAssertFalse(d.contains("Loki"))
    }
}
