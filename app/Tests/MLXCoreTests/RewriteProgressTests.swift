import XCTest
@testable import MLXCore

final class RewriteProgressTests: XCTestCase {

    /// The sheet names each wait: a model loading, the prompt being read, a think, the reply.
    func testEachEventMovesTheStageItNames() {
        var p = RewriteProgress()
        XCTAssertEqual(p.stage, .waiting)
        p.apply(.loading("Qwen3.8-27B"))
        XCTAssertEqual(p.stage, .loading("Qwen3.8-27B"))
        p.apply(.sent)
        XCTAssertEqual(p.stage, .waiting)
        p.apply(.reasoning("The user wants "))
        p.apply(.reasoning("a fox."))
        XCTAssertEqual(p.stage, .thinking)
        XCTAssertEqual(p.thought, "The user wants a fox.")
        p.apply(.content("A red "))
        p.apply(.content("fox."))
        XCTAssertEqual(p.stage, .writing)
        XCTAssertEqual(p.text, "A red fox.")
    }

    /// A blank lead-in before the answer is not the answer starting.
    func testWhitespaceContentDoesNotEndTheThink() {
        var p = RewriteProgress()
        p.apply(.reasoning("hmm"))
        p.apply(.content("\n\n"))
        XCTAssertEqual(p.stage, .thinking)
    }

    /// A finished reply with nothing in it says why instead of leaving an empty box.
    func testAnEmptyReplyIsReportedAndATruncatedThinkSaysSo() {
        var p = RewriteProgress()
        p.apply(.content("A fox."))
        XCTAssertNil(p.emptyReason)

        var silent = RewriteProgress()
        silent.apply(.content("  "))
        XCTAssertEqual(silent.emptyReason, "The model finished without writing anything. Try again.")

        var overthought = RewriteProgress()
        overthought.apply(.reasoning("Let me think about this very carefully"))
        overthought.apply(.truncated)
        XCTAssertEqual(overthought.emptyReason,
                       "The model used its whole token budget thinking and wrote nothing. Try again.")
    }
}
