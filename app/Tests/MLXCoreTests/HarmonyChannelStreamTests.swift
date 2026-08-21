import XCTest

@testable import MLXCore

final class HarmonyChannelStreamTests: XCTestCase {
    func testModelEmittedAnalysisBecomesReasoning() {
        var stream = HarmonyChannelStream()
        let a = stream.feed("<|channel|>analysis<|message|>First thought")
        let b = stream.feed(" and second.<|end|><|start|>assistant<|channel|>final<|message|>")
        let c = stream.feed("The answer.<|end|>")

        XCTAssertEqual(a.content + b.content + c.content, "The answer.")
        XCTAssertEqual(a.reasoning + b.reasoning + c.reasoning, "First thought and second.")
    }

    func testHarmonyMarkersSplitAcrossChunksNeverLeak() {
        var stream = HarmonyChannelStream()
        let chunks = [
            "<|chan", "nel|>anal", "ysis<|mess", "age|>think",
            "ing<|en", "d|><|start|>ass", "istant<|chan",
            "nel|>final<|message|>answer", "<|en", "d|>"
        ]
        var content = ""
        var reasoning = ""
        for chunk in chunks {
            let out = stream.feed(chunk)
            content += out.content
            reasoning += out.reasoning
        }
        let tail = stream.finish()
        content += tail.content
        reasoning += tail.reasoning

        XCTAssertEqual(content, "answer")
        XCTAssertEqual(reasoning, "thinking")
        XCTAssertFalse(content.contains("<|"))
        XCTAssertFalse(reasoning.contains("<|"))
    }

    func testNormalContentWithoutHarmonyIsUnchanged() {
        var stream = HarmonyChannelStream()
        let out = stream.feed("hello, world")
        XCTAssertEqual(out.content, "hello, world")
        XCTAssertEqual(out.reasoning, "")
    }

    func testTruncatedAnalysisFlushesAsReasoning() {
        var stream = HarmonyChannelStream()
        _ = stream.feed("<|channel|>analysis<|message|>unfinished thought")
        let out = stream.finish()
        XCTAssertEqual(out.content, "")
        XCTAssertEqual(out.reasoning, "unfinished thought")
    }

    func testFinalChannelIsContent() {
        var stream = HarmonyChannelStream()
        let out = stream.feed("<|channel|>final<|message|>visible answer<|end|>")
        XCTAssertEqual(out.content, "visible answer")
        XCTAssertEqual(out.reasoning, "")
    }
}
