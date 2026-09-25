import XCTest
@testable import MLXCore

/// `MLXCore bench` posts to the same board as the Benchmarks window, so it
/// must never share unasked or name a model the window would not.
final class BenchmarkCLITests: XCTestCase {

    func testFlagsAndDefaults() throws {
        let bare = try BenchmarkCLI.parse([]).get()
        XCTAssertEqual(bare, BenchmarkCLI.Options(port: 11234, model: nil, note: nil, share: false))
        let long = String(repeating: "x", count: BenchmarkResult.maxNoteLength + 40)
        let all = try BenchmarkCLI.parse(["--port", "8031", "--model", "qwen", "--note", long, "--share"]).get()
        XCTAssertEqual(all, BenchmarkCLI.Options(port: 8031, model: "qwen",
                                                 note: String(long.prefix(BenchmarkResult.maxNoteLength)), share: true))
    }

    func testBadInputIsAnErrorNotADefault() {
        XCTAssertEqual(BenchmarkCLI.parse(["--sahre"]), .failure(.init(message: "unknown argument --sahre")))
        for args in [["--port"], ["--port", "abc"], ["--port", "0"], ["--port", "70000"], ["--model"], ["extra"]] {
            guard case .failure = BenchmarkCLI.parse(args) else { return XCTFail("\(args) should not parse") }
        }
    }

    func testPicksAResidentLocalChatModelLikeTheWindow() {
        func model(_ id: String, chat: Bool = true, loaded: Bool = true, lan: String? = nil) -> ModelInfo {
            var entry: [String: Any] = ["id": id, "capabilities": chat ? ["chat"] : ["image"], "loaded": loaded]
            if let lan { entry["lan_peer"] = lan }
            return APIClient.parseModelInfo(entry)
        }
        let models = [model("flux", chat: false), model("stub", loaded: false), model("peer", lan: "studio.local"), model("qwen")]
        XCTAssertEqual(BenchmarkCLI.pickModel(models, requested: nil)?.name, "qwen")
        XCTAssertEqual(BenchmarkCLI.pickModel(models, requested: "qwen")?.name, "qwen")
        for other in ["flux", "stub", "peer", "missing"] {
            XCTAssertNil(BenchmarkCLI.pickModel(models, requested: other), other)
        }
    }
}
