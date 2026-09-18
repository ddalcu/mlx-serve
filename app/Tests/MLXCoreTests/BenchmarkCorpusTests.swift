import XCTest
import CryptoKit
@testable import MLXCore

/// The ladder corpus is a port of llmprobe's `src/bench/corpus.ts`. Two
/// implementations of one workload have to produce the same bytes, or a
/// number the app publishes and a number llmprobe publishes describe
/// different prompts under the same rung label.
final class BenchmarkCorpusTests: XCTestCase {

    // MARK: - Byte budget

    func testTheContextHitsItsByteTargetExactly() {
        // `buildCodeContext` cuts the concatenated modules at the byte target;
        // the fit assumes every byte it asked for landed.
        for bytes in [512, 4096, 65536] {
            XCTAssertEqual(BenchmarkCorpus.buildCodeContext(bytes: bytes).utf8.count, bytes)
        }
        XCTAssertEqual(BenchmarkCorpus.buildCodeContext(bytes: 0), "")
        XCTAssertEqual(BenchmarkCorpus.buildCodeContext(bytes: -5), "")
    }

    func testTheCorpusIsPureASCIISoByteAndCharacterCountsAgree() {
        // The TypeScript slices in UTF-16 units; the port slices bytes. They
        // agree only while every character is one byte.
        let text = BenchmarkCorpus.buildCodeContext(bytes: 65536)
        XCTAssertTrue(text.utf8.allSatisfy { $0 < 0x80 })
        XCTAssertEqual(text.count, text.utf8.count)
    }

    func testTheCorpusIsDeterministic() {
        XCTAssertEqual(BenchmarkCorpus.buildCodeContext(bytes: 4096),
                       BenchmarkCorpus.buildCodeContext(bytes: 4096))
    }

    // MARK: - Planted constant

    func testThePlantedModuleLandsOnAModuleBoundaryPastTheMidpoint() {
        let text = BenchmarkCorpus.buildCodeContextWithConstant(bytes: 4096)
        guard let at = text.range(of: BenchmarkCorpus.plantedModule) else {
            return XCTFail("planted module missing")
        }
        let before = text[..<at.lowerBound]
        XCTAssertTrue(before.hasSuffix("\n\n"), "constant must sit between files, not inside one")
        XCTAssertGreaterThanOrEqual(before.utf8.count, 2048)
        XCTAssertEqual(text.utf8.count, 4096 + BenchmarkCorpus.plantedModule.utf8.count)
    }

    func testUsedPlantedConstantMatchesOnNameOrValue() {
        XCTAssertTrue(BenchmarkCorpus.usedPlantedConstant("if (elapsed > RETRY_BUDGET_MS) throw"))
        XCTAssertTrue(BenchmarkCorpus.usedPlantedConstant("const budget = 7413;"))
        XCTAssertFalse(BenchmarkCorpus.usedPlantedConstant("const budget = 5000;"))
        XCTAssertFalse(BenchmarkCorpus.usedPlantedConstant(""))
    }

    // MARK: - Parity with the TypeScript

    func testTheConstantCorpusMatchesLLMProbeByteForByte() {
        // SHA-256 of `buildCodeContextWithConstant(4096)` as computed from
        // llmprobe's corpus.ts with node at port time. A drift in either
        // implementation changes the prompt and every rung label lies.
        let text = BenchmarkCorpus.buildCodeContextWithConstant(bytes: 4096)
        let digest = SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined()
        XCTAssertEqual(digest, "8a8daeeb86be6184a09030f893ff8eb8ef958726cdc00ab12034a8dd68eec8eb")
        XCTAssertTrue(BenchmarkCorpus.buildCodeContext(bytes: 64)
            .hasPrefix("// src/order/resolve.ts\nimport { Clock } from \"../runtime/clock\""))
    }

    // MARK: - Filler fit

    func testWithNoObservationTheFitIsTheFlatGuess() {
        XCTAssertEqual(BenchmarkCorpus.fillerBytesFor(target: 4096, fits: [], fixedChars: 300), 16384)
    }

    func testOneObservationSubtractsTheOverheadItKnowsAbout() {
        // 2048 filler + 300 fixed = 2348 bytes counted as 600 tokens on top of
        // a 10-token template → 3.98 bytes/token; 4096 target → (4086 × 3.98) − 300.
        let fits = [BenchmarkCorpus.LadderFit(bytes: 2048, tokens: 600)]
        let expected = Int((((4096.0 - 10) * (2348.0 / 590)) - 300).rounded())
        XCTAssertEqual(BenchmarkCorpus.fillerBytesFor(target: 4096, fits: fits, fixedChars: 300), expected)
    }

    func testTwoObservationsFitAStraightLineThroughTheLastPair() {
        let fits = [BenchmarkCorpus.LadderFit(bytes: 2000, tokens: 500),
                    BenchmarkCorpus.LadderFit(bytes: 4000, tokens: 1000)]
        // slope 4 bytes/token: 4000 + 4 × (2000 − 1000)
        XCTAssertEqual(BenchmarkCorpus.fillerBytesFor(target: 2000, fits: fits, fixedChars: 300), 8000)
    }

    func testTheFitIsClampedSoBadUsageCannotAskForAHundredMegabytePrompt() {
        let absurd = [BenchmarkCorpus.LadderFit(bytes: 2048, tokens: 11)]  // 1 filler token
        XCTAssertEqual(BenchmarkCorpus.fillerBytesFor(target: 512, fits: absurd, fixedChars: 300), 512 * 8)
        let tiny = [BenchmarkCorpus.LadderFit(bytes: 100, tokens: 2000)]
        XCTAssertEqual(BenchmarkCorpus.fillerBytesFor(target: 512, fits: tiny, fixedChars: 300), 512 * 2)
    }

    func testDegenerateObservationsAreIgnored() {
        let junk = [BenchmarkCorpus.LadderFit(bytes: 0, tokens: 0),
                    BenchmarkCorpus.LadderFit(bytes: 2048, tokens: 0)]
        XCTAssertEqual(BenchmarkCorpus.fillerBytesFor(target: 512, fits: junk, fixedChars: 300), 2048)
    }

    // MARK: - Prompt assembly

    func testTheCacheBustTagLeadsThePromptAndTheInstructionEndsIt() {
        let prompt = BenchmarkCorpus.prompt(archive: "ARCHIVE", tag: "abc", seq: 3,
                                            instruction: BenchmarkCorpus.codeInstruction)
        XCTAssertTrue(prompt.hasPrefix("[probe abc-3] ARCHIVE"))
        XCTAssertTrue(prompt.hasSuffix("\n\n" + BenchmarkCorpus.codeInstruction))
    }

    func testTheCodingAndCountingRunsShareTheirArchiveSoTheCountingRunIsAPrefixHit() {
        // The ceiling run measures speculation over the SAME prefix. A fresh
        // tag would make it pay the prefill again and measure nothing new.
        let coding = BenchmarkCorpus.prompt(archive: "A", tag: "t", seq: 1, instruction: "code")
        let counting = BenchmarkCorpus.prompt(archive: "A", tag: "t", seq: 1, instruction: "count")
        XCTAssertEqual(coding.prefix(12), counting.prefix(12))
    }

    func testRungFixedCharsFollowsTheInstructionLength() {
        XCTAssertEqual(BenchmarkCorpus.rungFixedChars, 120 + BenchmarkCorpus.codeInstruction.count + 30)
        XCTAssertTrue(BenchmarkCorpus.codeInstruction.contains(BenchmarkCorpus.plantedConstant))
    }
}
