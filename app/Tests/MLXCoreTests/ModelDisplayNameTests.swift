import XCTest
@testable import MLXCore

/// A repo id is a filename, not a title. These pin what the readable form does
/// to real ids from this app's own catalogue.
final class ModelDisplayNameTests: XCTestCase {

    func testTheScreenshotCase() {
        XCTAssertEqual(ModelDisplayName.pretty("mlx-community/gemma-4-12b-it-4bit"),
                       "Gemma 4 12b 4-bit")
    }

    /// A family glued to its version still reads as two words.
    func testAFamilyGluedToItsVersionSplits() {
        XCTAssertEqual(ModelDisplayName.pretty("gemma4-12b-it-4bit"), "Gemma 4 12b 4-bit")
        XCTAssertEqual(ModelDisplayName.pretty("qwen3-8b-4bit"), "Qwen 3 8b 4-bit")
    }

    /// The org is identical across most of the list, so it distinguishes
    /// nothing and only costs width.
    func testTheOrgIsDropped() {
        XCTAssertEqual(ModelDisplayName.pretty("mlx-community/Qwen3-8B-4bit"), "Qwen 3 8b 4-bit")
        XCTAssertEqual(ModelDisplayName.pretty("ddalcu/Kokoro-82M-MLX-Serve"), "Kokoro 82M")
    }

    /// Packaging tokens are true of every model here, so they say nothing.
    func testPackagingTokensAreDropped() {
        XCTAssertEqual(ModelDisplayName.pretty("gemma-3-12b-it-qat-4bit"), "Gemma 3 12b 4-bit")
        XCTAssertEqual(ModelDisplayName.pretty("Llama-3.2-3B-Instruct-4bit"), "Llama 3.2 3b 4-bit")
    }

    /// A MoE's active-parameter count is part of its identity, not noise.
    func testMoeActiveParametersSurvive() {
        XCTAssertEqual(ModelDisplayName.pretty("Qwen3-30B-A3B-4bit"), "Qwen 3 30b A3b 4-bit")
    }

    /// Quant spellings that are acronyms read as typos lowercased and as
    /// shouting title-cased.
    func testQuantAcronymsKeepTheirCase() {
        XCTAssertEqual(ModelDisplayName.pretty("Qwen3-8B-bf16"), "Qwen 3 8b BF16")
        XCTAssertEqual(ModelDisplayName.pretty("LFM2.5-2.6B-nvfp4"), "LFM 2.5 2.6b NVFP4")
    }

    /// A LAN id names the Mac it lives on. That is not part of the model's
    /// name, so it is split off before the words are cased and re-attached
    /// after — otherwise the peer gets title-cased into the middle of them.
    func testALanPeerIsKeptButNotTitleCased() {
        XCTAssertEqual(ModelDisplayName.pretty("mlx-community/gemma-4-12b-it-4bit@studio"),
                       "Gemma 4 12b 4-bit @ studio")
    }

    /// Never return nothing. A name we cannot improve is echoed as-is — a blank
    /// pill is worse than an ugly one.
    func testAnUnrecognisableIdIsEchoed() {
        XCTAssertEqual(ModelDisplayName.pretty(""), "")
        XCTAssertEqual(ModelDisplayName.pretty("org/"), "org/")
        XCTAssertEqual(ModelDisplayName.pretty("it-qat-mlx"), "it-qat-mlx",
                       "all-noise must not render as an empty string")
    }

    /// The id itself is what you paste into a config and search HF for, so it
    /// must survive somewhere — the readable form is additive, never a
    /// replacement for identity.
    func testTheReadableNameIsNotAnIdentity() {
        let id = "mlx-community/gemma-4-12b-it-4bit"
        XCTAssertNotEqual(ModelDisplayName.pretty(id), id)
    }
}
