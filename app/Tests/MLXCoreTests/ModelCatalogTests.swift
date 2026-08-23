import XCTest
@testable import MLXCore

/// Pins the curated download catalog (`gemmaModelOptions`) and the menu-bar tray
/// subset (`gemmaModelOptionsTrayMenu`). The tray filter keys on the literal
/// substring `"4bit"` in `id` — an entry written with `"4-bit"` would silently
/// vanish from the tray, so the surfacing is tested through the real filter.
final class ModelCatalogTests: XCTestCase {

    func testQwen38MtpIsCuratedAndSurfacesInTray() {
        let repo = "ddalcu/Qwen3.8-27B-MLX-Serve-4bit"
        XCTAssertTrue(
            gemmaModelOptions.contains { $0.repoId == repo },
            "Qwen 3.8 27B (4-bit, MTP) must be in the curated catalog"
        )
        XCTAssertTrue(
            gemmaModelOptionsTrayMenu.contains { $0.repoId == repo },
            "Qwen 3.8 27B (4-bit, MTP) must surface in the menu-bar tray (id needs the \"4bit\" token)"
        )
        XCTAssertFalse(
            gemmaModelOptions.contains { $0.repoId == "ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve" },
            "the Qwen 3.6 27B entry is superseded by the 3.8 build"
        )
    }

    /// The tray's DeepSeek-V4-Flash entry is the NATIVE MLX mirror, not the ds4
    /// GGUF it replaced: a whole safetensors repo (no `ggufFilename`, so it
    /// rides the plain repo-download path rather than `startGguf`), gated at the
    /// 128 GB Mac the conversion targets, and still carrying the `"dsv4"` token
    /// the tray filter keys on.
    func testDeepseekV4FlashTrayEntryIsTheNativeMlxMirror() {
        guard let ds4 = gemmaModelOptions.first(where: { $0.id.contains("dsv4") }) else {
            return XCTFail("DeepSeek-V4-Flash must be in the curated catalog")
        }
        XCTAssertEqual(ds4.repoId, "ddalcu/DeepSeek-V4-Flash-0731-iQ-MLX-3.3bpw")
        XCTAssertNil(ds4.ggufFilename, "the native MLX mirror fetches the whole safetensors repo")
        XCTAssertEqual(ds4.minHostRamBytes, 128 * (UInt64(1) << 30))
        XCTAssertTrue(gemmaModelOptionsTrayMenu.contains { $0.id == ds4.id },
                      "DS4 must surface in the menu-bar tray (id carries the \"dsv4\" token)")
        XCTAssertFalse(gemmaModelOptions.contains { $0.repoId == "antirez/deepseek-v4-gguf" },
                       "the ds4 GGUF entry is superseded by the native mirror")
    }

    /// Class guard: ids are the dictionary key into download state, so collisions
    /// silently merge two models' progress.
    func testCatalogIdsAreUnique() {
        let ids = gemmaModelOptions.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count, "duplicate id in gemmaModelOptions")
    }

    /// Class guard: every tray entry must satisfy the tray filter's own predicate,
    /// so a new entry can't claim tray membership without the right id token.
    func testTrayMenuMembersMatchFilterPredicate() {
        for opt in gemmaModelOptionsTrayMenu {
            XCTAssertTrue(
                opt.id.contains("4bit") || opt.id.contains("dsv4"),
                "tray entry \(opt.id) does not match the 4bit/dsv4 filter predicate"
            )
        }
    }
}
