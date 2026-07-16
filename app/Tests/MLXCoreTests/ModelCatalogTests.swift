import XCTest
@testable import MLXCore

/// Pins the curated download catalog (`gemmaModelOptions`) and the menu-bar tray
/// subset (`gemmaModelOptionsTrayMenu`). The tray filter keys on the literal
/// substring `"4bit"` in `id` — an entry written with `"4-bit"` would silently
/// vanish from the tray, so the surfacing is tested through the real filter.
final class ModelCatalogTests: XCTestCase {

    func testQwen36MtpIsCuratedAndSurfacesInTray() {
        let repo = "ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve"
        XCTAssertTrue(
            gemmaModelOptions.contains { $0.repoId == repo },
            "Qwen 3.6 27B (4-bit, MTP) must be in the curated catalog"
        )
        XCTAssertTrue(
            gemmaModelOptionsTrayMenu.contains { $0.repoId == repo },
            "Qwen 3.6 27B (4-bit, MTP) must surface in the menu-bar tray (id needs the \"4bit\" token)"
        )
    }

    /// The DeepSeek-V4-Flash ds4 entry rides in the tray, points at the imatrix
    /// quant (better quality at the same size), and — being a `deepseek_v4`
    /// GGUF — its download auto-pulls the MTP draft head via the shared
    /// `startGguf` path (`DownloadManager.mtpSidecarPath`). Pins that the tray
    /// download gets the imatrix model + MTP without a tray-specific code path.
    func testDeepseekV4FlashTrayEntryUsesImatrixAndTriggersMtpAutoDownload() {
        guard let ds4 = gemmaModelOptions.first(where: { $0.id == "dsv4-flash-gguf" }) else {
            return XCTFail("DeepSeek-V4-Flash (ds4) must be in the curated catalog")
        }
        XCTAssertTrue(gemmaModelOptionsTrayMenu.contains { $0.id == ds4.id },
                      "DS4 must surface in the menu-bar tray (id carries the \"dsv4\" token)")
        let file = ds4.ggufFilename ?? ""
        XCTAssertTrue(file.contains("imatrix"), "tray DS4 must download the imatrix build, got \(file)")
        XCTAssertTrue(file.contains("IQ2XXS"))
        // A `deepseek_v4` primary is what routes the download to MTP auto-pull.
        XCTAssertEqual(DownloadManager.ggufModelType(forBasename: file), "deepseek_v4")
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
