import XCTest
@testable import MLXCore

/// The community SDXL finetunes (Illustrious / Pony / NoobAI) reach the same
/// backend through THREE different repo shapes, and the shape is declared on the
/// preset rather than sniffed from its id. These pin the declarations, because
/// every one of them fails silently: a wrong bundle downloads the wrong bytes
/// (or twice as many), and a wrong capability flag hides a control the
/// checkpoint is actually steered with.
///
/// Only Illustrious ships in `ImageModelPreset.all` — Pony and both NoobAI
/// presets are deliberately CLI-only (`ImageModelPreset.all`'s doc comment),
/// so this suite reaches them directly rather than through the catalog. Their
/// download/load behaviour is real and exercised the same way regardless of
/// whether the app's picker lists them.
final class SdxlFinetuneCatalogTests: XCTestCase {

    private var finetunes: [ImageModelPreset] {
        [.illustriousXLv2_Q4, .ponyDiffusionV6XL, .noobaiXLv11, .noobaiXLVPred10]
    }

    func testTheFinetunesReachTheSdxlBackend() {
        let ids = Set(finetunes.map(\.id))
        XCTAssertTrue(ids.contains("sceneworks/illustrious-xl-v2-q4"))
        XCTAssertTrue(ids.contains("lylia/pony-diffusion-v6-xl"))
        XCTAssertTrue(ids.contains("laxhar/noobai-xl-v1.1"))
        XCTAssertTrue(ids.contains("laxhar/noobai-xl-vpred-1.0"))
        // All of them are served by the sdxl engine, not a family of their own.
        for p in finetunes { XCTAssertEqual(p.configName, "sdxl", p.id) }
    }

    func testOnlyIllustriousIsInTheAppsCuratedCatalog() {
        let catalogIds = Set(ImageModelPreset.all.map(\.id))
        XCTAssertTrue(catalogIds.contains("sceneworks/illustrious-xl-v2-q4"))
        XCTAssertFalse(catalogIds.contains("lylia/pony-diffusion-v6-xl"))
        XCTAssertFalse(catalogIds.contains("laxhar/noobai-xl-v1.1"))
        XCTAssertFalse(catalogIds.contains("laxhar/noobai-xl-vpred-1.0"))
    }

    func testEveryFinetuneTakesANegativePromptAndSdxlsOwnGrid() {
        // These are NOT distilled — they run real guidance, and the anime-SDXL
        // ecosystem steers with negative prompts. Withholding the field would
        // hide the control users actually reach for.
        for p in finetunes {
            XCTAssertTrue(p.supportsNegativePrompt, p.id)
            XCTAssertEqual(p.resolutionGrid.alignment, 64, p.id)
            XCTAssertEqual(p.resolutionGrid.minDim, 512, p.id)
        }
    }

    func testASubfolderVariantPullsOnlyItsOwnQuantAndFlattensIt() throws {
        let p = try XCTUnwrap(finetunes.first { $0.id == "sceneworks/illustrious-xl-v2-q4" })
        let sel = p.bundle.components[0].selection
        XCTAssertEqual(sel.subfolder, "q4")
        // Recursive is what reaches `q4/unet/…`; without it the depth gate keeps
        // only the subfolder's immediate children and no weights arrive.
        XCTAssertTrue(sel.recursive)

        let entries: [[String: Any]] = [
            ["path": "q4/model_index.json", "type": "file", "size": 700],
            ["path": "q4/unet/diffusion_pytorch_model.safetensors", "type": "file", "size": 2_595_555_776],
            ["path": "q4/text_encoder_2/model.safetensors", "type": "file", "size": 610_429_154],
            ["path": "q8/unet/diffusion_pytorch_model.safetensors", "type": "file", "size": 5_000_000_000],
            ["path": "bf16/unet/config.json", "type": "file", "size": 1850],
        ]
        let picked = DownloadManager.selectNeededFiles(from: entries, selection: sel).map(\.0)
        XCTAssertTrue(picked.contains("q4/unet/diffusion_pytorch_model.safetensors"))
        XCTAssertTrue(picked.contains("q4/model_index.json"))
        // The OTHER variants must not ride along — that is the whole point.
        XCTAssertFalse(picked.contains(where: { $0.hasPrefix("q8/") || $0.hasPrefix("bf16/") }))
        // And the prefix comes off on the way to disk, so the dir looks like a
        // plain diffusers SDXL repo to the server's folder loader.
        XCTAssertEqual(sel.localPath(forRemote: "q4/unet/diffusion_pytorch_model.safetensors"),
                       "unet/diffusion_pytorch_model.safetensors")
    }

    func testASingleFileCheckpointPullsExactlyOneWeightFileAndNoModelIndex() throws {
        let p = try XCTUnwrap(finetunes.first { $0.id == "laxhar/noobai-xl-vpred-1.0" })
        let sel = p.bundle.components[0].selection

        // NoobAI ships a COMPLETE diffusers folder beside the checkpoint. Pulling
        // both would double the download; worse, a `model_index.json` on disk
        // sends `Engine.loadAuto` down the folder path whose weights we skipped.
        let entries: [[String: Any]] = [
            ["path": "NoobAI-XL-Vpred-v1.0.safetensors", "type": "file", "size": 7_105_350_110],
            ["path": "model_index.json", "type": "file", "size": 671],
            ["path": "unet/diffusion_pytorch_model.safetensors", "type": "file", "size": 5_135_149_760],
            ["path": "text_encoder/model.safetensors", "type": "file", "size": 246_144_152],
            ["path": "scheduler/scheduler_config.json", "type": "file", "size": 676],
        ]
        let picked = DownloadManager.selectNeededFiles(from: entries, selection: sel).map(\.0)
        XCTAssertEqual(picked, ["NoobAI-XL-Vpred-v1.0.safetensors"])
    }

    func testPonyPullsItsCheckpointAndLeavesTheStandaloneVaeBehind() throws {
        let p = try XCTUnwrap(finetunes.first { $0.id == "lylia/pony-diffusion-v6-xl" })
        let sel = p.bundle.components[0].selection
        let entries: [[String: Any]] = [
            ["path": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors", "type": "file", "size": 6_938_041_050],
            ["path": "sdxl_vae.safetensors", "type": "file", "size": 334_641_162],
            ["path": "images/00056.jpeg", "type": "file", "size": 215_080],
        ]
        let picked = DownloadManager.selectNeededFiles(from: entries, selection: sel).map(\.0)
        XCTAssertEqual(picked, ["ponyDiffusionV6XL_v6StartWithThisOne.safetensors"])
    }

    func testNoTwoCatalogPresetsShareADownloadDestination() {
        // The download destination is derived from the repo id, so two presets
        // on one repo would overwrite each other's files on disk — each reading
        // as "downloaded" while holding the other's weights. Until a per-variant
        // destination exists, the catalog must not contain such a pair.
        var seen: [String: String] = [:]
        for p in ImageModelPreset.all {
            if let prior = seen[p.repo] {
                XCTFail("\(p.id) and \(prior) share repo \(p.repo) and would collide on disk")
            }
            seen[p.repo] = p.id
        }
    }
}
