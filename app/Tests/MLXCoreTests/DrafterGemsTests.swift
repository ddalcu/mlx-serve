import XCTest
@testable import MLXCore

/// The speculation socket: which gems fit a model, how the socket maps onto
/// the server's `drafter`/`mtp` settings, the one-time fold of the old global
/// drafter, and the pack update check that finds a new `drafter/`.
final class DrafterGemsTests: XCTestCase {
    private let pack27B = "ddalcu/Qwen3.8-27B-MLX-Serve-4bit"

    private func kinds(_ gems: [DrafterGem]) -> [DrafterGem.Kind] { gems.map(\.kind) }

    // MARK: - Catalog

    func testA27BPackTakesItsOwnDrafterWhenItShipsOneElseZLab() {
        let listed = DrafterGems.gems(forRepoId: pack27B, packFiles: ["drafter/config.json": 1, "drafter/model.safetensors": 3_849_999_999],
                                      localDrafter: false, mtpAvailable: false)
        XCTAssertEqual(listed, [DrafterGem(kind: .dflash2, repo: pack27B, subfolder: "drafter", sizeGB: 3.85)])

        let onDisk = DrafterGems.gems(forRepoId: pack27B, packFiles: nil, localDrafter: true, mtpAvailable: false)
        XCTAssertEqual(onDisk.first?.subfolder, "drafter")

        for files in [nil, ["config.json": Int64(1)]] as [[String: Int64]?] {
            let fallback = DrafterGems.gems(forRepoId: pack27B, packFiles: files, localDrafter: false, mtpAvailable: false)
            XCTAssertEqual(fallback.first?.repo, DrafterGems.qwen38DFlash2Repo)
            XCTAssertNil(fallback.first?.subfolder)
            XCTAssertNil(DrafterGems.defaultGem(fallback), "z-lab's drafter regresses the 6-bit and iQ packs: offered, never auto-filled")
        }
        XCTAssertEqual(DrafterGems.defaultGem(listed)?.subfolder, "drafter")
    }

    func testCompanionGemsFollowTheirTargets() {
        XCTAssertEqual(DrafterGems.gems(forRepoId: "mlx-community/gemma-4-e4b-it-4bit", packFiles: nil, localDrafter: false, mtpAvailable: false),
                       [DrafterGem(kind: .gemmaAssistant, repo: GemmaVariant.E4B.drafterRepoId, subfolder: nil, sizeGB: 0.16)])
        XCTAssertEqual(DrafterGems.gems(forRepoId: "ddalcu/Muse-Glimmer-30B-MLX-Serve-4bit", packFiles: nil, localDrafter: false, mtpAvailable: false).first?.repo,
                       DrafterGems.museAssistantRepo)
        XCTAssertEqual(DrafterGems.gems(forRepoId: "LiquidAI/LFM2.5-2.6B-MLX-4bit", packFiles: nil, localDrafter: false, mtpAvailable: false),
                       [DrafterGem(kind: .dspark, repo: "LiquidAI/LFM2.5-2.6B-DSpark", subfolder: nil, sizeGB: 0.66)])
        XCTAssertEqual(DrafterGems.gems(forRepoId: "LiquidAI/LFM2.5-8B-A1B-MLX-8bit", packFiles: nil, localDrafter: false, mtpAvailable: false).first?.repo,
                       "LiquidAI/LFM2.5-8B-A1B-DSpark")
        // The MoE Gemma regresses with a drafter; a drafter never pulls itself; GGUF has no drafter path.
        for repo in ["mlx-community/gemma-4-26b-a4b-it-4bit", GemmaVariant.E4B.drafterRepoId, DrafterGems.museAssistantRepo,
                     "LiquidAI/LFM2.5-2.6B-DSpark", "mlx-community/LFM2.5-VL-1.6B-4bit",
                     DrafterGems.qwen38DFlash2Repo, DrafterGems.glmDFlash2Repo, "unsloth/Qwen3.8-27B-GGUF",
                     "mlx-community/gemma-3-12b-it-4bit"] {
            XCTAssertEqual(DrafterGems.gems(forRepoId: repo, packFiles: nil, localDrafter: false, mtpAvailable: false), [], repo)
        }
    }

    func testMtpComesFirstAndAnyPackDrafterIsOffered() {
        let gems = DrafterGems.gems(forRepoId: "LiquidAI/LFM2.5-2.6B-MLX-8bit", packFiles: nil, localDrafter: true, mtpAvailable: true)
        XCTAssertEqual(kinds(gems), [.mtp, .packDrafter])
        XCTAssertEqual(DrafterGems.defaultGem(gems)?.kind, .packDrafter)
    }

    func testAGlmFlashPackOffersIncoaisDFlash2AfterMtpButNeverFillsTheSocketWithIt() {
        let mtp = DrafterGem(kind: .mtp, repo: "", subfolder: nil, sizeGB: 0)
        let glm = DrafterGem(kind: .dflash2, repo: DrafterGems.glmDFlash2Repo, subfolder: nil, sizeGB: 2.34)
        for repo in ["TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP", "zai-org/GLM-5.3-Flash"] {
            let gems = DrafterGems.gems(forRepoId: repo, packFiles: nil, localDrafter: false, mtpAvailable: true)
            XCTAssertEqual(gems, [mtp, glm], repo)
            XCTAssertNil(DrafterGems.defaultGem(gems), "MTP beats it on prose, long context and bursts: offered, never auto-filled")
        }
        XCTAssertEqual(DrafterGems.gems(forRepoId: "TensorFold/GLM-5.3-Flash-MLX-oQ4-MTP", packFiles: nil, localDrafter: false, mtpAvailable: false), [glm])

        // A pack that ships its own drafter/ offers that one instead.
        let own = DrafterGems.gems(forRepoId: "ddalcu/GLM-5.3-Flash-MLX-Serve-4bit", packFiles: nil, localDrafter: true, mtpAvailable: true)
        XCTAssertEqual(kinds(own), [.mtp, .packDrafter])

        // The socket binds the DFlash engine and writes the drafter's directory.
        XCTAssertTrue(DrafterSocket.gem(glm).bindsDflash(localDrafter: false))
        var o = ModelOverride()
        DrafterSocket.gem(glm).write(into: &o, gemPath: "/d/glm")
        XCTAssertEqual(o.drafter, "/d/glm")
        XCTAssertNil(o.mtp)

        // GGUF runs on ds4 (its own MTP); other GLMs have no drafter.
        for repo in ["unsloth/GLM-5.3-Flash-GGUF", "TensorFold/GLM-5.2-MLX-4bit", "mlx-community/GLM-5.3-4bit"] {
            XCTAssertEqual(DrafterGems.gems(forRepoId: repo, packFiles: nil, localDrafter: false, mtpAvailable: false), [], repo)
        }
    }

    func testTheMenuNamesTheLicenseWhereItAsksForADownload() {
        let glm = DrafterGem(kind: .dflash2, repo: DrafterGems.glmDFlash2Repo, subfolder: nil, sizeGB: 2.34)
        let zlab = DrafterGem(kind: .dflash2, repo: DrafterGems.qwen38DFlash2Repo, subfolder: nil, sizeGB: 3.85)
        XCTAssertEqual(SpeculationSocketRow.menuLabel(glm, onDisk: false, fits: true), "DFlash2 drafter (download 2.3 GB, non-commercial license)")
        XCTAssertEqual(SpeculationSocketRow.menuLabel(glm, onDisk: true, fits: true), "DFlash2 drafter")
        XCTAssertEqual(SpeculationSocketRow.menuLabel(glm, onDisk: false, fits: false), "DFlash2 drafter (2.3 GB, does not fit in memory)")
        XCTAssertFalse(SpeculationSocketRow.menuLabel(zlab, onDisk: false, fits: true).contains("license"), "only the CC BY-NC-ND drafter carries a note")
        XCTAssertNil(DrafterGem(kind: .mtp, repo: "", subfolder: nil, sizeGB: 0).licenseNote)
    }

    func testAGemDownloadTakesTheConfigAndWeightsAndNothingElse() {
        let incoai = [entry("README.md", 2_619), entry("assets/dflash2-figure.png", 286_889),
                      entry("config.json", 1_273), entry("model.safetensors", 2_342_169_800)]
        XCTAssertEqual(DownloadManager.selectNeededFiles(from: incoai).map(\.0), ["config.json", "model.safetensors"])
    }

    // MARK: - Socket <-> settings

    func testEverySocketStateRoundTripsThroughTheSettings() {
        let mtp = DrafterGem(kind: .mtp, repo: "", subfolder: nil, sizeGB: 0)
        let inPack = DrafterGem(kind: .dflash2, repo: pack27B, subfolder: "drafter", sizeGB: 3.85)
        let separate = DrafterGem(kind: .gemmaAssistant, repo: GemmaVariant.E4B.drafterRepoId, subfolder: nil, sizeGB: 0.16)
        let gems = [mtp, inPack, separate]
        let pathOf: (DrafterGem) -> String? = { $0 == separate ? "/d/e4b" : nil }

        let cases: [(DrafterSocket, String?, Bool?)] = [
            (.automatic, nil, nil),
            (.empty, "off", false),
            (.gem(mtp), "off", true),
            (.gem(inPack), "auto", nil),
            (.gem(separate), "/d/e4b", nil),
            (.custom("/elsewhere"), "/elsewhere", nil),
        ]
        for (socket, drafter, mtpValue) in cases {
            var o = ModelOverride()
            socket.write(into: &o, gemPath: "/d/e4b")
            XCTAssertEqual(o.drafter, drafter, "\(socket)")
            XCTAssertEqual(o.mtp, mtpValue, "\(socket)")
            XCTAssertEqual(DrafterSocket.read(o, gems: gems, pathOf: pathOf), socket)
        }
    }

    // MARK: - Row badge

    func testTheRowShowsTheStoneTheServerWillDispatch() {
        let contract: [String: Any] = ["block_size": 16, "mask_token_id": 1, "target_layer_ids": [1]]
        let configs: [String: [String: Any]] = [
            "/m/dflash/drafter": contract,
            "/m/dflash2/drafter": ["dflash_config": contract.merging(["selector_rank": 64, "selector_top_k": 4]) { $1 }],
            "/m/dspark/drafter": ["block_size": 7, "markov_rank": 256, "dflash_config": ["mask_token_id": 1, "target_layer_ids": [1]]],
            "/m/gemma/drafter": ["model_type": "gemma4_assistant"],
        ]
        func badge(_ dir: String, _ o: ModelOverride = ModelOverride(), mtpHead: Bool = false,
                   _ options: ServerOptions = ServerOptions()) -> SocketBadge {
            DrafterGems.badge(o, modelDir: dir, hasMtpHead: mtpHead, options: options) { configs[$0] }
        }
        XCTAssertEqual(badge("/m/dflash").stone, .sapphire)
        XCTAssertEqual(badge("/m/dflash2").stone, .topaz)
        XCTAssertEqual(badge("/m/dspark").stone, .amethyst)
        XCTAssertEqual(badge("/m/gemma").stone, .ruby)
        XCTAssertEqual(badge("/m/none", mtpHead: true).stone, .emerald)
        XCTAssertEqual(badge("/m/dflash2", mtpHead: true).stone, .topaz, "DFlash outranks MTP")

        var empty = ModelOverride(); DrafterSocket.empty.write(into: &empty)
        XCTAssertEqual(badge("/m/dflash2", empty, mtpHead: true), SocketBadge(stone: nil, skull: false))

        XCTAssertEqual(badge("/m/none", ModelOverride(mtp: false), mtpHead: true).stone, nil, "a model's mtp:false turns its head off")

        let lossy = ModelOverride(mtpAcceptance: .typical)
        XCTAssertEqual(badge("/m/none", lossy, mtpHead: true), SocketBadge(stone: .emerald, skull: true))
        XCTAssertEqual(badge("/m/dflash2", lossy, mtpHead: true), SocketBadge(stone: .topaz, skull: false),
                       "acceptance only bends MTP rounds")
    }

    func testDrafterRidesTheSettingsFile() throws {
        let path = (NSTemporaryDirectory() as NSString).appendingPathComponent("model-settings-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(atPath: path) }
        var file = ModelSettingsFile()
        file.set(ModelOverride(json: ["drafter": "off", "future_key": 1]), for: "/m/a")
        try file.save(path: path)
        let back = ModelSettingsFile.load(path: path).override(for: "/m/a")
        XCTAssertEqual(back?.drafter, "off")
        XCTAssertEqual(back?.extra["future_key"] as? Int, 1)
    }

    // MARK: - Migration

    func testTheGlobalDrafterFoldsIntoPerModelEntriesOnce() {
        var file = ModelSettingsFile()
        file.set(ModelOverride(json: ["drafter": "auto"]), for: "/m/decided")
        DrafterMigration.migrate(&file, pairs: ["/m/e4b": "/d/e4b", "/m/decided": "/d/x"],
                                 selected: "/m/moe", globalPath: "/d/moe", optedOut: false)
        XCTAssertEqual(file.override(for: "/m/e4b")?.drafter, "/d/e4b")
        XCTAssertEqual(file.override(for: "/m/moe")?.drafter, "/d/moe", "an explicit global pick belongs to the selected model")
        XCTAssertEqual(file.override(for: "/m/decided")?.drafter, "auto", "an existing choice is never overwritten")

        var off = ModelSettingsFile()
        DrafterMigration.migrate(&off, pairs: ["/m/e4b": "/d/e4b"], selected: "/m/e4b", globalPath: "", optedOut: true)
        XCTAssertEqual(off.override(for: "/m/e4b")?.drafter, "off", "the opt-out must stick per model")
    }

    func testTheLegacyGlobalDrafterIsReadFromTheStoredBlob() throws {
        let suite = "drafter-migration-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        XCTAssertNil(ServerOptions.legacyDrafter(defaults))
        defaults.set(try JSONSerialization.data(withJSONObject: ["port": 8080, "drafterPath": "/d/e4b", "drafterOptOut": true]),
                     forKey: "serverOptions")
        let legacy = try XCTUnwrap(ServerOptions.legacyDrafter(defaults))
        XCTAssertEqual(legacy.path, "/d/e4b")
        XCTAssertTrue(legacy.optedOut)
    }

    // MARK: - Files

    private func entry(_ path: String, _ size: Int) -> [String: Any] { ["path": path, "type": "file", "size": size] }

    func testAPackDownloadBringsItsDrafterAndTheSocketCanLeaveItOut() {
        let entries = [entry("config.json", 10), entry("model.safetensors", 20), entry("mtp/weights.safetensors", 5),
                       entry("drafter/config.json", 1), entry("drafter/model.safetensors", 30), entry("drafter/README.md", 1),
                       entry("drafter/assets/figure.png", 1)]
        XCTAssertEqual(DownloadManager.selectNeededFiles(from: entries).map(\.0),
                       ["config.json", "model.safetensors", "mtp/weights.safetensors", "drafter/config.json", "drafter/model.safetensors"],
                       "the whole pack, as mlx-serve pull fetches it")
        XCTAssertEqual(DownloadManager.selectNeededFiles(from: entries, selection: .chatWithoutDrafter).map(\.0),
                       ["config.json", "model.safetensors", "mtp/weights.safetensors"])
        XCTAssertEqual(DownloadManager.packBytesWithoutDrafter(entries), 35, "the socket fit bills the gem once")
        let selection = FileSelection.packFolder("drafter")
        let drafter = DownloadManager.selectNeededFiles(from: entries, selection: selection)
        XCTAssertEqual(drafter.map(\.0), ["drafter/config.json", "drafter/model.safetensors"])
        XCTAssertEqual(selection.localPath(forRemote: "drafter/config.json"), "drafter/config.json")
    }

    func testAnUpdateCountsMissingAndResizedFilesOnly() throws {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("pack-update-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        try Data(count: 10).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("config.json")))
        try Data(count: 3).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("model.safetensors")))
        let entries = [entry("config.json", 10), entry("model.safetensors", 20),
                       entry("drafter/config.json", 1), entry("drafter/model.safetensors", 30)]

        XCTAssertEqual(PackUpdateCheck.pending(PackUpdateCheck.wanted(entries, withDrafter: true), inDir: dir),
                       PackUpdate(files: 3, bytes: 51, replaces: 1))
        XCTAssertEqual(PackUpdateCheck.pending(PackUpdateCheck.wanted(entries, withDrafter: false), inDir: dir),
                       PackUpdate(files: 1, bytes: 20, replaces: 1), "a socket switched off never asks for the drafter")
        XCTAssertNil(PackUpdateCheck.pending([("config.json", 10)], inDir: dir))
    }
}
