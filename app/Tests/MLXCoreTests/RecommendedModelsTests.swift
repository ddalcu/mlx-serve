import XCTest
@testable import MLXCore

/// Pins the "Recommended" pane's data: two family catalogs (Gemma 4, Qwen
/// 3.5/3.6), each ascending by size, plus the RAM-requirements math that
/// drives the pane's dim-but-never-hide treatment. The pane exists to answer
/// "which model should I download" for someone with zero AI experience, so
/// every entry's copy and sizing claims are load-bearing — get either wrong
/// and the recommendation itself is wrong, not just a cosmetic bug.
final class RecommendedModelsTests: XCTestCase {

    private let GiB: UInt64 = 1_073_741_824

    /// Every recommended pick across all four family sections — the union the
    /// invariant tests below sweep, so a new section can't slip past them.
    private var allRecommended: [RecommendedModelPick] {
        RecommendedModelPick.allCatalogs
    }

    /// `allCatalogs` is what the invariant sweeps run over, so it has to BE the
    /// four sections — a section left out of it is a section with no guards.
    func testAllCatalogsIsTheUnionOfTheFourSections() {
        let union = RecommendedModelPick.gemmaCatalog
            + RecommendedModelPick.qwenCatalog
            + RecommendedModelPick.poolsideCatalog
            + RecommendedModelPick.largestCatalog
        XCTAssertEqual(RecommendedModelPick.allCatalogs, union)
    }

    // MARK: - Catalog shape

    /// The pane's whole layout assumes exactly four family sections
    /// (Gemma 4, Qwen, poolside Laguna, Largest).
    func testExactlyFourFamiliesArePresent() {
        let families = Set(allRecommended.map(\.family))
        XCTAssertEqual(families, [.gemma, .qwen, .poolside, .largest])
    }

    /// A family catalog can't be empty — a section with zero rows would be a
    /// dead header in the UI.
    func testNoFamilyCatalogIsEmpty() {
        XCTAssertFalse(RecommendedModelPick.gemmaCatalog.isEmpty)
        XCTAssertFalse(RecommendedModelPick.qwenCatalog.isEmpty)
        XCTAssertFalse(RecommendedModelPick.poolsideCatalog.isEmpty)
        XCTAssertFalse(RecommendedModelPick.largestCatalog.isEmpty)
    }

    /// Every entry in `gemmaCatalog` is actually Gemma, and every entry in
    /// `qwenCatalog` is actually Qwen — the section header promises this.
    func testEveryEntryMatchesItsCatalogsFamily() {
        for p in RecommendedModelPick.gemmaCatalog {
            XCTAssertEqual(p.family, .gemma, p.id)
        }
        for p in RecommendedModelPick.qwenCatalog {
            XCTAssertEqual(p.family, .qwen, p.id)
        }
        for p in RecommendedModelPick.poolsideCatalog {
            XCTAssertEqual(p.family, .poolside, p.id)
        }
        for p in RecommendedModelPick.largestCatalog {
            XCTAssertEqual(p.family, .largest, p.id)
        }
    }

    /// Each family list renders smallest-to-largest, so a beginner scans it
    /// as "gets more capable as you go".
    func testEachFamilyCatalogIsSortedAscendingBySize() {
        let gemmaSizes = RecommendedModelPick.gemmaCatalog.map(\.sizeGB)
        XCTAssertEqual(gemmaSizes, gemmaSizes.sorted())
        let qwenSizes = RecommendedModelPick.qwenCatalog.map(\.sizeGB)
        XCTAssertEqual(qwenSizes, qwenSizes.sorted())
        let poolsideSizes = RecommendedModelPick.poolsideCatalog.map(\.sizeGB)
        XCTAssertEqual(poolsideSizes, poolsideSizes.sorted())
        let hunyuanSizes = RecommendedModelPick.largestCatalog.map(\.sizeGB)
        XCTAssertEqual(hunyuanSizes, hunyuanSizes.sorted())
    }

    /// No id collisions within or across the catalogs — ids key the
    /// SwiftUI `ForEach`/download-state lookups.
    func testNoDuplicateIdsAcrossAllCatalogs() {
        let ids = allRecommended.map(\.id)
        XCTAssertEqual(ids.count, Set(ids).count)
    }

    /// Every repo id must look like a real, resolvable HuggingFace path
    /// (`org/repo`, no whitespace) — a typo here silently 404s the download.
    func testRepoIdsAreWellFormed() {
        for p in allRecommended {
            XCTAssertTrue(p.repoId.contains("/"), p.repoId)
            XCTAssertFalse(p.repoId.contains(" "), p.repoId)
            XCTAssertEqual(p.repoId.split(separator: "/").count, 2, p.repoId)
        }
    }

    /// Every entry needs real, non-empty plain-English copy — an empty blurb
    /// or tagline would silently render a blank description.
    func testEveryPickHasNonEmptyBeginnerCopy() {
        for p in allRecommended {
            XCTAssertFalse(p.name.isEmpty, p.id)
            XCTAssertFalse(p.tagline.isEmpty, p.id)
            XCTAssertGreaterThan(p.blurb.count, 40, "\(p.id) blurb reads as a stub")
        }
    }

    // MARK: - Capability scores (the three bars)

    /// Every score is on the 0–100 scale the bars divide by, and every pick
    /// names a real context window. A stray 0 or 120 renders an empty or
    /// overflowing track with nothing else to catch it.
    func testEveryPickHasScoresInRange() {
        for p in allRecommended {
            XCTAssertTrue((0...100).contains(p.intelligence), "\(p.id) intelligence \(p.intelligence)")
            XCTAssertTrue((0...100).contains(p.speed), "\(p.id) speed \(p.speed)")
            XCTAssertGreaterThan(p.contextTokens, 0, p.id)
            XCTAssertGreaterThan(p.activeParamsB, 0, p.id)
        }
    }

    /// **The invariant that makes hand-tuning the speed scores safe.** Decode
    /// on Apple Silicon is bandwidth-bound, so a model that wakes MORE
    /// parameters per token can never be faster than one that wakes fewer. A
    /// tie in active params is unconstrained — that's where quantization
    /// (26B-A4B at 4-bit vs 8-bit) and expert-bank size legitimately differ.
    func testABiggerModelIsNeverScoredFasterThanASmallerOne() {
        let byActiveDescending = allRecommended.sorted { $0.activeParamsB > $1.activeParamsB }
        for (i, big) in byActiveDescending.enumerated() {
            for small in byActiveDescending[(i + 1)...] where small.activeParamsB < big.activeParamsB {
                XCTAssertLessThanOrEqual(
                    big.speed, small.speed,
                    "\(big.id) wakes \(big.activeParamsB)B and is scored FASTER (\(big.speed)) than \(small.id) at \(small.activeParamsB)B (\(small.speed))"
                )
            }
        }
    }

    /// The same weights quantized twice are the same model: an 8-bit build is
    /// slower and bigger, never smarter.
    func testAHigherPrecisionBuildInheritsItsSiblingsIntelligence() {
        XCTAssertEqual(RecommendedModelPick.gemma26bA4b8bit.intelligence,
                       RecommendedModelPick.gemma26bA4b.intelligence)
        XCTAssertEqual(RecommendedModelPick.gemma31B8bit.intelligence,
                       RecommendedModelPick.gemma31B.intelligence)
        XCTAssertLessThan(RecommendedModelPick.gemma31B8bit.speed,
                          RecommendedModelPick.gemma31B.speed)
    }

    /// Only the three picks the site has no entry for are flagged estimated —
    /// leaving a real score flagged (or an invented one unflagged) is the whole
    /// point of carrying the flag.
    func testOnlyTheModelsAbsentFromTheIndexAreFlaggedEstimated() {
        let estimated = Set(allRecommended.filter(\.intelligenceIsEstimated).map(\.id))
        XCTAssertEqual(estimated, ["laguna-xs-2.1-nvfp4", "laguna-s-2.1-nvfp4", "hy3-oq2e", "qwen38-27b"])
    }

    /// The bar fractions the pane draws stay inside the track, and context —
    /// which is compared on a log scale — orders the way the raw windows do.
    func testBarFractionsStayInsideTheTrack() {
        for p in allRecommended {
            XCTAssertTrue((0...1).contains(p.intelligenceBar), p.id)
            XCTAssertTrue((0...1).contains(p.speedBar), p.id)
            XCTAssertTrue((0...1).contains(p.contextBar), p.id)
        }
        // 128K < 256K < 1M, and the 1M pick fills the track.
        XCTAssertLessThan(RecommendedModelPick.gemmaE4B.contextBar,
                          RecommendedModelPick.gemma12B.contextBar)
        XCTAssertLessThan(RecommendedModelPick.gemma12B.contextBar,
                          RecommendedModelPick.deepseekV4Flash.contextBar)
        XCTAssertEqual(RecommendedModelPick.deepseekV4Flash.contextBar, 1.0, accuracy: 0.001)
    }

    /// The context bar shows the MODEL's window, not the RAM-clamped effective
    /// one — pinned against each checkpoint's own `max_position_embeddings`.
    func testContextWindowsMatchTheCheckpoints() {
        XCTAssertEqual(RecommendedModelPick.gemmaE2B.contextTokens, 131_072)
        XCTAssertEqual(RecommendedModelPick.gemmaE4B.contextTokens, 131_072)
        XCTAssertEqual(RecommendedModelPick.gemma31B.contextTokens, 262_144)
        XCTAssertEqual(RecommendedModelPick.qwen38_27b.contextTokens, 262_144)
        XCTAssertEqual(RecommendedModelPick.lagunaS21.contextTokens, 262_144)
        XCTAssertEqual(RecommendedModelPick.deepseekV4Flash.contextTokens, 1_048_576)
    }

    /// `activeParamsB` is a fact per checkpoint, not a restatement of the
    /// headline size: an MoE counts only what it wakes.
    func testActiveParamsAreTheWokenParametersNotTheTotal() {
        XCTAssertEqual(RecommendedModelPick.lagunaS21.activeParamsB, 8.5)      // 117.6B total
        XCTAssertEqual(RecommendedModelPick.qwen36_35bA3b.activeParamsB, 3.0)  // 35B total
        XCTAssertEqual(RecommendedModelPick.hy3_295b.activeParamsB, 21.0)      // 295B total
        XCTAssertEqual(RecommendedModelPick.deepseekV4Flash.activeParamsB, 13.0) // 284B total
        XCTAssertEqual(RecommendedModelPick.gemma31B.activeParamsB, 31.0)      // dense
    }

    // MARK: - Starter recommendation (RAM tiers)

    /// The four bands, sampled in the middle of each.
    func testStarterPickPerRamTier() {
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 8 * GiB).id, "gemma-4-e2b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 12 * GiB).id, "gemma-4-e4b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 24 * GiB).id, "gemma-4-12b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 64 * GiB).id, "qwen38-27b")
    }

    /// Bands are upper-inclusive, so a machine sitting exactly ON a boundary
    /// takes the smaller side — it has the least headroom in its band.
    func testStarterPickBoundariesAreExact() {
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 16 * GiB).id, "gemma-4-e4b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 17 * GiB).id, "gemma-4-12b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 32 * GiB).id, "gemma-4-12b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 33 * GiB).id, "qwen38-27b")
    }

    /// Every tier's pick actually runs on the SMALLEST Mac in its band — a
    /// recommendation the machine can't load is worse than no recommendation.
    func testEveryStarterTierFitsTheBottomOfItsBand() {
        let bottoms: [UInt64] = [4 * GiB, 8 * GiB + 1, 16 * GiB + 1, 32 * GiB + 1]
        for bytes in bottoms {
            let pick = RecommendedModelPick.starterPick(physicalMemoryBytes: bytes)
            XCTAssertTrue(pick.meetsSystemRequirements(physicalMemoryBytes: bytes),
                          "\(pick.id) needs \(pick.approxRAMNeededGB) GB but was recommended at \(bytes / GiB) GB")
        }
    }

    /// Total: no input can fail to produce a recommendation, including the
    /// degenerate ones a `physicalMemory` read could theoretically hand back.
    func testStarterPickIsTotal() {
        for bytes: UInt64 in [0, 1, 2 * GiB, 96 * GiB, 512 * GiB, UInt64.max] {
            XCTAssertFalse(RecommendedModelPick.starterPick(physicalMemoryBytes: bytes).repoId.isEmpty)
        }
    }

    /// Every starter tier is a plain safetensors pick. The shared card handles
    /// a GGUF pick (`ggufFilename` → the quant download path) because it must
    /// not assume otherwise, but nothing routes there today.
    func testNoStarterTierIsAGgufPick() {
        for bytes: UInt64 in [8 * GiB, 16 * GiB, 32 * GiB, 128 * GiB] {
            XCTAssertNil(RecommendedModelPick.starterPick(physicalMemoryBytes: bytes).ggufFilename)
        }
    }

    // MARK: - Meets-system-requirements (reduced opacity, never hidden)

    /// A model well within this Mac's RAM meets requirements.
    func testMeetsSystemRequirementsWhenPlentyOfHeadroom() {
        XCTAssertTrue(RecommendedModelPick.gemmaE4B.meetsSystemRequirements(physicalMemoryBytes: 16 * GiB))
    }

    /// A model bigger than this Mac's total RAM does not meet requirements —
    /// this is the signal that sorts it behind the "Requires more RAM"
    /// disclosure, never a reason to drop it from the list.
    func testDoesNotMeetSystemRequirementsWhenTooBig() {
        XCTAssertFalse(RecommendedModelPick.gemma31B8bit.meetsSystemRequirements(physicalMemoryBytes: 16 * GiB))
    }

    /// The threshold includes the same ~20% overhead the rest of the app
    /// budgets for RAM-vs-disk-weight, so a model whose weights alone are
    /// just under total RAM still correctly reads as "won't fit".
    func testRequirementsThresholdIncludesOverhead() {
        // gemma31B: 17.2 GB weights -> ~20.64 GB needed. 18 GB of RAM covers
        // the raw weights but not the overhead.
        let pick = RecommendedModelPick.gemma31B
        XCTAssertFalse(pick.meetsSystemRequirements(physicalMemoryBytes: 18 * GiB))
        XCTAssertTrue(pick.meetsSystemRequirements(physicalMemoryBytes: 32 * GiB))
    }

    /// Hunyuan 3 is now the imatrix 2-bit oQ2e build (~84 GB, full 192-expert
    /// model) — the largest pick that still fits a 128 GB Mac WITH a usable
    /// context window. It sorts INLINE on 128 GB (weights×1.2 ≈ 100 GB ≤ 128)
    /// but lands behind the "Requires more RAM" disclosure on a 96 GB Mac. The
    /// older ~105 GB mixed build was gated above 128 because it left almost no
    /// context; this build is the fix, so the gate moved down.
    func testHunyuan3FitsOn128GBButNotBelow() {
        let hy3 = RecommendedModelPick.hy3_295b
        XCTAssertTrue(hy3.meetsSystemRequirements(physicalMemoryBytes: 128 * GiB))
        XCTAssertFalse(hy3.meetsSystemRequirements(physicalMemoryBytes: 96 * GiB))
        XCTAssertTrue(hy3.blurb.contains("128 GB"), "blurb must name the 128 GB target")
    }

    // MARK: - Partitioning (inline vs "Requires more RAM" disclosure)

    /// On a big enough Mac, everything in a family fits — the disclosure
    /// never appears (nothing goes into `requiresMoreRAM`).
    func testPartitionPutsEverythingInFitsOnAHighRamMac() {
        let split = RecommendedModelPick.gemmaCatalog.partitionedByRequirements(physicalMemoryBytes: 128 * GiB)
        XCTAssertEqual(split.fits.count, RecommendedModelPick.gemmaCatalog.count)
        XCTAssertTrue(split.requiresMoreRAM.isEmpty)
    }

    /// On a small Mac, the biggest Gemma picks land in `requiresMoreRAM`
    /// while the small ones stay in `fits` — and nothing is dropped: the two
    /// buckets together must reconstruct the original catalog.
    func testPartitionSplitsBySizeOnALowRamMac() {
        let split = RecommendedModelPick.gemmaCatalog.partitionedByRequirements(physicalMemoryBytes: 16 * GiB)
        XCTAssertTrue(split.fits.contains(.gemmaE2B))
        XCTAssertTrue(split.requiresMoreRAM.contains(.gemma31B8bit))
        XCTAssertEqual(Set(split.fits + split.requiresMoreRAM), Set(RecommendedModelPick.gemmaCatalog))
    }

    /// Each bucket preserves the catalog's ascending-size order — the
    /// partition must not reshuffle, only split.
    func testPartitionPreservesAscendingOrderWithinEachBucket() {
        let split = RecommendedModelPick.gemmaCatalog.partitionedByRequirements(physicalMemoryBytes: 16 * GiB)
        XCTAssertEqual(split.fits.map(\.sizeGB), split.fits.map(\.sizeGB).sorted())
        XCTAssertEqual(split.requiresMoreRAM.map(\.sizeGB), split.requiresMoreRAM.map(\.sizeGB).sorted())
    }

    // MARK: - Known-good entries (regression pins)

    /// The Qwen 27B slot is the 3.8 build with the MTP head in the checkpoint —
    /// the same geometry as the 3.6 27B it replaced, newer weights, vision, and
    /// the built-in speculative-decode speedup. There is exactly ONE 27B pick:
    /// two entries of the same size class in one section is a coin flip for a
    /// beginner, which is what this pane exists to remove.
    func testQwenTwentySevenBPickIsThe38MtpBuild() {
        let repoIds = RecommendedModelPick.qwenCatalog.map(\.repoId)
        XCTAssertTrue(repoIds.contains("ddalcu/Qwen3.8-27B-MLX-Serve-4bit"))
        XCTAssertFalse(repoIds.contains("ddalcu/Qwen3.6-27B-4bit-MTP-MLX-Serve"))
    }

    func testGemma4EverydayPicksArePresent() {
        let repoIds = Set(RecommendedModelPick.gemmaCatalog.map(\.repoId))
        XCTAssertTrue(repoIds.contains("mlx-community/gemma-4-e4b-it-4bit"))
        XCTAssertTrue(repoIds.contains("mlx-community/gemma-4-12b-it-4bit"))
    }

    /// The "Largest models" section holds the biggest picks, ordered smallest-
    /// first like every other catalog: the compact ~84 GB Hunyuan 3 oQ2e build
    /// then the ~118 GB native-MLX DeepSeek-V4-Flash mirror.
    func testLargestSectionHoldsHunyuanThenDeepseek() {
        XCTAssertEqual(RecommendedModelPick.largestCatalog.map(\.id), ["hy3-oq2e", "deepseek-v4-flash"])
    }

    /// DeepSeek-V4-Flash is served by our OWN native `deepseek_v4` MLX arch, not
    /// the embedded ds4 GGUF engine — so the pick is the mixed 2/3/8-bit mirror
    /// and fetches a whole safetensors repo (no `ggufFilename`). Its RAM gate is
    /// an explicit 128 GB: weights×1.2 would claim 141 GB and hide the model
    /// from the exact machine the conversion was built for (~110 GB resident).
    func testDeepseekV4FlashIsTheNativeMlxMirror() {
        let ds4 = RecommendedModelPick.deepseekV4Flash
        XCTAssertEqual(ds4.family, .largest)
        XCTAssertEqual(ds4.repoId, "ddalcu/DeepSeek-V4-Flash-0731-iQ-MLX-3.3bpw")
        XCTAssertNil(ds4.ggufFilename, "the native MLX mirror fetches the whole safetensors repo")
        XCTAssertEqual(ds4.approxRAMNeededGB, 128.0)
        XCTAssertTrue(ds4.meetsSystemRequirements(physicalMemoryBytes: 128 * GiB))
        XCTAssertFalse(ds4.meetsSystemRequirements(physicalMemoryBytes: 96 * GiB))
        XCTAssertNil(RecommendedModelPick.gemmaE4B.ggufFilename, "safetensors picks fetch the whole repo")
        XCTAssertFalse(RecommendedModelPick.allCatalogs.contains { $0.repoId == "antirez/deepseek-v4-gguf" },
                       "the ds4 GGUF pick is superseded by the native mirror")
    }

    /// poolside's Laguna family is its own section — coding-specialist MoEs
    /// that aren't Gemma or Qwen. Two picks, ascending, both poolside's own
    /// NVFP4 4-bit MLX builds: the XS 2.1 the 26.7.12 perf round was validated
    /// on (121 tok/s decode on an M4 Max), then the full S 2.1. The old
    /// 2-bit community S build (`pipenetwork/Laguna-S-2.1-MLX-2bit`) was
    /// dropped: noticeably worse output quality than the NVFP4 original.
    func testPoolsideSectionHoldsLagunaXSThenS() {
        XCTAssertEqual(RecommendedModelPick.poolsideCatalog.map(\.id),
                       ["laguna-xs-2.1-nvfp4", "laguna-s-2.1-nvfp4"])
        let xs = RecommendedModelPick.lagunaXS21
        XCTAssertEqual(xs.family, .poolside)
        XCTAssertEqual(xs.repoId, "poolside/Laguna-XS-2.1-NVFP4-mlx")
        XCTAssertNil(xs.ggufFilename, "the MLX build fetches the whole safetensors repo")
        let laguna = RecommendedModelPick.lagunaS21
        XCTAssertEqual(laguna.family, .poolside)
        XCTAssertEqual(laguna.repoId, "poolside/Laguna-S-2.1-NVFP4-mlx")
        XCTAssertNil(laguna.ggufFilename, "the MLX build fetches the whole safetensors repo")
    }

    /// The NVFP4 Laguna S build is ~67 GB on disk, so ~80 GB with the ×1.2 RAM
    /// overhead: it fits a 96 GB Mac inline but lands behind "Requires more RAM"
    /// on a 64 GB one.
    func testLagunaFitsBigMacButNotMidRangeOne() {
        let laguna = RecommendedModelPick.lagunaS21
        XCTAssertTrue(laguna.meetsSystemRequirements(physicalMemoryBytes: 96 * GiB))
        XCTAssertFalse(laguna.meetsSystemRequirements(physicalMemoryBytes: 64 * GiB))
    }

    /// The XS build is ~20 GB on disk, so ~24 GB with the ×1.2 RAM overhead:
    /// it fits a 32 GB Mac inline but lands behind "Requires more RAM" on a
    /// 16 GB one.
    func testLagunaXSFitsThirtyTwoGigMacButNotSixteen() {
        let xs = RecommendedModelPick.lagunaXS21
        XCTAssertTrue(xs.meetsSystemRequirements(physicalMemoryBytes: 32 * GiB))
        XCTAssertFalse(xs.meetsSystemRequirements(physicalMemoryBytes: 16 * GiB))
    }

    /// The old 0.8B entry-level Qwen pick was replaced with 9B — too small
    /// to be a meaningful comparison against the Gemma lineup.
    func testEntryLevelQwenPickIsNineBNotZeroEightB() {
        let repoIds = RecommendedModelPick.qwenCatalog.map(\.repoId)
        XCTAssertTrue(repoIds.contains("mlx-community/Qwen3.5-9B-MLX-4bit"))
        XCTAssertFalse(repoIds.contains { $0.contains("0.8B") })
    }

    /// The Recommended table names the quant beside the size so nobody finds
    /// out what they downloaded from the folder name.
    func testQuantLabelIsDerivedFromRepoIdOrGgufFile() {
        func label(_ repoId: String, gguf: String? = nil) -> String? {
            var p = RecommendedModelPick.gemmaCatalog[0]
            p = RecommendedModelPick(id: p.id, name: p.name, tagline: p.tagline, blurb: p.blurb,
                                     repoId: repoId, sizeGB: p.sizeGB, family: p.family,
                                     intelligence: p.intelligence, intelligenceIsEstimated: p.intelligenceIsEstimated,
                                     speed: p.speed, contextTokens: p.contextTokens, activeParamsB: p.activeParamsB,
                                     ggufFilename: gguf)
            return p.quantLabel
        }
        XCTAssertEqual(label("mlx-community/gemma-4-12b-it-4bit"), "4-bit")
        XCTAssertEqual(label("mlx-community/gemma-4-31b-it-8bit"), "8-bit")
        XCTAssertEqual(label("mlx-community/Hy3-oQ2e"), "oQ2e")
        XCTAssertEqual(label("ddalcu/DeepSeek-V4-Flash-0731-MLX-Serve-mixed-2-3-8bit"), "mixed 2/3/8-bit")
        XCTAssertEqual(label("ddalcu/DeepSeek-V4-Flash-0731-iQ-MLX-3.3bpw"), "iQ-MLX 3.3 bpw")
        XCTAssertEqual(label("poolside/Laguna-S-2.1-NVFP4-mlx"), "NVFP4")
        XCTAssertEqual(label("x/y", gguf: "model-Q4_K_M.gguf"), "Q4_K_M")
        XCTAssertNil(label("x/plain-model"))
        for pick in RecommendedModelPick.gemmaCatalog + RecommendedModelPick.qwenCatalog
                + RecommendedModelPick.poolsideCatalog + RecommendedModelPick.largestCatalog {
            XCTAssertNotNil(pick.quantLabel, pick.repoId)
        }
    }
}
