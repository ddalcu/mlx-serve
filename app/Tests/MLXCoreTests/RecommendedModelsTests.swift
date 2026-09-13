import XCTest
@testable import MLXCore

/// Pins the "Recommended" pane's data: three sections (Gemma 4, Qwen,
/// Largest), each ascending by size, plus the RAM-requirements math that
/// drives the pane's dim-but-never-hide treatment. The pane exists to answer
/// "which model should I download" for someone with zero AI experience, so
/// every entry's copy and sizing claims are load-bearing — get either wrong
/// and the recommendation itself is wrong, not just a cosmetic bug.
final class RecommendedModelsTests: XCTestCase {

    private let GiB: UInt64 = 1_073_741_824

    /// Every recommended pick across all three sections — the union the
    /// invariant tests below sweep, so a new section can't slip past them.
    private var allRecommended: [RecommendedModelPick] {
        RecommendedModelPick.allCatalogs
    }

    /// `allCatalogs` is what the invariant sweeps run over, so it has to BE the
    /// three sections — a section left out of it is a section with no guards.
    func testAllCatalogsIsTheUnionOfTheThreeSections() {
        let union = RecommendedModelPick.gemmaCatalog
            + RecommendedModelPick.qwenCatalog
            + RecommendedModelPick.largestCatalog
        XCTAssertEqual(RecommendedModelPick.allCatalogs, union)
    }

    // MARK: - Catalog shape

    /// The pane's whole layout assumes exactly three sections
    /// (Gemma 4, Qwen, Largest).
    func testExactlyThreeFamiliesArePresent() {
        let families = Set(allRecommended.map(\.family))
        XCTAssertEqual(families, [.gemma, .qwen, .largest])
    }

    /// A family catalog can't be empty — a section with zero rows would be a
    /// dead header in the UI.
    func testNoFamilyCatalogIsEmpty() {
        XCTAssertFalse(RecommendedModelPick.gemmaCatalog.isEmpty)
        XCTAssertFalse(RecommendedModelPick.qwenCatalog.isEmpty)
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
        let largestSizes = RecommendedModelPick.largestCatalog.map(\.sizeGB)
        XCTAssertEqual(largestSizes, largestSizes.sorted())
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
    /// A pick scored at its draft-head rate is a different quantity and is
    /// swept separately below.
    func testABiggerModelIsNeverScoredFasterThanASmallerOne() {
        let byActiveDescending = allRecommended.filter { !$0.speedIsWithMtp }
            .sorted { $0.activeParamsB > $1.activeParamsB }
        for (i, big) in byActiveDescending.enumerated() {
            for small in byActiveDescending[(i + 1)...] where small.activeParamsB < big.activeParamsB {
                XCTAssertLessThanOrEqual(
                    big.speed, small.speed,
                    "\(big.id) wakes \(big.activeParamsB)B and is scored FASTER (\(big.speed)) than \(small.id) at \(small.activeParamsB)B (\(small.speed))"
                )
            }
        }
    }

    /// The MTP-scored picks are the ones whose checkpoint ships a draft head
    /// this app runs by default, scored from the bench's `mtp` cells: the
    /// 35B-A3B is the fastest thing here, Flash-Next sits above the 27B.
    func testMtpScoredPicksAreTheOnesShippingADraftHead() {
        let mtp = Set(allRecommended.filter(\.speedIsWithMtp).map(\.id))
        XCTAssertEqual(mtp, ["qwen38-27b", "qwen36-35b-a3b", "qwen38-flash-next"])
        XCTAssertEqual(RecommendedModelPick.qwen36_35bA3b.speed, allRecommended.map(\.speed).max())
        XCTAssertGreaterThan(RecommendedModelPick.qwen38FlashNext.speed, RecommendedModelPick.qwen38_27b.speed)
        XCTAssertLessThan(RecommendedModelPick.qwen38FlashNext.speed, RecommendedModelPick.qwen36_35bA3b.speed)
    }

    /// The same weights quantized twice are the same model: an 8-bit build is
    /// slower and bigger, never smarter.
    func testAHigherPrecisionBuildInheritsItsSiblingsIntelligence() {
        XCTAssertEqual(RecommendedModelPick.gemma26bA4b8bit.intelligence,
                       RecommendedModelPick.gemma26bA4b.intelligence)
        XCTAssertLessThan(RecommendedModelPick.gemma26bA4b8bit.speed,
                          RecommendedModelPick.gemma26bA4b.speed)
    }

    /// The Gemma section starts at E4B (E2B is too small to recommend) and
    /// carries one 8-bit build, the MoE; the dense 31B at 8-bit is gone.
    func testGemmaSectionDropsE2BAndThe31B8bit() {
        let ids = RecommendedModelPick.gemmaCatalog.map(\.id)
        XCTAssertEqual(ids.first, "gemma-4-e4b")
        XCTAssertFalse(ids.contains("gemma-4-e2b"))
        XCTAssertFalse(ids.contains("gemma-4-31b-8bit"))
        XCTAssertTrue(ids.contains("gemma-4-31b"))
    }

    /// The hover card over the bars names each bar and prints its score —
    /// the bars alone cannot say which is which.
    func testCapabilityTipNamesBothBarsWithTheirScores() {
        let fn = CapabilityTip.lines(for: .qwen38FlashNext)
        XCTAssertEqual(fn, ["Intelligence: 67 (our estimate)", "Speed: 47 (with its built-in draft head)"])
        let e4b = CapabilityTip.lines(for: .gemmaE4B)
        XCTAssertEqual(e4b, ["Intelligence: 20", "Speed: 57"])
    }

    /// Only the picks the site has no entry for are flagged estimated —
    /// leaving a real score flagged (or an invented one unflagged) is the whole
    /// point of carrying the flag.
    func testOnlyTheModelsAbsentFromTheIndexAreFlaggedEstimated() {
        let estimated = Set(allRecommended.filter(\.intelligenceIsEstimated).map(\.id))
        XCTAssertEqual(estimated, ["qwen38-27b", "qwen38-flash-next"])
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
        XCTAssertEqual(RecommendedModelPick.gemmaE4B.contextTokens, 131_072)
        XCTAssertEqual(RecommendedModelPick.gemma31B.contextTokens, 262_144)
        XCTAssertEqual(RecommendedModelPick.qwen38_27b.contextTokens, 262_144)
        XCTAssertEqual(RecommendedModelPick.qwen38FlashNext.contextTokens, 262_144)
        XCTAssertEqual(RecommendedModelPick.deepseekV4Flash.contextTokens, 1_048_576)
    }

    /// `activeParamsB` is a fact per checkpoint, not a restatement of the
    /// headline size: an MoE counts only what it wakes.
    func testActiveParamsAreTheWokenParametersNotTheTotal() {
        XCTAssertEqual(RecommendedModelPick.qwen36_35bA3b.activeParamsB, 3.0)  // 35B total
        XCTAssertEqual(RecommendedModelPick.qwen38FlashNext.activeParamsB, 6.0) // 125B total
        XCTAssertEqual(RecommendedModelPick.deepseekV4Flash.activeParamsB, 13.0) // 284B total
        XCTAssertEqual(RecommendedModelPick.gemma31B.activeParamsB, 31.0)      // dense
    }

    // MARK: - Starter recommendation (RAM tiers)

    /// The four bands, sampled in the middle of each.
    func testStarterPickPerRamTier() {
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 8 * GiB).id, "gemma-4-e4b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 24 * GiB).id, "gemma-4-12b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 64 * GiB).id, "qwen38-27b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 128 * GiB).id, "qwen38-flash-next")
    }

    /// Bands are upper-inclusive, so a machine sitting exactly ON a boundary
    /// takes the smaller side — it has the least headroom in its band.
    func testStarterPickBoundariesAreExact() {
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 16 * GiB).id, "gemma-4-e4b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 17 * GiB).id, "gemma-4-12b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 32 * GiB).id, "gemma-4-12b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 33 * GiB).id, "qwen38-27b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 95 * GiB).id, "qwen38-27b")
        XCTAssertEqual(RecommendedModelPick.starterPick(physicalMemoryBytes: 96 * GiB).id, "qwen38-flash-next")
    }

    /// Every tier's pick actually runs on the SMALLEST Mac in its band — a
    /// recommendation the machine can't load is worse than no recommendation.
    func testEveryStarterTierFitsTheBottomOfItsBand() {
        let bottoms: [UInt64] = [8 * GiB, 16 * GiB + 1, 32 * GiB + 1, 96 * GiB]
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
        XCTAssertFalse(RecommendedModelPick.gemma31B.meetsSystemRequirements(physicalMemoryBytes: 16 * GiB))
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

    /// Flash-Next is ~100 GB on disk but the 32 GB n-gram table is mmapped,
    /// never resident, so the honest RAM gate is the ~70 GB of weights plus
    /// headroom: inline on a 96 GB Mac (tight against Metal's default working
    /// set there), behind "Requires more RAM" on 64 GB.
    func testFlashNextFitsOn96GBButNotBelow() {
        let fn = RecommendedModelPick.qwen38FlashNext
        XCTAssertTrue(fn.meetsSystemRequirements(physicalMemoryBytes: 96 * GiB))
        XCTAssertFalse(fn.meetsSystemRequirements(physicalMemoryBytes: 64 * GiB))
        XCTAssertTrue(fn.blurb.contains("96 GB"), "blurb must name the 96 GB target")
        let mac96 = SystemMemoryInfo(totalBytes: 96 * GiB, usableBytes: UInt64(96 * 0.84) * GiB)
        XCTAssertEqual(mac96.fit(neededGB: fn.approxRAMNeededGB), .tight)
    }

    /// DeepSeek serves in ~110 GB on a 128 GB Mac: the row must read TIGHT
    /// against that Mac's default Metal working set (~107 GB), never
    /// "exceeds", while a 96 GB Mac still fails the gate.
    func testDeepseekReadsTightNotExceedsOn128GB() {
        let ds = RecommendedModelPick.deepseekV4Flash
        let mac128 = SystemMemoryInfo(totalBytes: 128 * GiB, usableBytes: UInt64(107.5 * Double(GiB)))
        XCTAssertEqual(mac128.fit(neededGB: ds.approxRAMNeededGB), .tight)
        XCTAssertFalse(ds.meetsSystemRequirements(physicalMemoryBytes: 96 * GiB))
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
        XCTAssertTrue(split.fits.contains(.gemmaE4B))
        XCTAssertTrue(split.requiresMoreRAM.contains(.gemma31B))
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
    /// first like every other catalog: the ~100 GB Qwen 3.8 Flash-Next pack
    /// then the ~130 GB native-MLX DeepSeek-V4-Flash mirror.
    func testLargestSectionHoldsFlashNextThenDeepseek() {
        XCTAssertEqual(RecommendedModelPick.largestCatalog.map(\.id), ["qwen38-flash-next", "deepseek-v4-flash"])
        let fn = RecommendedModelPick.qwen38FlashNext
        XCTAssertEqual(fn.repoId, "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit")
        XCTAssertEqual(fn.quantLabel, "mixed 4/8-bit")
        XCTAssertEqual(fn.intelligence, RecommendedModelPick.deepseekV4Flash.intelligence)
    }

    /// The 35B slot is our own MLX-Serve pack (in-checkpoint MTP head).
    func testQwenThirtyFiveBPickIsTheMlxServePack() {
        XCTAssertEqual(RecommendedModelPick.qwen36_35bA3b.repoId, "ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit")
        XCTAssertFalse(RecommendedModelPick.qwenCatalog.contains { $0.repoId == "mlx-community/Qwen3.6-35B-A3B-4bit" })
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
        XCTAssertEqual(ds4.approxRAMNeededGB, 104.0)
        XCTAssertTrue(ds4.meetsSystemRequirements(physicalMemoryBytes: 128 * GiB))
        XCTAssertFalse(ds4.meetsSystemRequirements(physicalMemoryBytes: 96 * GiB))
        XCTAssertNil(RecommendedModelPick.gemmaE4B.ggufFilename, "safetensors picks fetch the whole repo")
        XCTAssertFalse(RecommendedModelPick.allCatalogs.contains { $0.repoId == "antirez/deepseek-v4-gguf" },
                       "the ds4 GGUF pick is superseded by the native mirror")
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
        for pick in RecommendedModelPick.allCatalogs {
            XCTAssertNotNil(pick.quantLabel, pick.repoId)
        }
    }
}
