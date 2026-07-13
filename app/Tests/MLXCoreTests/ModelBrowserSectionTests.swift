import XCTest
@testable import MLXCore

/// Unit tests for the pure decisions behind the Model Browser's sidebar.
///
/// The browser used to be one pane with a "Downloaded" push-button toggle that
/// silently swapped the data source, hid on-disk models from search results,
/// and collided with the "Downloads" (HuggingFace pull count) column. The
/// sidebar splits those into named destinations; everything that decides what a
/// row or a badge shows lives here so it can be tested without SwiftUI.
final class ModelBrowserSectionTests: XCTestCase {

    // MARK: - Sections

    func testEverySectionHasADistinctTitleAndIcon() {
        let titles = ModelBrowserSection.allCases.map(\.title)
        let icons = ModelBrowserSection.allCases.map(\.systemImage)
        XCTAssertEqual(Set(titles).count, ModelBrowserSection.allCases.count)
        XCTAssertEqual(Set(icons).count, ModelBrowserSection.allCases.count)
        XCTAssertFalse(titles.contains(where: \.isEmpty))
    }

    /// The HuggingFace search field, format picker, and Search button belong to
    /// Discover alone. Showing them over a filesystem list is what made the old
    /// toggle read as a filter.
    func testOnlyDiscoverShowsTheSearchControls() {
        for section in ModelBrowserSection.allCases {
            XCTAssertEqual(section.showsSearchControls, section == .discover, "\(section)")
        }
    }

    /// "Downloaded" is gone as a label — it collided with the HF pull-count
    /// column. Nothing in the sidebar may reintroduce it.
    func testNoSectionIsCalledDownloaded() {
        for section in ModelBrowserSection.allCases {
            XCTAssertNotEqual(section.title.lowercased(), "downloaded")
        }
    }

    // MARK: - Badges

    private func counts(myModels: Int = 0, active: Int = 0, drafters: Int = 0, media: Int = 0) -> ModelBrowserBadgeCounts {
        ModelBrowserBadgeCounts(myModels: myModels, activeDownloads: active, draftersReady: drafters, mediaReady: media)
    }

    func testDiscoverNeverCarriesABadge() {
        let c = counts(myModels: 9, active: 3, drafters: 2, media: 4)
        XCTAssertNil(c.badge(for: .discover))
    }

    func testBadgesShowTheirCount() {
        let c = counts(myModels: 3, active: 1, drafters: 2, media: 4)
        XCTAssertEqual(c.badge(for: .myModels), "3")
        XCTAssertEqual(c.badge(for: .downloads), "1")
        XCTAssertEqual(c.badge(for: .drafters), "2")
        XCTAssertEqual(c.badge(for: .media), "4")
    }

    func testZeroCountsHideTheBadgeEntirely() {
        let c = counts()
        XCTAssertNil(c.badge(for: .myModels))
        XCTAssertNil(c.badge(for: .downloads))
        XCTAssertNil(c.badge(for: .drafters))
        XCTAssertNil(c.badge(for: .media))
    }

    /// An in-flight download must be visible from Discover — that's the whole
    /// point of promoting Downloads to a sidebar destination.
    func testDownloadsBadgeAppearsAsSoonAsOneIsInFlight() {
        XCTAssertEqual(counts(active: 1).badge(for: .downloads), "1")
    }

    // MARK: - Live disk polling

    /// Only the panes that render on-disk state need the 1 Hz `refreshModels()`
    /// rescan, and only while something is actually downloading.
    func testLivePollOnlyWhereDiskStateIsShownAndOnlyWhileDownloading() {
        XCTAssertTrue(ModelBrowserSection.shouldLivePoll(section: .myModels, hasActiveDownloads: true))
        XCTAssertTrue(ModelBrowserSection.shouldLivePoll(section: .downloads, hasActiveDownloads: true))
        XCTAssertFalse(ModelBrowserSection.shouldLivePoll(section: .discover, hasActiveDownloads: true))
        XCTAssertFalse(ModelBrowserSection.shouldLivePoll(section: .drafters, hasActiveDownloads: true))

        for section in ModelBrowserSection.allCases {
            XCTAssertFalse(ModelBrowserSection.shouldLivePoll(section: section, hasActiveDownloads: false), "\(section)")
        }
    }

    // MARK: - Row action state machine

    func testIncompatibleModelIsUnsupportedRegardlessOfEverythingElse() {
        let a = ModelRowAction.resolve(isCompatible: false, isReady: true, status: .downloading, hasPartial: true)
        XCTAssertEqual(a, .unsupported)
    }

    /// The bug this whole change exists to kill: a model that finished
    /// downloading used to be filtered OUT of the search results. Now it stays,
    /// in an `.onDisk` state.
    func testReadyModelResolvesToOnDiskInsteadOfVanishing() {
        XCTAssertEqual(ModelRowAction.resolve(isCompatible: true, isReady: true, status: nil, hasPartial: false), .onDisk)
        XCTAssertEqual(ModelRowAction.resolve(isCompatible: true, isReady: true, status: .completed, hasPartial: false), .onDisk)
    }

    /// A `.completed` status with the readiness check still failing (e.g. a
    /// half-written GGUF) also renders as on-disk — preserving the old view's
    /// behaviour, which showed a trash can in that case.
    func testCompletedStatusResolvesToOnDiskEvenWhenNotReady() {
        XCTAssertEqual(ModelRowAction.resolve(isCompatible: true, isReady: false, status: .completed, hasPartial: false), .onDisk)
    }

    func testDownloadingCarriesItsProgress() {
        XCTAssertEqual(
            ModelRowAction.resolve(isCompatible: true, isReady: false, status: .downloading, hasPartial: false, progress: 0.62),
            .downloading(progress: 0.62)
        )
    }

    func testFailedDistinguishesResumeFromRetry() {
        XCTAssertEqual(ModelRowAction.resolve(isCompatible: true, isReady: false, status: .failed, hasPartial: true), .failed(resumable: true))
        XCTAssertEqual(ModelRowAction.resolve(isCompatible: true, isReady: false, status: .failed, hasPartial: false), .failed(resumable: false))
    }

    func testUntouchedModelOffersDownloadOrResume() {
        XCTAssertEqual(ModelRowAction.resolve(isCompatible: true, isReady: false, status: nil, hasPartial: false), .notDownloaded(resumable: false))
        XCTAssertEqual(ModelRowAction.resolve(isCompatible: true, isReady: false, status: .idle, hasPartial: true), .notDownloaded(resumable: true))
    }

    // MARK: - "Use" resolution

    private func local(_ name: String, path: String, kind: ModelKind = .base, type: String = "gemma4") -> LocalModel {
        LocalModel(id: path, name: name, path: path, sizeFormatted: "4 GB", modelType: type, source: .mlxServe, kind: kind)
    }

    func testUseResolvesAPickableBaseModelAtThePath() {
        let models = [local("gemma-4-e4b-it-4bit", path: "/m/gemma")]
        XCTAssertEqual(ModelBrowserUse.pickableModel(atPath: "/m/gemma", in: models)?.name, "gemma-4-e4b-it-4bit")
    }

    func testUseIsUnavailableWithoutALocalPath() {
        XCTAssertNil(ModelBrowserUse.pickableModel(atPath: nil, in: [local("g", path: "/m/gemma")]))
    }

    /// A drafter is on disk and deletable, but it is never the server's chat
    /// model — offering "Use" on it would load a checkpoint that can't serve.
    func testUseIsUnavailableForADrafter() {
        let models = [local("gemma-4-e4b-it-assistant-bf16", path: "/m/draft", kind: .drafter, type: "gemma4_assistant")]
        XCTAssertNil(ModelBrowserUse.pickableModel(atPath: "/m/draft", in: models))
    }

    func testUseIsUnavailableForAnEncoderOrMediaModel() {
        let bert = [local("bge-small", path: "/m/bge", type: "bert")]
        XCTAssertNil(ModelBrowserUse.pickableModel(atPath: "/m/bge", in: bert))

        let flux = [local("flux", path: "/m/flux", type: "flux2")]
        XCTAssertNil(ModelBrowserUse.pickableModel(atPath: "/m/flux", in: flux))
    }

    func testUseToleratesTrailingSlashesAndRelativeComponents() {
        let models = [local("gemma", path: "/m/gemma")]
        XCTAssertNotNil(ModelBrowserUse.pickableModel(atPath: "/m/gemma/", in: models))
        XCTAssertNotNil(ModelBrowserUse.pickableModel(atPath: "/m/./gemma", in: models))
    }

    // MARK: - My Models grouping

    func testMyModelsGroupsBySourceInAStableOrderAndDropsEmptyGroups() {
        let models = [
            local("lmstudio-a", path: "/lm/a"),
            local("mlx-b", path: "/mlx/b"),
        ]
        var lm = models[0]
        lm = LocalModel(id: lm.id, name: lm.name, path: lm.path, sizeFormatted: lm.sizeFormatted,
                        modelType: lm.modelType, source: .lmStudio, kind: .base)

        let groups = ModelBrowserUse.groupedBySource([lm, models[1]], filter: "")
        XCTAssertEqual(groups.map(\.source), [.mlxServe, .lmStudio], "mlx-serve models come first; custom group is absent")
        XCTAssertEqual(groups[0].models.map(\.name), ["mlx-b"])
        XCTAssertEqual(groups[1].models.map(\.name), ["lmstudio-a"])
    }

    /// The old "Downloaded" tab showed only `source == .mlxServe`, so it was a
    /// strict subset of the models the tray picker offers. My Models shows them
    /// all, grouped — the two lists must finally agree.
    func testMyModelsIncludesExternallyDiscoveredModels() {
        let lm = LocalModel(id: "/lm/a", name: "lmstudio-a", path: "/lm/a", sizeFormatted: "1 GB",
                            modelType: "llama", source: .lmStudio, kind: .base)
        let groups = ModelBrowserUse.groupedBySource([lm], filter: "")
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].source, .lmStudio)
    }

    func testMyModelsFilterIsCaseInsensitiveAndDropsNowEmptyGroups() {
        let a = local("Gemma-4-E4B", path: "/m/a")
        let b = LocalModel(id: "/lm/b", name: "Qwen3", path: "/lm/b", sizeFormatted: "1 GB",
                           modelType: "qwen3", source: .lmStudio, kind: .base)

        let groups = ModelBrowserUse.groupedBySource([a, b], filter: "gemma")
        XCTAssertEqual(groups.count, 1)
        XCTAssertEqual(groups[0].source, .mlxServe)
        XCTAssertEqual(groups[0].models.map(\.name), ["Gemma-4-E4B"])
    }

    func testEverySourceHasAGroupTitle() {
        for source in [LocalModelSource.mlxServe, .lmStudio, .custom] {
            XCTAssertFalse(ModelBrowserUse.groupTitle(source).isEmpty, "\(source)")
        }
    }
}
