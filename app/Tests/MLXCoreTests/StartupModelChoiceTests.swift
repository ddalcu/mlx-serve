import XCTest
@testable import MLXCore

/// The launch gate: whether auto-start loads a model, and which one.
final class StartupModelChoiceTests: XCTestCase {

    private typealias Launch = StartupModelChoice.Launch

    private let installed = ["/models/qwen", "/models/gemma"]

    private func scratchDefaults(_ name: String = #function) -> UserDefaults {
        let suite = "StartupModelChoiceTests.\(name)"
        UserDefaults().removePersistentDomain(forName: suite)
        return UserDefaults(suiteName: suite)!
    }

    // MARK: - Start the server vs load a model

    func testAutoStartOffStartsNothing() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: false,
                                      loadModelAtStart: true,
                                      mode: .pinned,
                                      pinnedPath: "/models/qwen",
                                      lastUsed: "/models/qwen",
                                      installedPaths: installed),
            .doNothing)
    }

    /// Auto-start alone brings the server up with no model.
    func testAutoStartAloneIsHeadless() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: false,
                                      mode: .pinned,
                                      pinnedPath: "/models/qwen",
                                      lastUsed: "/models/gemma",
                                      installedPaths: installed),
            .headless)
    }

    func testPinnedModeLoadsThePinnedModel() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      mode: .pinned,
                                      pinnedPath: "/models/gemma",
                                      lastUsed: "/models/qwen",
                                      installedPaths: installed),
            .load(path: "/models/gemma"))
    }

    /// What has been used since must not move a pin.
    func testPinnedModeIgnoresTheLastUsedModel() {
        XCTAssertEqual(
            StartupModelChoice.resolved(mode: .pinned,
                                        pinnedPath: "/models/gemma",
                                        lastUsed: "/models/qwen",
                                        installedPaths: installed),
            "/models/gemma")
    }

    /// "Always this model" selected before anything was pinned.
    func testPinnedModeWithNothingPinnedResolvesToNothing() {
        XCTAssertNil(
            StartupModelChoice.resolved(mode: .pinned,
                                        pinnedPath: "",
                                        lastUsed: "/models/qwen",
                                        installedPaths: installed))
    }

    // MARK: - "Last model used" is a MODE, not a magic path

    /// The same stored preference follows the last-used model at start time.
    func testLastUsedModeResolvesAtStartTime() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      mode: .lastUsed,
                                      pinnedPath: nil,
                                      lastUsed: "/models/qwen",
                                      installedPaths: installed),
            .load(path: "/models/qwen"))
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      mode: .lastUsed,
                                      pinnedPath: nil,
                                      lastUsed: "/models/gemma",
                                      installedPaths: installed),
            .load(path: "/models/gemma"))
    }

    /// A stored pin is inert under `.lastUsed`, not a fallback.
    func testLastUsedModeIgnoresAStoredPin() {
        XCTAssertEqual(
            StartupModelChoice.resolved(mode: .lastUsed,
                                        pinnedPath: "/models/gemma",
                                        lastUsed: "/models/qwen",
                                        installedPaths: installed),
            "/models/qwen")
    }

    /// No raw value can be read as a path or as an absent value.
    func testNoModeIsSpelledAsAPath() {
        for mode in StartupModelChoice.Mode.allCases {
            XCTAssertFalse(mode.rawValue.hasPrefix("/"), "\(mode) reads as a path")
            XCTAssertFalse(mode.rawValue.isEmpty, "\(mode) reads as an absent value")
        }
    }

    func testTheDefaultModeIsLastUsed() {
        XCTAssertEqual(StartupModelChoice.Mode.default, .lastUsed)
    }

    // MARK: - Nothing to load

    /// A fresh install starts headless rather than picking a model for the user.
    func testNoLastUsedStartsHeadlessRatherThanPickingSomething() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      mode: .lastUsed,
                                      pinnedPath: nil,
                                      lastUsed: nil,
                                      installedPaths: installed),
            .headless)
    }

    /// An uninstalled model never reaches `--model`.
    func testUninstalledLastUsedStartsHeadless() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      mode: .lastUsed,
                                      pinnedPath: nil,
                                      lastUsed: "/models/deleted",
                                      installedPaths: installed),
            .headless)
    }

    func testUninstalledPinStartsHeadless() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      mode: .pinned,
                                      pinnedPath: "/models/deleted",
                                      lastUsed: "/models/qwen",
                                      installedPaths: installed),
            .headless)
    }

    /// A Mac with nothing chat-pickable at all.
    func testEmptyLibraryStartsHeadless() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      mode: .lastUsed,
                                      pinnedPath: nil,
                                      lastUsed: "/models/qwen",
                                      installedPaths: []),
            .headless)
    }

    // MARK: - Recording the last model used

    func testNothingRecordedYetReadsAsNil() {
        XCTAssertNil(StartupModelChoice.lastUsed(defaults: scratchDefaults()))
    }

    func testRecordedLoadIsReadBack() {
        let d = scratchDefaults()
        StartupModelChoice.recordLoaded(path: "/models/qwen", defaults: d)
        XCTAssertEqual(StartupModelChoice.lastUsed(defaults: d), "/models/qwen")
    }

    func testTheMostRecentLoadWins() {
        let d = scratchDefaults()
        StartupModelChoice.recordLoaded(path: "/models/qwen", defaults: d)
        StartupModelChoice.recordLoaded(path: "/models/gemma", defaults: d)
        XCTAssertEqual(StartupModelChoice.lastUsed(defaults: d), "/models/gemma")
    }

    /// Registry and LAN ids cannot be handed to `--model`, so only absolute paths record.
    func testNonPathIdsAreNotRecorded() {
        let d = scratchDefaults()
        StartupModelChoice.recordLoaded(path: "/models/qwen", defaults: d)
        StartupModelChoice.recordLoaded(path: "lan:some-peer-model", defaults: d)
        StartupModelChoice.recordLoaded(path: "a1b2c3d4", defaults: d)
        StartupModelChoice.recordLoaded(path: "", defaults: d)
        XCTAssertEqual(StartupModelChoice.lastUsed(defaults: d), "/models/qwen")
    }

    // MARK: - Seeding the pin

    /// A first pin opens on what `.lastUsed` would have loaded.
    func testTheFirstPinSeedsFromTheLastModelUsed() {
        XCTAssertEqual(
            StartupModelChoice.seedPin(lastUsed: "/models/gemma", installedPaths: installed),
            "/models/gemma")
    }

    /// With no usable last-used model the seed is the library's first model, never a missing path.
    func testAPinWithNoAnswerToInheritSeedsFromTheLibrary() {
        XCTAssertEqual(
            StartupModelChoice.seedPin(lastUsed: nil, installedPaths: installed),
            "/models/qwen")
        XCTAssertEqual(
            StartupModelChoice.seedPin(lastUsed: "/models/deleted", installedPaths: installed),
            "/models/qwen")
    }

    /// No chat model on the Mac seeds nothing, which resolves to headless.
    func testAnEmptyLibrarySeedsNothing() {
        XCTAssertEqual(StartupModelChoice.seedPin(lastUsed: "/models/qwen", installedPaths: []), "")
        XCTAssertNil(StartupModelChoice.resolved(mode: .pinned,
                                                 pinnedPath: "",
                                                 lastUsed: nil,
                                                 installedPaths: []))
    }

    // MARK: - LAN duty at launch

    /// LAN duty at launch loads only what the plan chose.
    func testLanDutyAtLaunchLoadsNothingUnlessTheStartupChoiceAskedFor() {
        XCTAssertEqual(StartupModelChoice.lanStartPath(plan: .doNothing), "")
        XCTAssertEqual(StartupModelChoice.lanStartPath(plan: .headless), "")
        XCTAssertEqual(StartupModelChoice.lanStartPath(plan: .load(path: "/models/qwen")),
                       "/models/qwen")
    }

    func testAPlanNamesTheModelItLoads() {
        XCTAssertNil(Launch.doNothing.modelPath)
        XCTAssertNil(Launch.headless.modelPath)
        XCTAssertEqual(Launch.load(path: "/models/gemma").modelPath, "/models/gemma")
    }

    // MARK: - The tray's Start button

    /// The tray's Start loads the selection only when "Load a model at start" is on.
    func testTrayStartLoadsNothingUnlessTheSettingAsksForIt() {
        XCTAssertFalse(
            StartupModelChoice.trayStartLoadsModel(loadModelAtStart: false,
                                                   selectedModelPath: "/models/qwen"))
        XCTAssertTrue(
            StartupModelChoice.trayStartLoadsModel(loadModelAtStart: true,
                                                   selectedModelPath: "/models/qwen"))
    }

    /// Nothing selected means a plain headless start.
    func testTrayStartWithNothingSelectedIsHeadless() {
        XCTAssertFalse(
            StartupModelChoice.trayStartLoadsModel(loadModelAtStart: true, selectedModelPath: ""))
    }
}
