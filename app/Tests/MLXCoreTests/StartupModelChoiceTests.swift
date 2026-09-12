import XCTest
@testable import MLXCore

/// The launch gate: what auto-start actually starts, and which model — if any —
/// comes up with it.
///
/// The bug this pins (issue #214): "Auto-start on launch" passed `--model`,
/// which the server treats as an eager, blocking load, so one checkbox labelled
/// *start* read tens of gigabytes off disk at login. Splitting the two
/// decisions is only safe if the split itself is checkable — the gate is a
/// single branch in `AppState.init` that nobody can watch run.
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

    /// The whole point of the change: auto-start alone brings the server up
    /// WITHOUT a model. If this ever goes back to `.load`, login is slow again.
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

    /// A pin is a pin: what has been used since must not move it.
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

    /// Resolved at START time, so the same stored preference gives a different
    /// answer as the last-used model moves. Nothing about the stored value
    /// changes between these two cases.
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

    /// The mode wins over a pin that happens to still be stored — the pinned
    /// path is inert data under `.lastUsed`, not a fallback.
    func testLastUsedModeIgnoresAStoredPin() {
        XCTAssertEqual(
            StartupModelChoice.resolved(mode: .lastUsed,
                                        pinnedPath: "/models/gemma",
                                        lastUsed: "/models/qwen",
                                        installedPaths: installed),
            "/models/qwen")
    }

    /// No mode is spelled as a path, so no path can be mistaken for a mode.
    /// A raw value that collides with an absolute path would put the sentinel
    /// straight back.
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

    /// Fresh install. Not an error, and emphatically not "pick one for them" —
    /// a startup that loads a model the user never chose is worse than one that
    /// loads none.
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

    /// The model was uninstalled between launches. `--model <gone>` is an
    /// instant FileNotFound, which is the failure this change exists to avoid.
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

    /// A registry id is a directory BASENAME (for a Hugging Face snapshot, a
    /// commit hash) and a LAN id names another Mac's model. Neither can be
    /// handed to `--model`, so neither may become the last model used.
    func testNonPathIdsAreNotRecorded() {
        let d = scratchDefaults()
        StartupModelChoice.recordLoaded(path: "/models/qwen", defaults: d)
        StartupModelChoice.recordLoaded(path: "lan:some-peer-model", defaults: d)
        StartupModelChoice.recordLoaded(path: "a1b2c3d4", defaults: d)
        StartupModelChoice.recordLoaded(path: "", defaults: d)
        XCTAssertEqual(StartupModelChoice.lastUsed(defaults: d), "/models/qwen")
    }

    // MARK: - Seeding the pin

    /// Switching to "Always this model" for the first time opens on what the
    /// other mode was already going to load — the control shows what is about
    /// to happen rather than a blank row.
    func testTheFirstPinSeedsFromTheLastModelUsed() {
        XCTAssertEqual(
            StartupModelChoice.seedPin(lastUsed: "/models/gemma", installedPaths: installed),
            "/models/gemma")
    }

    /// Nothing used yet, or used and since uninstalled: the seed falls to the
    /// library's first model, never to a path that is no longer there.
    func testAPinWithNoAnswerToInheritSeedsFromTheLibrary() {
        XCTAssertEqual(
            StartupModelChoice.seedPin(lastUsed: nil, installedPaths: installed),
            "/models/qwen")
        XCTAssertEqual(
            StartupModelChoice.seedPin(lastUsed: "/models/deleted", installedPaths: installed),
            "/models/qwen")
    }

    /// No chat model on the Mac: empty, which `resolved` reads back as
    /// "nothing pinned" and start reads as headless.
    func testAnEmptyLibrarySeedsNothing() {
        XCTAssertEqual(StartupModelChoice.seedPin(lastUsed: "/models/qwen", installedPaths: []), "")
        XCTAssertNil(StartupModelChoice.resolved(mode: .pinned,
                                                 pinnedPath: "",
                                                 lastUsed: nil,
                                                 installedPaths: []))
    }

    // MARK: - LAN duty at launch

    /// LAN sharing brings a server up even with auto-start off, and that start
    /// must not be the back door this split closed: it loads what the launch
    /// plan asked for, which for a plan that asked for nothing is nothing.
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

    /// The rule above is only worth anything if the launch path asks it. A
    /// bare `ensureServerForLan()` there loads `selectedModelPath` — the eager
    /// login load, reached by a Mac that merely shares its models — and no
    /// unit test of a pure function can see that, so the call site is scanned.
    func testTheLaunchTimeLanStartGoesThroughTheStartupPlan() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe/AppState.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        let launchBlock = try XCTUnwrap(
            source.range(of: "if serverOptions.lanShareEnabled || serverOptions.lanDiscoverEnabled {"),
            "the launch-time LAN block moved — re-point this scan")
        let call = source[launchBlock.upperBound...].prefix(200)
        XCTAssertTrue(call.contains("StartupModelChoice.lanStartPath(plan:"),
                      "launch-time LAN duty must start the server the startup plan asked for")
    }

    // MARK: - The tray's Start button

    /// "Start Server" is the same sentence the auto-start checkbox makes, so it
    /// answers to the same setting. It used to pass the selection
    /// unconditionally: a tray-started server always had a checkpoint resident,
    /// and ejecting it brought it straight back, because `--model` makes that
    /// entry the registry's default.
    func testTrayStartLoadsNothingUnlessTheSettingAsksForIt() {
        XCTAssertFalse(
            StartupModelChoice.trayStartLoadsModel(loadModelAtStart: false,
                                                   selectedModelPath: "/models/qwen"))
        XCTAssertTrue(
            StartupModelChoice.trayStartLoadsModel(loadModelAtStart: true,
                                                   selectedModelPath: "/models/qwen"))
    }

    /// Nothing picked means nothing to load, so the start is plain headless —
    /// never a `--model` with no model, and never a disabled button on a Mac
    /// that has every reason to want a server (media-only, or LAN-only).
    func testTrayStartWithNothingSelectedIsHeadless() {
        XCTAssertFalse(
            StartupModelChoice.trayStartLoadsModel(loadModelAtStart: true, selectedModelPath: ""))
    }

    // MARK: - The chat window's Start button

    /// Every start path now puts a model resident only ON DEMAND, never as the
    /// launch default: a `--model` entry is the registry's default and comes
    /// back on the next request after an eject. The chat toolbar's Start is the
    /// last one that did that, and no unit test of a pure function can see
    /// which method a button calls, so the call site is scanned.
    func testTheChatStartButtonDoesNotLaunchWithAModel() throws {
        let view = try appSource("Views/ChatView.swift")
        let button = try XCTUnwrap(view.range(of: "@ViewBuilder private var serverStartControl"),
                                   "the chat Start control moved — re-point this scan")
        let body = view[button.upperBound...].prefix(1200)
        XCTAssertTrue(body.contains("appState.startServer(loadingSelection: true)"),
                      "chat Start must go through the one button-start path")
        XCTAssertFalse(body.contains("appState.ensureServerForLan()"),
                       "ensureServerForLan passes --model, which pins the model as the launch default")
    }

    /// And that one method is headless-then-hot-load, with the pill spinning:
    /// a start the user pressed owes them a spinner, and `ensureDefaultChatModel`
    /// is the ONE hot-load path (it also records the last model used).
    func testEveryButtonStartIsHeadlessThenAnOnDemandLoad() throws {
        let state = try appSource("AppState.swift")
        let fn = try XCTUnwrap(state.range(of: "func startServer(loadingSelection: Bool)"),
                               "startServer moved — re-point this scan")
        let body = state[fn.upperBound...].prefix(900)
        XCTAssertTrue(body.contains("server.startHeadless("), "the start itself must be headless")
        XCTAssertTrue(body.contains("ensureDefaultChatModel("), "the load must go through the one hot-load path")
        XCTAssertTrue(body.contains("loadingModelPath = path"), "the pill must name and spin on the loading model")
        XCTAssertFalse(body.contains("server.start(modelPath:"), "a --model start pins the launch default")
    }

    private func appSource(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe")
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }
}
