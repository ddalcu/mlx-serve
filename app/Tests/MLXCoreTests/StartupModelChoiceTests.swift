import XCTest
@testable import MLXCore

/// The launch gate: what auto-start actually starts.
///
/// The bug this pins (issue #214): "Auto-start on launch" passed `--model`,
/// which the server treats as an eager, blocking load, so one checkbox labelled
/// *start* read tens of gigabytes off disk at login. Splitting the two
/// decisions is only safe if the split itself is checkable — the gate is a
/// single branch in `AppState.init` that nobody can watch run.
final class StartupModelChoiceTests: XCTestCase {

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
                                      choice: "/models/qwen",
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
                                      choice: "/models/qwen",
                                      lastUsed: "/models/gemma",
                                      installedPaths: installed),
            .headless)
    }

    func testLoadAtStartWithAnExplicitPickLoadsThatModel() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      choice: "/models/gemma",
                                      lastUsed: "/models/qwen",
                                      installedPaths: installed),
            .load(path: "/models/gemma"))
    }

    // MARK: - "Last model used"

    /// Resolved at START time, not when the setting was saved — the saved value
    /// is the sentinel, so a different last-used model gives a different answer
    /// from the same stored preference.
    func testLastUsedResolvesAtStartTime() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      choice: StartupModelChoice.lastUsedTag,
                                      lastUsed: "/models/qwen",
                                      installedPaths: installed),
            .load(path: "/models/qwen"))
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      choice: StartupModelChoice.lastUsedTag,
                                      lastUsed: "/models/gemma",
                                      installedPaths: installed),
            .load(path: "/models/gemma"))
    }

    /// Fresh install. Not an error, and emphatically not "pick one for them" —
    /// a startup that loads a model the user never chose is worse than one that
    /// loads none.
    func testNoLastUsedStartsHeadlessRatherThanPickingSomething() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      choice: StartupModelChoice.lastUsedTag,
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
                                      choice: StartupModelChoice.lastUsedTag,
                                      lastUsed: "/models/deleted",
                                      installedPaths: installed),
            .headless)
    }

    func testUninstalledExplicitPickStartsHeadless() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      choice: "/models/deleted",
                                      lastUsed: "/models/qwen",
                                      installedPaths: installed),
            .headless)
    }

    /// A Mac with nothing chat-pickable at all.
    func testEmptyLibraryStartsHeadless() {
        XCTAssertEqual(
            StartupModelChoice.launch(autoStart: true,
                                      loadModelAtStart: true,
                                      choice: StartupModelChoice.lastUsedTag,
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
}
