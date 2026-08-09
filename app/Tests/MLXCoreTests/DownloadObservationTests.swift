import XCTest
@testable import MLXCore

/// AppState forwards `objectWillChange` for the server and the agent store —
/// NOT for DownloadManager (progress publishes at ~4 Hz would re-render every
/// AppState observer for the length of a transfer). So a view that reads
/// download state through `appState.downloads` observes nothing: the sidebar's
/// Models badge never appeared while a transfer ran, and the create banner's
/// "not downloaded" pill lingered after the bytes landed. Readers declare
/// `@EnvironmentObject var downloads: DownloadManager` instead — the chat
/// window injects it (`testTheChatWindowInjectsEveryObjectItsHostedPanesRead`).
final class DownloadObservationTests: XCTestCase {

    func testViewsNeverReadDownloadStateThroughAppState() throws {
        let viewsDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe/Views")
        let files = try FileManager.default.contentsOfDirectory(at: viewsDir,
                                                                includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "swift" }
        XCTAssertGreaterThan(files.count, 10, "the Views directory moved?")

        for file in files {
            let source = try String(contentsOf: file, encoding: .utf8)
            for (idx, line) in source.components(separatedBy: "\n").enumerated()
            where line.contains("appState.downloads")
                && !line.trimmingCharacters(in: .whitespaces).hasPrefix("//") {
                // Handing the OBJECT to the environment is the fix, not the bug.
                XCTAssertTrue(line.contains(".environmentObject(appState.downloads)"), """
                    \(file.lastPathComponent):\(idx + 1) reads download state through \
                    appState, which does not forward DownloadManager's publishes — \
                    declare `@EnvironmentObject var downloads: DownloadManager` and \
                    read that instead.
                    """)
            }
        }
    }
}
