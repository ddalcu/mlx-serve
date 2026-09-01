import XCTest
@testable import MLXCore

/// Guards for Settings ▸ Interface wiring — the two places a display pref can
/// silently stop applying: a window scene that forgets `.appAppearance()`,
/// and the transcript, whose metrics read UserDefaults with no SwiftUI
/// dependency of their own.
final class AppearanceSettingsTests: XCTestCase {
    private func source(_ relative: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try String(contentsOf: root.appendingPathComponent(relative), encoding: .utf8)
    }

    /// `.appAppearance()` is applied per scene BY HAND, so a new `Window` can
    /// forget it and ship a window that ignores Settings ▸ Interface — the
    /// same class as the window-injection rules.
    func testEveryWindowSceneAppliesTheAppAppearance() throws {
        let text = try source("Sources/MLXServe/MLXServeApp.swift")
        let scenes = text.components(separatedBy: "Window(\"").dropFirst()
        XCTAssertGreaterThanOrEqual(scenes.count, 5, "expected the app's window scenes to be found")
        for scene in scenes {
            let name = scene.prefix(while: { $0 != "\"" })
            XCTAssertTrue(scene.contains(".appAppearance()"),
                          "the \(name) window scene does not apply .appAppearance()")
        }
    }

    /// `ChatMetrics` reads the text-size and compact-mode keys straight off
    /// UserDefaults — no SwiftUI dependency — so a Settings change re-renders
    /// the transcript only because `ChatDetailView` observes the same keys
    /// via `@AppStorage` and re-ids the row stack. Without that, old rows
    /// keep the old font while new rows get the new one.
    func testTheTranscriptObservesTheInterfaceKeysItRendersWith() throws {
        let text = try source("Sources/MLXServe/Views/ChatView.swift")
        XCTAssertTrue(text.contains("@AppStorage(InterfacePrefKey.textSize)"),
                      "ChatDetailView does not observe the text-size key")
        XCTAssertTrue(text.contains("@AppStorage(InterfacePrefKey.compactMode)"),
                      "ChatDetailView does not observe the compact-mode key")
    }
}
