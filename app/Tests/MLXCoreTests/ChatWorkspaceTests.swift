import XCTest
@testable import MLXCore

/// The Model Browser moved from its own `Window` into the chat window's detail
/// column. These pin the three things that make that safe: the gate can't cover
/// the browser, every entry point goes through one chokepoint, and the retired
/// window id is gone from every surface that used to open it.
final class ChatWorkspaceTests: XCTestCase {

    private func source(_ relativePath: String) throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent(relativePath)
        return try String(contentsOf: url, encoding: .utf8)
    }

    // MARK: - The gate must not cover its own cure

    /// `ChatModelGateSheet` blocks the whole window and its only door closes it.
    /// With the browser inside that window, presenting the gate over the models
    /// pane locks the user out of the exact screen that resolves the gate.
    func testTheModelGateStandsDownWhileTheModelsPaneIsShowing() {
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
                                            workspace: .models(.recommended)),
            "the gate must not cover the browser it is asking the user to use")
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
                                            workspace: .models(.discover)),
            "…on any section")
    }

    /// And it is deferred, not dismissed: back in a conversation with still no
    /// model, it presents again.
    func testTheGateReturnsInConversationMode() {
        XCTAssertTrue(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
                                            workspace: .conversation))
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: false, cancelled: false,
                                            workspace: .conversation))
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: true,
                                            workspace: .conversation),
            "Cancel still wins — it is the sheet's one door")
    }

    func testEntryLandsOnTheRecommendedSection() {
        XCTAssertEqual(ChatWorkspace.defaultEntry, .models(.recommended))
        XCTAssertTrue(ChatWorkspace.defaultEntry.isModels)
        XCTAssertEqual(ChatWorkspace.conversation.section, nil)
    }

    // MARK: - One chokepoint in

    /// Five surfaces used to call `openAndFocus("modelBrowser")` on their own.
    /// They all go through `AppState.showModels(...)` now — which both opens the
    /// chat window and sets the mode. A surface that set the mode itself would
    /// switch a window nobody is looking at.
    func testOnlyAppStateSwitchesIntoTheModelsPane() throws {
        let appState = try source("Sources/MLXServe/AppState.swift")
        XCTAssertTrue(appState.contains("func showModels("),
                      "AppState owns the one way into the models pane")

        for path in ["Sources/MLXServe/Views/ChatView.swift",
                     "Sources/MLXServe/Views/ChatModelPill.swift",
                     "Sources/MLXServe/Views/WelcomeView.swift",
                     "Sources/MLXServe/MLXServeApp.swift"] {
            let text = try source(path)
            XCTAssertFalse(text.contains("chatWorkspace = .models"), """
                \(path) sets the workspace directly — go through \
                AppState.showModels(), which also brings the window forward.
                """)
        }
    }

    // MARK: - The retired window

    /// A scene id left behind after its `Window` is deleted is a control that
    /// opens nothing: `openWindow(id:)` on an unknown id is a no-op with no
    /// error anywhere.
    func testTheModelBrowserWindowIsGoneFromEverySurface() throws {
        for path in ["Sources/MLXServe/MLXServeApp.swift",
                     "Sources/MLXServe/Views/ChatEmptyState.swift",
                     "Sources/MLXServe/Views/ChatModelPill.swift",
                     "Sources/MLXServe/Services/AppActivation.swift"] {
            let text = try source(path)
            // The OPENING spellings only: a chip may still be identified as
            // "tasks" (its own id), it just must not open a window with it.
            for opener in ["window(\"tasks\")", "openAndFocus(\"tasks\")", "id: \"tasks\")"] {
                XCTAssertFalse(text.contains(opener), """
                    \(path) still opens the retired "tasks" window (\(opener)) — \
                    Tasks is a mode of the chat window now.
                    """)
            }
            XCTAssertFalse(text.contains("\"modelBrowser\""), """
                \(path) still references the retired "modelBrowser" window id — \
                the browser is a mode of the chat window now.
                """)
        }
    }

    /// Moving a view into another window means moving its ENVIRONMENT with it,
    /// and SwiftUI reports a missing `@EnvironmentObject` as a runtime trap —
    /// no compile error, nothing at all until the view first renders. Live
    /// crash 2026-08-08: the browser's window injected four objects, the chat
    /// window three of them, and opening the models pane killed the app inside
    /// `ModelBrowserPane.downloads.getter`.
    ///
    /// The map is explicit so a NEW `@EnvironmentObject` in the browser fails
    /// here — with the name of the type to inject — rather than at runtime.
    func testTheChatWindowInjectsEveryObjectItsHostedPanesRead() throws {
        let expectedInjection: [String: String] = [
            "AppState": ".environmentObject(appState)",
            "ServerManager": ".environmentObject(appState.server)",
            "DownloadManager": ".environmentObject(appState.downloads)",
            "HFSearchService": ".environmentObject(hfSearch)",
            "ImageGenService": ".environmentObject(appState.imageGen)",
            "VideoGenService": ".environmentObject(appState.videoGen)",
            "AudioGenService": ".environmentObject(appState.audioGen)",
            "MusicGenService": ".environmentObject(appState.musicGen)",
            "Model3DGenService": ".environmentObject(appState.model3dGen)",
        ]

        // Every view the chat window hosts as a MODE — the browser and the four
        // media generators. All five were windows of their own, with their own
        // environments; the chat window inherited that obligation.
        let hosted = ["Sources/MLXServe/Views/ModelBrowserView.swift",
                      "Sources/MLXServe/Views/ImageGenView.swift",
                      "Sources/MLXServe/Views/VideoGenView.swift",
                      "Sources/MLXServe/Views/AudioGenView.swift",
                      "Sources/MLXServe/Views/Model3DGenView.swift"]
        let pattern = try NSRegularExpression(pattern: #"@EnvironmentObject\s+var\s+\w+\s*:\s*(\w+)"#)
        var types = Set<String>()
        for path in hosted {
            let text = try source(path)
            let range = NSRange(text.startIndex..., in: text)
            for match in pattern.matches(in: text, range: range) {
                if let r = Range(match.range(at: 1), in: text) {
                    types.insert(String(text[r]))
                }
            }
        }
        XCTAssertFalse(types.isEmpty, "the regex stopped matching — fix the audit, not the app")

        let app = try source("Sources/MLXServe/MLXServeApp.swift")
        guard let chatScene = app.range(of: #"Window("MLX Core", id: "chat")"#),
              let nextScene = app.range(of: "Window(", range: chatScene.upperBound..<app.endIndex) else {
            return XCTFail("the chat Window scene moved — update this audit")
        }
        let scene = String(app[chatScene.lowerBound..<nextScene.lowerBound])

        for type in types.sorted() {
            guard let injection = expectedInjection[type] else {
                XCTFail("""
                    A pane hosted by the chat window declares @EnvironmentObject \
                    of type \(type), which this audit doesn't know how to inject. \
                    Add it to `expectedInjection` AND to the chat Window scene — \
                    a missing one is a crash at first render, not a build error.
                    """)
                continue
            }
            XCTAssertTrue(scene.contains(injection), """
                The chat window must inject \(type) (`\(injection)`) — a pane it \
                hosts reads it, and SwiftUI traps at render time when it is \
                absent.
                """)
        }
    }


    /// The sidebar is a list of DESTINATIONS above the conversation list, and
    /// selecting one changes only the content area — the panel itself never
    /// rearranges, so the places stay where the eye learned them.
    ///
    /// The route in and the route back are the same row, tinted while its pane
    /// is up. That matters most for the entries that open this window ALREADY
    /// in a pane (the tray, the welcome screen, the Tools menu, a tapped task
    /// notification): they arrive with nothing to have watched.
    func testTheSidebarListsEveryDestinationAboveTheConversations() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        for row in ["New Chat", "Agents", "Tasks", "Code Launcher", "Models", "Settings"] {
            XCTAssertTrue(chat.contains("\"\(row)\""), "the sidebar is missing the \(row) destination")
        }
        XCTAssertTrue(chat.contains("Text(\"Recent\")"),
                      "the conversation list needs its heading")
        // Pinned above the list, so no destination scrolls away.
        guard let inset = chat.range(of: "safeAreaInset(edge: .top)"),
              let rows = chat.range(of: "destinationRow(", range: inset.upperBound..<chat.endIndex) else {
            return XCTFail("the destinations must ride the sidebar's top inset")
        }
        XCTAssertLessThan(inset.lowerBound, rows.lowerBound)
        // Both directions from the same row.
        XCTAssertTrue(chat.contains("appState.showConversation()"))
        XCTAssertTrue(chat.contains("appState.showModels()"))
        XCTAssertTrue(chat.contains("appState.showTasks()"))
        // The switcher these replaced is gone, not left as a second route.
        XCTAssertFalse(chat.contains("SidebarModeSwitcher"))
    }

    /// The browser's sub-items live across the top of the CONTENT area, because
    /// the sidebar is the conversation list. `allCases`, never a hand-written
    /// array — that is where a section quietly goes missing.
    func testTheBrowserCarriesItsOwnSectionBar() throws {
        let browser = try source("Sources/MLXServe/Views/ModelBrowserView.swift")
        XCTAssertTrue(browser.contains("ForEach(ModelBrowserSection.allCases)"),
                      "the section bar must iterate the whole catalogue")
        XCTAssertFalse(browser.contains("NavigationSplitView {"), """
            The pane renders inside the chat window's own NavigationSplitView — \
            a nested split view is what the section bar replaced.
            """)
    }

    /// Create mode lost its sidebar page list when the Chats/Models/Create
    /// switcher went — it is reached from the chat's discovery chips and the
    /// tray now. What must still hold is that every generator in the shared
    /// catalogue has a page to land on: `showCreate(.video)` with no video case
    /// would switch the window to a blank column.
    func testEveryGeneratorStillHasAPage() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        for view in ["ImageGenView()", "VideoGenView()", "AudioGenView()", "Model3DGenView()"] {
            XCTAssertTrue(chat.contains(view), "create mode must host \(view)")
        }
        XCTAssertEqual(GenExperiment.allCases.count, 4,
                       "a fifth generator needs a case in ChatDetailView.createPane")
    }
}
