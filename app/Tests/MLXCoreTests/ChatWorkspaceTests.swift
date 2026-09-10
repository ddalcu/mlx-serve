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
                                            workspace: .models(.recommended), welcomePresented: false),
            "the gate must not cover the browser it is asking the user to use")
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
                                            workspace: .models(.discover), welcomePresented: false),
            "…on any section")
    }

    /// And it is deferred, not dismissed: back in a conversation with still no
    /// model, it presents again.
    func testTheGateReturnsInConversationMode() {
        XCTAssertTrue(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
                                            workspace: .conversation, welcomePresented: false))
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: false, cancelled: false,
                                            workspace: .conversation, welcomePresented: false))
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: true,
                                            workspace: .conversation, welcomePresented: false),
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

    /// The tray's Chat button lands on the TRANSCRIPT, not merely on the
    /// window: with the window parked on Models/Tasks/Create, a Chat that only
    /// focuses leaves the user staring at the pane they left (live report
    /// 2026-08-09). Same door shape as showModels() — mode AND window.
    func testTheTrayChatButtonSwitchesBackToTheConversation() throws {
        let appState = try source("Sources/MLXServe/AppState.swift")
        XCTAssertTrue(appState.contains("func showChat("),
                      "AppState owns the one door into the conversation")
        let app = try source("Sources/MLXServe/MLXServeApp.swift")
        XCTAssertTrue(app.contains("openChat: { appState.showChat() }"),
                      "the tray's Chat button must switch the mode, not just focus the window")
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
            XCTAssertFalse(text.contains("\"sandboxTerminal\""), """
                \(path) still references the retired "sandboxTerminal" window id — \
                terminals are rows of the chat window now.
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
            "TaskScheduler": ".environmentObject(appState.taskScheduler)",
            "AgentStore": ".environmentObject(appState.agents)",
            "TerminalSessionStore": ".environmentObject(appState.terminals)",
        ]

        // Every view the chat window hosts as a MODE — the browser, the four
        // media generators, and Tasks (whose two panes are columns of the
        // window's own split). All were windows of their own, with their own
        // environments; the chat window inherited that obligation.
        let hosted = ["Sources/MLXServe/Views/ModelBrowserView.swift",
                      "Sources/MLXServe/Views/ImageGenView.swift",
                      "Sources/MLXServe/Views/VideoGenView.swift",
                      "Sources/MLXServe/Views/AudioGenView.swift",
                      "Sources/MLXServe/Views/Model3DGenView.swift",
                      "Sources/MLXServe/Views/TasksView.swift",
                      // Agents is a mode now too, and its panes read a store
                      // the chat scene did not inject — the same first-render
                      // trap the Tasks columns hit, invisible to this audit for
                      // exactly as long as the file was missing from this list.
                      "Sources/MLXServe/Views/AgentsWindow.swift",
                      "Sources/MLXServe/Views/TerminalPane.swift"]
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

    /// A column of a split view must be a view TYPE, never a computed property
    /// read off a synthetic instance.
    ///
    /// Live crash 2026-08-08: the Tasks columns were `TasksView().taskList` and
    /// `TasksView().taskDetail`. That builds a view VALUE and immediately
    /// evaluates a property which touches `@EnvironmentObject` — but SwiftUI
    /// fills that storage when it INSTALLS a view in the hierarchy, and this
    /// instance never was one, so the first click on Tasks trapped in
    /// `TasksView.$appState.getter`. The `.environmentObject(…)` chained at the
    /// call site cannot help: it decorates the view the property already
    /// returned, long after the property read the empty box.
    ///
    /// The previous audit could not see it — it checks that the WINDOW injects
    /// what its panes read, which was true the whole time. What was wrong is
    /// where the reading happened.
    func testSplitViewColumnsAreViewTypesNotPropertiesOfSyntheticInstances() throws {
        // Comments describe this bug, so scanning them finds it in the prose
        // that warns about it.
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces).hasPrefix("//") ? "" : String($0) }
            .joined(separator: "\n")

        // `Foo(…).bar` where `bar` is NOT a call. A view MODIFIER is always
        // `.foo(…)`, so requiring the absence of `(` separates the legitimate
        // shape from the crashing one without an allowlist to keep up to date.
        let pattern = try NSRegularExpression(
            pattern: #"\b([A-Z]\w*(?:View|Pane))\s*\([^)]*\)\s*\.\s*([a-z]\w*)\s*(?![\(\w])"#)
        let range = NSRange(chat.startIndex..., in: chat)
        let offenders: [String] = pattern.matches(in: chat, range: range).compactMap { match in
            Range(match.range, in: chat).map { String(chat[$0]) }
        }
        XCTAssertTrue(offenders.isEmpty, """
            A split-view column reads a property off a freshly constructed view: \
            \(offenders). SwiftUI never installed that instance, so any \
            @EnvironmentObject it touches traps at first render. Give the column \
            its own View type (`TaskListPane` / `TaskDetailPane`) instead.
            """)
    }

    /// `NavigationSplitViewVisibility` is interpreted against the split view's
    /// COLUMN COUNT, so one state shared by a two-column and a three-column
    /// split means different things in each. `.doubleColumn` — which SwiftUI
    /// resolves to on its own for an ordinary two-column window — is "sidebar +
    /// detail" in the chat view and "content + detail, SIDEBAR HIDDEN" in the
    /// three-column Tasks view. Sharing one value ate the top-level sidebar as
    /// soon as Tasks was opened from a chat window that had been shown once; a
    /// fresh launch was still on `.automatic` and looked right, which is what
    /// made it read as intermittent rather than as a wiring bug.
    func testEachSplitViewOwnsItsColumnVisibilityState() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        let pattern = try NSRegularExpression(pattern: #"NavigationSplitView\(columnVisibility:\s*\$(\w+)\)"#)
        let range = NSRange(chat.startIndex..., in: chat)
        let bindings: [String] = pattern.matches(in: chat, range: range).compactMap { match in
            Range(match.range(at: 1), in: chat).map { String(chat[$0]) }
        }
        XCTAssertEqual(bindings.count, 2, "expected the two split views — update this audit")
        XCTAssertEqual(Set(bindings).count, bindings.count, """
            The two NavigationSplitViews share one columnVisibility state \
            (\(bindings)). That value is read against each split's COLUMN COUNT: \
            .doubleColumn is "sidebar + detail" in the two-column chat view and \
            "content + detail, sidebar hidden" in the three-column Tasks view, so \
            sharing it hides the top-level sidebar in Tasks.
            """)
        // …and the three-column one starts on the only value that shows three.
        XCTAssertTrue(chat.contains("tasksColumnVisibility = NavigationSplitViewVisibility.all"), """
            The Tasks split must start at `.all` — `.automatic` only happens to \
            show three columns, and no other value can.
            """)
    }

    /// The Tasks column's title row draws no rule under itself.
    ///
    /// The rule was never an explicit `Divider` — it came from `.background(.bar)`
    /// on a `safeAreaInset`, and a `.bar` material draws a separator along its
    /// edge. The backdrop existed only so rows didn't scroll through the title,
    /// which a plain sibling row above the list doesn't need. Searching for
    /// "Divider" would have found nothing and cleared a header that had one.
    func testTheTasksHeaderDrawsNoRuleUnderItself() throws {
        let file = try source("Sources/MLXServe/Views/TasksView.swift")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces).hasPrefix("//") ? "" : String($0) }
            .joined(separator: "\n")
        // Scoped to the list pane: the DETAIL column has a legitimate divider
        // between a task's header and its run history, and a file-wide ban
        // would forbid it.
        guard let start = file.range(of: "struct TaskListPane"),
              let end = file.range(of: "struct TaskDetailPane",
                                   range: start.upperBound..<file.endIndex) else {
            return XCTFail("the Tasks panes moved — update this audit")
        }
        let tasks = String(file[start.upperBound..<end.lowerBound])

        XCTAssertFalse(tasks.contains(".background(.bar)"), """
            The Tasks header is back on a `.bar` backdrop, which draws the \
            separator this removed. A header that is a sibling of the list needs \
            no backdrop at all.
            """)
        XCTAssertFalse(tasks.contains("safeAreaInset(edge: .top)"), """
            The header is back to overlaying the list, which is what forced an \
            opaque backdrop (and therefore a rule) in the first place.
            """)
        XCTAssertFalse(tasks.contains("Divider()"),
                       "no explicit rule under the Tasks title either")
        // The title and the + moved INTO the toolbar once the chat's own
        // cluster was gone and the band was free. Both are static, which is why
        // they are safe there — the `»`-eviction rule is about members that
        // appear or change width at runtime.
        XCTAssertTrue(tasks.contains("\"Tasks\""), "the column still needs its title")
        XCTAssertTrue(tasks.contains(".toolbar {"), "title and + ride the toolbar now")
        // The + sits beside the title in ONE leading item, and draws no fill of
        // its own: a ToolbarItem on macOS 26 draws a capsule around whatever it
        // holds, so a button that also draws one is a box inside a box (the
        // class the chat's old cluster answered with
        // `.sharedBackgroundVisibility(.hidden)`).
        XCTAssertTrue(tasks.contains("ToolbarItem(placement: .navigation)"),
                      "title and + ride one leading item, so the + is next to the word")
        // …and the platform's own capsule is suppressed, or it wraps the title
        // AND the button into one "Tasks +" lozenge.
        XCTAssertTrue(tasks.contains(".sharedBackgroundVisibility(.hidden)"), """
            The toolbar item must drop its shared background: on macOS 26 that \
            capsule wraps everything the item holds, so a title beside a button \
            renders as one lozenge around both.
            """)
        XCTAssertTrue(tasks.contains("paneTitle("),
                      "Tasks and Agents share one title-bar shape (`PaneTitleBar`)")
        XCTAssertFalse(tasks.contains(".buttonStyle(.borderless)"), """
            `.borderless` gives a bare glyph with nothing to aim at — inside a \
            toolbar item the capsule is the target, which is why `.plain` is \
            right HERE and was wrong in the pane header.
            """)
    }

    /// Both Tasks columns are real view types, each declaring the environment it
    /// reads — which is what makes the window's injection reach them.
    func testTasksColumnsAreTheirOwnPaneTypes() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        XCTAssertTrue(chat.contains("TaskListPane()"), "the content column must be TaskListPane")
        XCTAssertTrue(chat.contains("TaskDetailPane()"), "the detail column must be TaskDetailPane")
        XCTAssertFalse(chat.contains("TasksView()"), """
            TasksView() is back as a synthetic instance — that is the 2026-08-08 \
            crash shape.
            """)

        let tasks = try source("Sources/MLXServe/Views/TasksView.swift")
        for pane in ["struct TaskListPane: View", "struct TaskDetailPane: View"] {
            XCTAssertTrue(tasks.contains(pane), "missing \(pane)")
        }
    }

    /// One meaning for gray in the sidebar means one SHAPE as well as one
    /// colour. A conversation's fill has to ride the row CONTENT: as a
    /// `listRowBackground` it filled the entire row rect — that modifier is the
    /// row's backdrop, and the `listRowInsets` beside it move only the content —
    /// so a selected chat ran edge to edge beneath destinations inset 8pt.
    func testTheConversationHighlightIsInsetLikeADestinationRow() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        guard let list = chat.range(of: "private var conversationsSidebar"),
              let end = chat.range(of: "private func destinationRow",
                                   range: list.upperBound..<chat.endIndex) else {
            return XCTFail("the sidebar's conversation list moved — update this audit")
        }
        let body = String(chat[list.upperBound..<end.lowerBound])

        // The fill is a `.background` on the row, and the row's own backdrop is
        // clear — the two halves of the fix; either alone still draws edge to
        // edge. Which modifier OWNS the fill is the whole question, so ask it
        // directly: walk back from the fill to the nearest enclosing modifier.
        guard let fill = body.range(of: "SidebarRowStyle.fill(selected: isSelected") else {
            return XCTFail("the conversation rows stopped reading SidebarRowStyle")
        }
        let before = String(body[body.startIndex..<fill.lowerBound])
        let owner: String? = [".listRowBackground(", ".background("]
            .compactMap { modifier in before.range(of: modifier, options: .backwards).map { ($0.lowerBound, modifier) } }
            .max(by: { $0.0 < $1.0 })?.1
        XCTAssertEqual(owner, ".background(", """
            The conversation highlight must be a `.background` on the row \
            content, not the row's `listRowBackground` — that one spans the \
            full row rect and ignores `listRowInsets`, which is exactly the \
            edge-to-edge selection this test exists to prevent.
            """)
        // …and it fills the row INSIDE the stack's gutter, which is what makes
        // it the same width as a destination's. The gutter itself is pinned by
        // `testEverySidebarRowSharesOneGutterAndOneInset`.
        XCTAssertTrue(body.contains("ChatMetrics.sidebarRowInset"), """
            The conversation rows must take their inner inset from the shared \
            constant — a literal here is how this drifted from the destinations.
            """)
    }

    /// A conversation row is as tall as what is IN it: a chat with no agent
    /// subtitle matches a destination row exactly, and only the ones carrying a
    /// subtitle grow.
    ///
    /// `maxHeight: .infinity` on the row, with a trailing `Spacer` inside the
    /// label, proposed the largest height the list would hand out and let the
    /// spacer soak it up — so every row sat in the two-line block sized for its
    /// tallest neighbour, which reads as a tall highlight wrapped around one
    /// line of text.
    func testAConversationRowIsOnlyAsTallAsItsContent() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        guard let list = chat.range(of: "private var conversationsSidebar"),
              let end = chat.range(of: "private func destinationRow",
                                   range: list.upperBound..<chat.endIndex) else {
            return XCTFail("the sidebar's conversation list moved — update this audit")
        }
        let body = String(chat[list.upperBound..<end.lowerBound])
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces).hasPrefix("//") ? "" : String($0) }
            .joined(separator: "\n")

        XCTAssertFalse(body.contains("maxHeight: .infinity"), """
            A conversation row must not claim the full row height — that is what \
            inflated every row to the tallest one's size.
            """)
        XCTAssertFalse(body.contains("Spacer(minLength: 0)"), """
            A trailing Spacer in the row label expands to whatever height is \
            proposed, which is the other half of the same bug.
            """)
        // The floor is the destination row's own height, so a single-line chat
        // and a destination are the same size — one number, read from one place.
        XCTAssertTrue(body.contains("minHeight: ChatMetrics.sidebarButtonHeight"), """
            A single-line conversation row should match a destination row: use \
            `minHeight: ChatMetrics.sidebarButtonHeight`, never a literal or a \
            fixed height.
            """)
    }

    /// Every row in the sidebar is the same width, inset the same, whichever
    /// half of the panel it belongs to.
    ///
    /// They weren't: the destinations are a `VStack` with `.padding(.horizontal,
    /// 8)`, the conversations are `List` rows with `.listRowInsets(leading: 8)`
    /// — and those measure from the list's OWN content area, which a `.sidebar`
    /// list has already inset. The identical "8" therefore drew two different
    /// widths, the chats pushed right by the list's built-in margin. Two
    /// literals that happen to match are not the same number; one constant is.
    func testEverySidebarRowSharesOneGutterAndOneInset() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
            .split(separator: "\n", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces).hasPrefix("//") ? "" : String($0) }
            .joined(separator: "\n")

        // The conversations are NOT a List. Its `.sidebar` style wraps content
        // in a margin of its own that holds every row ~18pt in from the panel
        // edge, `listRowInsets` does not control it (zeroing them changed
        // nothing on screen), and no API removes it. These rows draw their own
        // background, hover, selection and separators, so the List was
        // contributing only that margin.
        guard let list = chat.range(of: "private var conversationsSidebar"),
              let end = chat.range(of: "private func sectionHeader",
                                   range: list.upperBound..<chat.endIndex) else {
            return XCTFail("the sidebar's conversation list moved — update this audit")
        }
        let sidebar = String(chat[list.upperBound..<end.lowerBound])
        for listism in ["List {", ".listStyle(", ".listRowInsets(", ".listRowBackground(", ".listRowSeparator("] {
            XCTAssertFalse(sidebar.contains(listism), """
                The conversation column is back on a List (`\(listism)`). Its \
                style adds a horizontal margin the destinations above don't \
                have, which is the misalignment this replaced — and nothing in \
                listRowInsets can take it back off.
                """)
        }

        for literal in ["leading: 8", "trailing: 8", "trailing: 6"] {
            XCTAssertFalse(chat.contains(literal), """
                A sidebar row still carries the hand-written inset `\(literal)`. \
                The panel has one gutter (ChatMetrics.sidebarGutter) and one row \
                inset (ChatMetrics.sidebarRowInset); a literal beside them is how \
                the two halves drifted apart.
                """)
        }

        // Both halves are plain stacks taking the SAME gutter — that is what
        // makes them line up, rather than two numbers talked into agreeing.
        XCTAssertEqual(
            chat.components(separatedBy: ".padding(.horizontal, ChatMetrics.sidebarGutter)").count - 1, 2,
            "exactly two columns take the gutter: the destination stack and the conversation stack")
        // …and the row's own inner inset is shared with the destination label
        // and the section headings, so every label starts on one line.
        XCTAssertGreaterThanOrEqual(
            chat.components(separatedBy: "ChatMetrics.sidebarRowInset").count - 1, 3,
            "the row inset should be read by the destination label, the rows and the headers")
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
        for row in ["Agents", "Tasks", "Models", "Settings"] {
            XCTAssertTrue(chat.contains("\"\(row)\""), "the sidebar is missing the \(row) destination")
        }
        // New Chat and the coding CLIs moved into the + beside the Sessions
        // heading (2026-09-02): one menu, New Chat first, the tray's shared
        // launcher list under it. Neither is a destination row any more.
        XCTAssertFalse(chat.contains("destinationRow(\"New Chat\""), "New Chat is the Sessions + now")
        XCTAssertFalse(chat.contains("destinationLabel(\"Code\""), "Code is the Sessions + now")
        guard let menu = SourceScan.declarationBody(from: "private var newSessionMenu", in: chat) else {
            return XCTFail("the Sessions + menu is gone")
        }
        XCTAssertTrue(menu.contains("\"New Chat\""), "the + offers a new chat first")
        XCTAssertTrue(menu.contains("CLILauncherMenuItems("), "the + offers the shared CLI list")
        XCTAssertLessThan(menu.range(of: "\"New Chat\"")!.lowerBound,
                          menu.range(of: "CLILauncherMenuItems(")!.lowerBound)
        XCTAssertTrue(chat.contains("sectionHeader(\"Sessions\") { newSessionMenu }"),
                      "the + sits on the Sessions heading")
        // Two section headings now, and the Agents one renders only when it has
        // rows — a heading with nothing under it promises content that is not
        // there, which is why it can't live in the pinned top inset.
        XCTAssertTrue(chat.contains("sectionHeader(\"Sessions\")"),
                      "the conversation list needs its heading")
        XCTAssertTrue(chat.contains("sectionHeader(\"Agents\")"),
                      "agent threads get their own section above the chats")
        XCTAssertFalse(chat.contains("Text(\"Recent\")"),
                       "\"Recent\" was renamed to \"Sessions\"")
        XCTAssertTrue(chat.contains("if !agentRows.isEmpty"),
                      "the Agents section must be hidden when empty")
        // Pinned above the list, so no destination scrolls away.
        guard let inset = chat.range(of: "safeAreaInset(edge: .top)"),
              let rows = chat.range(of: "destinationRow(", range: inset.upperBound..<chat.endIndex) else {
            return XCTFail("the destinations must ride the sidebar's top inset")
        }
        XCTAssertLessThan(inset.lowerBound, rows.lowerBound)
        // The Create section: rows for the generators, from the SAME catalogue
        // the chips and the Tools menu iterate (`sidebarCreateItems`, pinned in
        // ChatEmptyStateTests) — a hand list here is where the three surfaces
        // would drift apart.
        XCTAssertTrue(chat.contains("ChatEmptyState.sidebarCreateItems"),
                      "the sidebar's Create rows must come from the shared catalogue")
        XCTAssertTrue(chat.contains("sectionHeader(\"Create\")"),
                      "the generator rows get their own heading above Chats")
        // Both directions from the same row.
        XCTAssertTrue(chat.contains("appState.showConversation()"))
        XCTAssertTrue(chat.contains("appState.showModels()"))
        XCTAssertTrue(chat.contains("appState.showTasks()"))
        // The switcher these replaced is gone, not left as a second route.
        XCTAssertFalse(chat.contains("SidebarModeSwitcher"))
    }

    /// A HOSTED pane must not demand a window-sized minimum: the gen views kept
    /// their 820-880pt window-era floors after becoming pages of the chat
    /// window, so at a small window the whole split overflowed the detail
    /// column and clipped BOTH edges (live screenshot 2026-08-09) — the right
    /// answer is the preview side resizing. The controls column keeps its
    /// form-protecting floor; what must never come back is a root frame sized
    /// for a window this view no longer is.
    func testGenPanesDoNotDemandWindowSizedMinimums() throws {
        for pane in ["ImageGenView", "VideoGenView", "AudioGenView",
                     "MusicGenView", "Model3DGenView"] {
            let text = try source("Sources/MLXServe/Views/\(pane).swift")
            for floor in ["minWidth: 7", "minWidth: 8", "minWidth: 9",
                          "minHeight: 6", "minHeight: 7"] {
                XCTAssertFalse(text.contains(floor), """
                    \(pane) demands a window-sized minimum (\(floor)…) — it is \
                    a page of the chat window now; let the preview side shrink.
                    """)
            }
        }
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
