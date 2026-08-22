import XCTest
@testable import MLXCore

/// ⌘L — the model switcher. The pill's menu is the mouse route to the same
/// rows; this is the typing one, so what it OFFERS and what a pick MEANS are
/// the two things that must not drift from it.
final class ModelPaletteTests: XCTestCase {

    private func local(_ name: String,
                       path: String? = nil,
                       type: String = "gemma4",
                       source: LocalModelSource = .mlxServe,
                       kind: ModelKind = .base,
                       size: String = "5 GB") -> LocalModel {
        LocalModel(id: "\(source):\(name)", name: name, path: path ?? "/models/\(name)",
                   sizeFormatted: size, modelType: type, source: source, kind: kind)
    }

    private func lan(_ id: String, peer: String) -> ModelInfo {
        var info = ModelInfo(name: id, quantBits: 4, layers: 1, hiddenSize: 1,
                             vocabSize: 1, contextLength: 4096, modelMaxTokens: 4096)
        info.lanPeer = peer
        info.capabilities = ["chat"]
        return info
    }

    // MARK: - What it offers

    /// Same filter as the composer's pill (`isChatPickable`): a drafter, a
    /// media checkpoint or an embeddings encoder in this list is a row that
    /// 400s the next message.
    func testOnlyChatPickableModelsGetARow() {
        let rows = ModelPalette.rows(local: [
            local("mlx-community/gemma-4-e4b-it-4bit"),
            local("mlx-community/gemma-4-e4b-it-assistant-bf16", type: "gemma4", kind: .drafter),
            local("ddalcu/LTX-2.5-MLX-Serve-8bit", type: "AudioVideo"),
            local("mlx-community/bge-small-en-v1.5", type: "bert"),
        ], lan: [])
        XCTAssertEqual(rows.map(\.tag), ["/models/mlx-community/gemma-4-e4b-it-4bit"])
    }

    /// Local groups in `LocalModelSource.allCases` order, network last — the
    /// pill's own order, because a second ordering is a second answer to
    /// "which model is at the top of the list?".
    func testGroupsFollowThePillsOrderWithTheNetworkLast() {
        let rows = ModelPalette.rows(local: [
            local("other/llama-3-8b", type: "llama", source: .lmStudio),
            local("mlx-community/gemma-4-e4b-it-4bit"),
        ], lan: [lan("qwen3-8b@studio", peer: "studio")])
        XCTAssertEqual(rows.map(\.section),
                       [LocalModelSource.mlxServe.sectionTitle,
                        LocalModelSource.lmStudio.sectionTitle,
                        ModelPalette.networkSection])
    }

    /// A peer entry that does not advertise chat is somebody else's image
    /// model — the same `lanModels(capability:)` question the pill asks.
    func testANetworkRowMustAdvertiseChat() {
        var media = lan("ltx@studio", peer: "studio")
        media.capabilities = ["video"]
        let rows = ModelPalette.rows(local: [], lan: [lan("qwen3-8b@studio", peer: "studio"), media])
        XCTAssertEqual(rows.map(\.tag), ["lan:qwen3-8b@studio"])
    }

    // MARK: - What a pick means

    /// The palette decides nothing about loading: it hands back a
    /// `ChatModelSelection` tag, so the local/LAN split stays in the one place
    /// the tray and the pill already read.
    func testRowsCarryChatModelSelectionTags() {
        let rows = ModelPalette.rows(local: [local("mlx-community/gemma-4-e4b-it-4bit")],
                                     lan: [lan("qwen3-8b@studio", peer: "studio")])
        XCTAssertEqual(ChatModelSelection.action(for: rows[0].tag),
                       .selectLocal("/models/mlx-community/gemma-4-e4b-it-4bit"))
        XCTAssertEqual(ChatModelSelection.action(for: rows[1].tag), .selectLan("qwen3-8b@studio"))
    }

    // MARK: - Filtering

    func testFilteringIsCaseInsensitiveAndMatchesTheRepoId() {
        let rows = ModelPalette.rows(local: [
            local("mlx-community/gemma-4-e4b-it-4bit"),
            local("mlx-community/Qwen3.6-27B-8bit", type: "qwen3_5"),
        ], lan: [])
        XCTAssertEqual(ModelPalette.filtered(rows, query: "GEMMA").count, 1)
        XCTAssertEqual(ModelPalette.filtered(rows, query: "e4b").first?.tag,
                       "/models/mlx-community/gemma-4-e4b-it-4bit")
        XCTAssertEqual(ModelPalette.filtered(rows, query: "").count, 2, "an empty query hides nothing")
        XCTAssertEqual(ModelPalette.filtered(rows, query: "   ").count, 2, "…nor does whitespace")
        XCTAssertTrue(ModelPalette.filtered(rows, query: "mistral").isEmpty)
    }

    /// Every word must land, so a second word NARROWS instead of starting a
    /// new search — "qwen 8bit" is how you tell two quants of one repo apart.
    func testEveryWordOfTheQueryMustMatch() {
        let rows = ModelPalette.rows(local: [
            local("mlx-community/Qwen3.6-27B-8bit", type: "qwen3_5"),
            local("mlx-community/Qwen3.6-27B-4bit", type: "qwen3_5"),
        ], lan: [])
        XCTAssertEqual(ModelPalette.filtered(rows, query: "qwen 8bit").map(\.tag),
                       ["/models/mlx-community/Qwen3.6-27B-8bit"])
        XCTAssertTrue(ModelPalette.filtered(rows, query: "qwen mistral").isEmpty)
    }

    /// The on-disk PATH is not searchable text. Every model under
    /// `~/.mlx-serve/models` shares most of one, so matching it makes "models",
    /// "users" and the account name match everything — a filter that answers
    /// "all of them" is a filter that has stopped working.
    func testTheFilesystemPathIsNotMatched() {
        let rows = ModelPalette.rows(
            local: [local("mlx-community/gemma-4-e4b-it-4bit",
                          path: "/Users/pat/.mlx-serve/models/mlx-community/gemma-4-e4b-it-4bit")],
            lan: [])
        XCTAssertTrue(ModelPalette.filtered(rows, query: "Users").isEmpty)
        XCTAssertTrue(ModelPalette.filtered(rows, query: "mlx-serve").isEmpty)
        XCTAssertEqual(ModelPalette.filtered(rows, query: "gemma").count, 1)
    }

    /// A peer's name is worth typing: "studio" is how you ask for the models
    /// on the other Mac.
    func testANetworkRowIsFoundByItsPeerName() {
        let rows = ModelPalette.rows(local: [], lan: [lan("qwen3-8b@studio", peer: "studio")])
        XCTAssertEqual(ModelPalette.filtered(rows, query: "studio").count, 1)
    }

    // MARK: - Selection

    /// ⌘L then Return must be a no-op, not a switch to whatever sorts first:
    /// opening the picker is not asking to change model.
    func testSelectionOpensOnTheCurrentModel() {
        let rows = ModelPalette.rows(local: [
            local("mlx-community/gemma-4-e4b-it-4bit"),
            local("mlx-community/Qwen3.6-27B-8bit", type: "qwen3_5"),
        ], lan: [])
        XCTAssertEqual(ModelPalette.selection(in: rows, current: rows[1].tag), 1)
    }

    /// Chatting over the network, the LAN row is the current one — the same
    /// precedence `ChatModelSelection.tag` gives it.
    func testTheNetworkRowCanBeTheCurrentOne() {
        let rows = ModelPalette.rows(local: [local("mlx-community/gemma-4-e4b-it-4bit")],
                                     lan: [lan("qwen3-8b@studio", peer: "studio")])
        let current = ChatModelSelection.tag(localPath: "/models/mlx-community/gemma-4-e4b-it-4bit",
                                             lanChatModelId: "qwen3-8b@studio")
        XCTAssertEqual(ModelPalette.selection(in: rows, current: current), 1)
    }

    /// A current model the list does not hold (filtered out, or a peer that
    /// just went away) falls to the top rather than nowhere — and an empty
    /// list selects nothing instead of row zero, which does not exist.
    func testAnAbsentCurrentModelSelectsTheTopRow() {
        let rows = ModelPalette.rows(local: [local("mlx-community/gemma-4-e4b-it-4bit")], lan: [])
        XCTAssertEqual(ModelPalette.selection(in: rows, current: "lan:gone@studio"), 0)
        XCTAssertNil(ModelPalette.selection(in: [], current: "anything"))
    }

    /// Arrows CLAMP. Wrapping means holding ↓ past the last model silently
    /// puts you back on the first one, and Return then loads a model nobody
    /// was looking at.
    func testArrowsClampAtBothEnds() {
        XCTAssertEqual(ModelPalette.move(0, by: -1, count: 3), 0)
        XCTAssertEqual(ModelPalette.move(2, by: 1, count: 3), 2)
        XCTAssertEqual(ModelPalette.move(0, by: 1, count: 3), 1)
        XCTAssertNil(ModelPalette.move(0, by: 1, count: 0), "nothing to select in an empty list")
        XCTAssertEqual(ModelPalette.move(nil, by: 1, count: 3), 0, "the first arrow key starts at the top")
    }

    /// Typing shrinks the list under the selection, so the index is clamped
    /// where it is READ — an index into rows that no longer exist is the
    /// out-of-bounds trap, and the row it names has changed anyway.
    func testAShrinkingListPullsTheSelectionBackInRange() {
        let rows = ModelPalette.rows(local: [
            local("mlx-community/gemma-4-e4b-it-4bit"),
            local("mlx-community/Qwen3.6-27B-8bit", type: "qwen3_5"),
        ], lan: [])
        let narrowed = ModelPalette.filtered(rows, query: "gemma")
        XCTAssertEqual(ModelPalette.tag(at: 1, in: narrowed), nil)
        XCTAssertEqual(ModelPalette.clamped(1, count: narrowed.count), 0)
        XCTAssertNil(ModelPalette.clamped(0, count: 0))
    }

    // MARK: - Rows read as models

    /// Two rows that read identically are a coin flip. `duplicateNames` is the
    /// pill's own answer; here the engine rides the DETAIL line, since the list
    /// has no checkmark keyed by title to protect.
    func testDuplicateNamesAreToldApartByEngine() {
        let mlx = local("org/model-x")
        var gguf = local("org/model-x", path: "/models/org/model-x.gguf", type: "llama")
        gguf.quantFile = "model-x-Q4_K_M.gguf"
        let rows = ModelPalette.rows(local: [mlx, gguf], lan: [])
        XCTAssertEqual(rows.count, 2)
        XCTAssertNotEqual(rows[0].detail, rows[1].detail)
    }

    /// The title is the readable name (`ModelDisplayName`), the same rename the
    /// pill shows — the repo id stays the identity underneath and stays
    /// searchable.
    func testTitleIsTheReadableName() {
        let rows = ModelPalette.rows(local: [local("mlx-community/gemma-4-e4b-it-4bit")], lan: [])
        XCTAssertEqual(rows[0].title, ModelDisplayName.pretty("mlx-community/gemma-4-e4b-it-4bit"))
        XCTAssertEqual(ModelPalette.filtered(rows, query: "mlx-community").count, 1,
                       "the repo id is still what you can type")
    }

    // MARK: - Wiring

    private func source(_ relativePath: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXCoreTests
            .deletingLastPathComponent()   // Tests
            .deletingLastPathComponent()   // app
        return try String(contentsOf: root.appendingPathComponent(relativePath), encoding: .utf8)
    }

    /// Two sheets on one window is one sheet plus a thing nobody can see, and
    /// the gate is the blocking one — so it stands down while the palette is
    /// up, exactly as it already stands down over the models pane. Deferred,
    /// never dismissed: it presents again the moment the palette closes.
    func testTheGateStandsDownWhileThePaletteIsUp() {
        XCTAssertFalse(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
                                            workspace: .conversation, welcomePresented: false,
                                            palettePresented: true))
        XCTAssertTrue(
            ChatWorkspace.gateShouldPresent(gateIsBlocking: true, cancelled: false,
                                            workspace: .conversation, welcomePresented: false,
                                            palettePresented: false))
    }

    /// ⌘L is a menu key equivalent, so it works from anywhere in the app — and
    /// it goes through the ONE door on `AppState`, which both sets the flag and
    /// brings the chat window forward. A command that flipped the flag itself
    /// would open a picker on a window nobody is looking at.
    func testCommandLIsTheShortcutAndItRoutesThroughTheOneDoor() throws {
        let app = try source("Sources/MLXServe/MLXServeApp.swift")
        XCTAssertTrue(app.contains(#".keyboardShortcut("l", modifiers: [.command])"#),
                      "⌘L must be bound in the menus")
        XCTAssertTrue(app.contains("appState.showModelPalette()"),
                      "the command opens the palette through AppState's door")
    }

    /// ONE presentation site, on the whole window rather than one split: the
    /// picker has to open in Tasks and Agents too, which are a different
    /// `NavigationSplitView`.
    func testThePaletteIsPresentedExactlyOnce() throws {
        let chat = try source("Sources/MLXServe/Views/ChatView.swift")
        XCTAssertEqual(chat.components(separatedBy: "ModelPaletteSheet(").count - 1, 1)
    }

    /// Applying a pick is ONE method. The tray and the pill each had their own
    /// copy of "clear the LAN id, then set the path"; a third would be the
    /// per-surface-copy class the tag semantics themselves were centralised to
    /// avoid.
    func testEverySurfaceAppliesAPickThroughTheOneMethod() throws {
        for file in ["Sources/MLXServe/Views/ChatModelPill.swift",
                     "Sources/MLXServe/Views/StatusMenuView.swift",
                     "Sources/MLXServe/Views/ModelPaletteSheet.swift"] {
            let text = try source(file)
            XCTAssertTrue(text.contains("applyChatModelPick("),
                          "\(file) must apply a pick through AppState.applyChatModelPick")
            XCTAssertFalse(text.contains("appState.selectedModelPath ="),
                           "\(file) must not re-implement what picking a model means")
        }
    }
}
