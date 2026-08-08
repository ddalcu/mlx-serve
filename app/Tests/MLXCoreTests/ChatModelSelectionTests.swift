import XCTest
@testable import MLXCore

/// The chat model a picker shows and what picking a row means.
///
/// Extracted because there are now TWO pickers (the menu-bar tray and the chat
/// window's toolbar) and they must agree. A per-surface copy of this logic is
/// exactly how one picker ends up ignoring a LAN selection — the same class as
/// the rule that a chat surface routes through `server.chatModelId` rather than
/// reading `modelInfo?.name` for itself.
final class ChatModelSelectionTests: XCTestCase {

    func testLanSelectionWinsOverTheLocalPath() {
        // A LAN chat is served by another Mac; the local `selectedModelPath` is
        // still set underneath and must not be what the picker ticks.
        XCTAssertEqual(
            ChatModelSelection.tag(localPath: "/models/local", lanChatModelId: "qwen@studio"),
            "lan:qwen@studio")
    }

    func testLocalPathIsTheTagWhenNoLanModelIsSelected() {
        XCTAssertEqual(ChatModelSelection.tag(localPath: "/models/local", lanChatModelId: nil),
                       "/models/local")
    }

    func testPickingALanRowSelectsTheLanModel() {
        XCTAssertEqual(ChatModelSelection.action(for: "lan:qwen@studio"), .selectLan("qwen@studio"))
    }

    func testPickingALocalRowClearsTheLanSelection() {
        // Without the clear, a local pick would leave the LAN id set and every
        // turn would keep going out to the network.
        XCTAssertEqual(ChatModelSelection.action(for: "/models/local"), .selectLocal("/models/local"))
    }

    func testTagsRoundTrip() {
        // Class guard: whatever the picker shows must decode back to the same
        // choice, or the checkmark lands on a row that isn't what loads.
        for (path, lan) in [("/a", nil), ("/b", "m@peer"), ("", "x@y")] as [(String, String?)] {
            let tag = ChatModelSelection.tag(localPath: path, lanChatModelId: lan)
            switch ChatModelSelection.action(for: tag) {
            case .selectLan(let id): XCTAssertEqual(id, lan)
            case .selectLocal(let p): XCTAssertEqual(p, path)
            }
        }
    }

    func testAPathContainingTheLanWordIsNotTreatedAsALanId() {
        // Only the "lan:" PREFIX marks a network row — a local folder called
        // "lan" or a path with "lan:" inside it must still load locally.
        XCTAssertEqual(ChatModelSelection.action(for: "/Users/me/lan/models"),
                       .selectLocal("/Users/me/lan/models"))
    }

    // MARK: - Header name
    //
    // The toolbar pill drops the org, which is the half of a Hugging Face id
    // that is identical across most of your models and was eating the width
    // budget mid-truncation ("mlx-commun…B-it-qat-4bit" told you nothing). The
    // MENU keeps full ids — that is where you're choosing between them, and two
    // orgs can ship the same model name.

    func testTheHeaderDropsTheOrg() {
        XCTAssertEqual(ChatModelPill.headerName("mlx-community/gemma-3-12b-it-qat-4bit"),
                       "gemma-3-12b-it-qat-4bit")
    }

    func testANameWithNoOrgIsUnchanged() {
        XCTAssertEqual(ChatModelPill.headerName("Select a model"), "Select a model")
        XCTAssertEqual(ChatModelPill.headerName("gemma-3-12b"), "gemma-3-12b")
    }

    /// A LAN id is `org/model@peer` — dropping the org must keep the peer, or
    /// the pill stops saying the answer is coming from another Mac.
    func testALanIdKeepsItsPeer() {
        XCTAssertEqual(ChatModelPill.headerName("mlx-community/qwen3-4b@studio"),
                       "qwen3-4b@studio")
    }

    /// Nested ids and a stray trailing slash must not produce an empty pill.
    func testDegenerateFormsNeverGoEmpty() {
        XCTAssertEqual(ChatModelPill.headerName("a/b/c"), "c")
        XCTAssertEqual(ChatModelPill.headerName("org/"), "org/")
        XCTAssertEqual(ChatModelPill.headerName(""), "")
    }
}

/// The red "Start" button beside the chat model picker: when it shows, and what
/// it says. The chat window is where you find out the server is down — the pill
/// just goes grey — so the fix has to be reachable from there rather than only
/// from the tray.
final class ChatServerStartControlTests: XCTestCase {

    /// A running server has nothing to start.
    func testHiddenWhileRunning() {
        XCTAssertEqual(ChatServerStartControl.resolve(status: .running, hasStartableModel: true), .hidden)
    }

    /// Stopped, with something to load: the button, in red.
    func testStartOfferedWhenStopped() {
        XCTAssertEqual(ChatServerStartControl.resolve(status: .stopped, hasStartableModel: true), .start)
        XCTAssertEqual(ChatServerStartControl.start.title, "Start")
        XCTAssertTrue(ChatServerStartControl.start.isRed)
    }

    /// A crashed server is offered the same way — "Error" in the tray is not an
    /// instruction, and the recovery is identical.
    func testStartOfferedAfterAnError() {
        XCTAssertEqual(ChatServerStartControl.resolve(status: .error("boom"), hasStartableModel: true), .start)
    }

    /// The same control keeps reporting while the model loads (which takes tens
    /// of seconds) — vanishing on click would read as the click not landing.
    func testStartingKeepsTheControlWithProgress() {
        let c = ChatServerStartControl.resolve(status: .starting, hasStartableModel: true)
        XCTAssertEqual(c, .starting)
        XCTAssertEqual(c.title, "Starting…")
        XCTAssertFalse(c.isEnabled)
        XCTAssertFalse(c.isRed, "a control you cannot press must not shout")
    }

    /// Nothing to start ⇒ no button. A disabled red control that never explains
    /// itself is the dead-control class; the pill already says "Select a model".
    func testHiddenWithNothingToStart() {
        XCTAssertEqual(ChatServerStartControl.resolve(status: .stopped, hasStartableModel: false), .hidden)
        XCTAssertEqual(ChatServerStartControl.resolve(status: .error("x"), hasStartableModel: false), .hidden)
    }

    /// `.starting` shows even with no local model selected: that state is only
    /// reachable because something already started it (a LAN pick boots the
    /// server headless), and hiding it mid-load would blink the toolbar.
    func testStartingShowsEvenWithNoLocalSelection() {
        XCTAssertEqual(ChatServerStartControl.resolve(status: .starting, hasStartableModel: false), .starting)
    }

    /// Only `.start` is pressable — the guard against wiring the action to a
    /// state that is already doing the thing.
    func testOnlyStartIsPressable() {
        XCTAssertTrue(ChatServerStartControl.start.isEnabled)
        XCTAssertFalse(ChatServerStartControl.hidden.isEnabled)
        XCTAssertFalse(ChatServerStartControl.starting.isEnabled)
    }
}

/// A media generator lives in the SAME registry as chat models — the image /
/// video / audio panes load one through `prepareGenModel`, and the server sorts
/// its default first, so `modelInfo` (`allModels.first`) can be a model that
/// cannot answer a single chat request.
///
/// Live 2026-08-05: with a media model loaded the chat window's pill named it,
/// dotted GREEN ("loaded and ready to answer") and every chat surface reading
/// `chatModelId` would have sent the turn to it. The picker's LIST was already
/// chat-only — the resolution underneath it was not.
final class ChatModelResolutionTests: XCTestCase {

    private func info(_ id: String, _ caps: [String], loaded: Bool = true) -> ModelInfo {
        APIClient.parseModelInfo(["id": id, "capabilities": caps, "loaded": loaded])
    }

    @MainActor
    func testALoadedMediaModelIsNeverTheChatModel() {
        let mgr = ServerManager()
        defer { mgr.lanChatModelId = nil }
        let video = info("ddalcu/MiniMax-H3-FL2VA-MLX-Serve-4bit", ["video"])
        let chat = info("mlx-community/LFM2.5-2.6B-8bit", ["chat"])

        mgr.allModels = [video, chat]
        mgr.modelInfo = video   // the gen flow loaded it; the server sorts it first

        XCTAssertEqual(mgr.chatModelInfo?.name, chat.name)
        XCTAssertEqual(mgr.chatModelId, chat.name, "a chat turn must not be addressed to a video model")
    }

    /// Nothing that can chat ⇒ NOTHING. Naming the generator would earn a
    /// "does not support this media modality" 400, and the pill's dot keys on
    /// nil to stay amber instead of claiming ready.
    @MainActor
    func testAMediaOnlyServerHasNoChatModel() {
        let mgr = ServerManager()
        defer { mgr.lanChatModelId = nil }
        let image = info("ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8", ["image"])
        mgr.allModels = [image]
        mgr.modelInfo = image

        XCTAssertNil(mgr.chatModelInfo)
        XCTAssertNil(mgr.chatModelId)
    }

    /// An UNLOADED chat stub is not "the model answering" — it is a model you
    /// could load. Reporting it would turn the pill's dot green on a server
    /// holding nothing but a generator.
    @MainActor
    func testAnUnloadedChatStubDoesNotStandInForAResidentOne() {
        let mgr = ServerManager()
        defer { mgr.lanChatModelId = nil }
        mgr.allModels = [info("ddalcu/Krea-2-Turbo-MLX-Serve-mixed-4-8", ["image"]),
                         info("mlx-community/LFM2.5-2.6B-8bit", ["chat"], loaded: false)]
        mgr.modelInfo = mgr.allModels[0]

        XCTAssertNil(mgr.chatModelInfo)
    }

    /// An embedding model is in the registry too (folder indexing loads one)
    /// and is just as unable to hold a conversation.
    @MainActor
    func testAnEmbeddingModelIsNotAChatModelEither() {
        let mgr = ServerManager()
        defer { mgr.lanChatModelId = nil }
        let embed = info("mlx-community/bge-small-en-v1.5-8bit", ["embeddings"])
        mgr.allModels = [embed]
        mgr.modelInfo = embed

        XCTAssertNil(mgr.chatModelId)
    }

    /// Unchanged where it already worked: a chat model that also takes images
    /// and audio advertises those as INPUTS and is still the chat model, and a
    /// LAN pick still wins over everything local.
    @MainActor
    func testAMultimodalChatModelAndALanPickAreUnaffected() {
        let mgr = ServerManager()
        defer { mgr.lanChatModelId = nil }
        let gemma = info("mlx-community/gemma-4-e4b-it-4bit", ["chat", "vision", "audio"])
        mgr.allModels = [gemma]
        mgr.modelInfo = gemma
        XCTAssertEqual(mgr.chatModelId, gemma.name)

        // A LAN selection wins even before discovery lands it in `allModels`.
        mgr.lanChatModelId = "big@studio"
        XCTAssertEqual(mgr.chatModelId, "big@studio")
    }

    /// Pre-Phase-G / GGUF entries report no capabilities at all and still chat
    /// (the same tolerance `slotKind` and `lanAdvertises` already carry).
    @MainActor
    func testAnEntryWithNoCapabilitiesStillCounts() {
        let mgr = ServerManager()
        defer { mgr.lanChatModelId = nil }
        let old = info("some/gguf-model", [])
        mgr.allModels = [old]
        mgr.modelInfo = old
        XCTAssertEqual(mgr.chatModelId, old.name)
    }
}

/// Class guard for the same bug one hop out: every surface that hands a MODEL
/// to something which will chat — the CLI launcher's generated agent configs,
/// the setup instructions, the headless test harness — must resolve it through
/// `chatModelId`/`chatModelInfo`, never `modelInfo` (which is whatever the
/// server loaded last, media generators included). A config baked with a video
/// model's id sends the CLI's first turn to a model that 400s.
final class ChatSurfaceModelSourceTests: XCTestCase {

    private static var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()    // MLXCoreTests
            .deletingLastPathComponent()    // Tests
            .deletingLastPathComponent()    // app
            .appendingPathComponent("Sources/MLXServe")
    }

    func testNoChatSurfaceBakesTheRawLoadedModelId() throws {
        let fm = FileManager.default
        let walker = try XCTUnwrap(fm.enumerator(at: Self.sourcesRoot, includingPropertiesForKeys: nil))
        var offenders: [String] = []
        for case let url as URL in walker where url.pathExtension == "swift" {
            let text = try String(contentsOf: url, encoding: .utf8)
            for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
                // The launcher's own parameter list is where the id is CONSUMED;
                // only the call sites that supply one are being audited.
                guard line.contains("servedModelId:"), line.contains("modelInfo?.name") else { continue }
                offenders.append("\(url.lastPathComponent): \(line.trimmingCharacters(in: .whitespaces))")
            }
        }
        XCTAssertTrue(offenders.isEmpty,
                      "route through server.chatModelId — a media model would be handed to a chat CLI: \(offenders)")
    }
}
