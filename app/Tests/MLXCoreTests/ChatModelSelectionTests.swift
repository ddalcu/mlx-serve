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
