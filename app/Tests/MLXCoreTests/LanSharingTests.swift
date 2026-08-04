import XCTest
import SwiftUI
@testable import MLXCore

/// LAN model sharing, app side: /v1/models entries badged `lan_peer` become
/// pickable network models (tray + every media pane), selections route the
/// remote id into requests, and `LanPick` keeps the "lan:"-prefixed
/// persistence format consistent across panes.
final class LanSharingTests: XCTestCase {

    func testParseModelInfoReadsLanPeerBadge() {
        let entry: [String: Any] = [
            "id": "gemma-4-e4b-it-4bit@Studio",
            "lan_peer": "Studio",
            "capabilities": ["chat", "vision"],
            "meta": ["context_length": 94000],
        ]
        let info = APIClient.parseModelInfo(entry)
        XCTAssertEqual(info.lanPeer, "Studio")
        // `name` keeps the raw routing id — that's what requests must send.
        XCTAssertEqual(info.name, "gemma-4-e4b-it-4bit@Studio")
        XCTAssertEqual(info.lanDisplayName, "gemma-4-e4b-it-4bit · Studio")
        XCTAssertEqual(info.contextLength, 94000)
        XCTAssertTrue(info.supportsVision)

        // Local entries stay unbadged.
        XCTAssertNil(APIClient.parseModelInfo(["id": "local-model"]).lanPeer)
    }

    func testLanPickIdHelpers() {
        XCTAssertEqual(LanPick.lanId("lan:model@peer"), "model@peer")
        XCTAssertNil(LanPick.lanId("flux2-klein-4b-q4"))
        XCTAssertEqual(LanPick.persisted(lanModel: "m@p", presetId: "x"), "lan:m@p")
        XCTAssertEqual(LanPick.persisted(lanModel: nil, presetId: "x"), "x")
        XCTAssertEqual(LanPick.peer(of: "gemma-4-e4b-it-4bit@Studio"), "Studio")
        // `base(of:)` is peer(of:)'s mirror — the model id without the peer.
        XCTAssertEqual(LanPick.base(of: "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit@Studio"),
                       "ddalcu/MiniMax-H3-FL2VA-MLX-Serve-8bit")
        XCTAssertEqual(LanPick.base(of: "no-peer-suffix"), "no-peer-suffix")
    }

    /// A LAN pick whose base id matches a local preset must adopt that preset:
    /// `model` is what the pane gates EVERYTHING on (resolutions, frame
    /// ladder, request capability gating), so leaving it on the previous local
    /// pick sent a remote H3 the local LTX's canvas and 8N+1 frame counts —
    /// below H3's trained floor, at an off-distribution size (bad output that
    /// looks like a model-quality problem). An unknown remote id keeps
    /// today's behavior: local preset untouched, request routed by lanModel.
    func testLanPickAdoptsTheMatchingLocalPreset() {
        var model = VideoModelPreset.ltx23Q4
        var lan: String? = nil
        var persisted = 0
        let selection = LanPick.selection(
            model: Binding(get: { model }, set: { model = $0 }),
            lanModel: Binding(get: { lan }, set: { lan = $0 }),
            resolve: { id in VideoModelPreset.all.first { $0.id == id } },
            persist: { persisted += 1 })

        selection.wrappedValue = "lan:" + VideoModelPreset.minimaxH3.id + "@studio"
        XCTAssertEqual(lan, VideoModelPreset.minimaxH3.id + "@studio")
        XCTAssertEqual(model.id, VideoModelPreset.minimaxH3.id,
                       "the pane must gate on the remote model's own preset")
        XCTAssertGreaterThan(persisted, 0)

        // Unknown remote → the local preset stays where it was.
        selection.wrappedValue = "lan:someone/custom-video@studio"
        XCTAssertEqual(lan, "someone/custom-video@studio")
        XCTAssertEqual(model.id, VideoModelPreset.minimaxH3.id)
    }

    @MainActor
    func testLanModelsFilterByBadgeAndCapability() {
        let mgr = ServerManager()
        defer { mgr.lanChatModelId = nil }
        mgr.allModels = [
            APIClient.parseModelInfo(["id": "big@studio", "lan_peer": "studio", "capabilities": ["chat"]]),
            APIClient.parseModelInfo(["id": "flux@studio", "lan_peer": "studio", "capabilities": ["image"]]),
            APIClient.parseModelInfo(["id": "local-chat", "capabilities": ["chat"]]),
        ]
        XCTAssertEqual(mgr.lanModels(capability: "chat").map(\.name), ["big@studio"])
        XCTAssertEqual(mgr.lanModels(capability: "image").map(\.name), ["flux@studio"])
        XCTAssertTrue(mgr.lanModels(capability: "3d").isEmpty)
    }

    /// Chat requests carry the LAN selection when set; its metadata (context
    /// length budgets, vision) resolves through the discovered entry.
    @MainActor
    func testChatModelIdAndInfoPreferLanSelection() {
        let mgr = ServerManager()
        defer { mgr.lanChatModelId = nil }
        let lan = APIClient.parseModelInfo([
            "id": "big@studio", "lan_peer": "studio",
            "capabilities": ["chat"], "meta": ["context_length": 131072],
        ])
        mgr.allModels = [lan]
        XCTAssertNil(mgr.chatModelId) // nothing selected, no local default

        mgr.lanChatModelId = "big@studio"
        XCTAssertEqual(mgr.chatModelId, "big@studio")
        XCTAssertEqual(mgr.chatModelInfo?.contextLength, 131072)

        mgr.lanChatModelId = nil
        XCTAssertNil(mgr.chatModelId)
    }
}
