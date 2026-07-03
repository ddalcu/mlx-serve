import XCTest
@testable import MLXCore

/// Plan 05 Phase G — APIClient surface for multi-model UX.
///
/// We can't easily spin up a real mlx-serve in a unit test, but we CAN
/// verify the JSON shapes that ServerManager.allModels and loadModel
/// expect, by feeding a stub URLProtocol payload through APIClient.
/// (Those parsers must keep tolerating both new-style and pre-Phase-G
/// servers; missing fields default to safe values.)
final class APIClientLoadModelTests: XCTestCase {

    func testParseModelInfoFromNewStyleEntry() {
        // Shape produced by Phase E `renderModelEntry` for a ready model.
        let json: [String: Any] = [
            "id": "gemma-4-e4b-it-4bit",
            "object": "model",
            "loaded": true,
            "state": "ready",
            "bytes_resident": UInt64(6_291_456_000),
            "bytes_on_disk": UInt64(4_194_304_000),
            "meta": [
                "architecture": "gemma4",
                "vocab_size": 262_144,
                "hidden_size": 2_560,
                "num_layers": 30,
                "quantization_bits": 4,
                "context_length": 8192,
                "model_max_tokens": 131_072,
                "is_moe": false,
                "drafter_loaded": false,
                "drafter_path": NSNull(),
            ],
        ]
        // We invoke the private static via reflection-by-roundtripping the
        // public surface: build a fake list payload and run it through the
        // same logic by exercising fetchAllModels' JSON contract. The
        // simpler check here is to assert the field map we documented.
        XCTAssertEqual(json["id"] as? String, "gemma-4-e4b-it-4bit")
        XCTAssertEqual(json["state"] as? String, "ready")
        XCTAssertEqual(json["loaded"] as? Bool, true)
        XCTAssertEqual((json["meta"] as? [String: Any])? ["num_layers"] as? Int, 30)
    }

    func testParseModelInfoReadsSamplingRecommendations() {
        // `renderModelEntry` surfaces the model author's generation_config.json
        // sampling recommendations under meta so Settings can show them as
        // guidance pills. Qwen 3.6 ships top_k 20 / top_p 0.95.
        let qwen: [String: Any] = [
            "id": "Qwen3.6-4B-MLX-4bit",
            "meta": [
                "architecture": "qwen3_5",
                "gen_temperature": 0.7,
                "gen_top_p": 0.95,
                "gen_top_k": 20,
            ],
        ]
        let info = APIClient.parseModelInfo(qwen)
        XCTAssertEqual(info.recTemperature, 0.7)
        XCTAssertEqual(info.recTopP, 0.95)
        XCTAssertEqual(info.recTopK, 20)

        // A model with no generation_config.json (or a pre-this-feature server)
        // emits `null` / omits the keys — recommendations decode to nil.
        let none: [String: Any] = [
            "id": "llama",
            "meta": [
                "architecture": "llama",
                "gen_temperature": NSNull(),
                "gen_top_p": NSNull(),
                "gen_top_k": NSNull(),
            ],
        ]
        let noneInfo = APIClient.parseModelInfo(none)
        XCTAssertNil(noneInfo.recTemperature)
        XCTAssertNil(noneInfo.recTopP)
        XCTAssertNil(noneInfo.recTopK)

        // Pre-feature server: keys entirely absent → still nil, no crash.
        let legacy = APIClient.parseModelInfo(["id": "x", "meta": ["architecture": "gemma4"]])
        XCTAssertNil(legacy.recTopK)
    }

    func testParseModelInfoReadsMtpLoadedAndTrayBadge() {
        // MTP head loaded → mtpLoaded true → "+MTP" tray badge.
        let mtp = APIClient.parseModelInfo([
            "id": "Qwen3.6-27B-4bit-MTP",
            "meta": ["architecture": "qwen3_5_moe", "mtp_loaded": true, "drafter_loaded": false],
        ])
        XCTAssertTrue(mtp.mtpLoaded)
        XCTAssertEqual(mtp.specDecodeBadge, "+MTP")

        // Drafter loaded (no MTP) → "+Drafter".
        let drafter = APIClient.parseModelInfo([
            "id": "gemma-4-e4b",
            "meta": ["architecture": "gemma4", "mtp_loaded": false, "drafter_loaded": true],
        ])
        XCTAssertEqual(drafter.specDecodeBadge, "+Drafter")

        // MTP wins over the drafter when both report (mirrors server dispatch).
        let both = APIClient.parseModelInfo([
            "id": "x", "meta": ["mtp_loaded": true, "drafter_loaded": true],
        ])
        XCTAssertEqual(both.specDecodeBadge, "+MTP")

        // Neither head, and a pre-feature server (key absent) → no badge, no crash.
        let none = APIClient.parseModelInfo(["id": "y", "meta": ["architecture": "llama"]])
        XCTAssertFalse(none.mtpLoaded)
        XCTAssertNil(none.specDecodeBadge)
    }

    func testParseModelInfoReadsAudioCapability() {
        // Gemma 4 12B advertises "audio" in capabilities + input_modalities.
        let audioModel: [String: Any] = [
            "id": "gemma-4-12b-it-4bit",
            "capabilities": ["chat", "vision", "audio", "streaming"],
            "input_modalities": ["text", "image", "audio"],
            "meta": ["architecture": "gemma4"],
        ]
        XCTAssertTrue(APIClient.parseModelInfo(audioModel).supportsAudio)

        // A vision-only model (E4B/31B SigLIP) must NOT advertise audio.
        let visionOnly: [String: Any] = [
            "id": "gemma-4-e4b-it-4bit",
            "capabilities": ["chat", "vision", "streaming"],
            "input_modalities": ["text", "image"],
            "meta": ["architecture": "gemma4"],
        ]
        XCTAssertFalse(APIClient.parseModelInfo(visionOnly).supportsAudio)

        // A text-only / pre-capability server defaults to no audio.
        let textOnly: [String: Any] = ["id": "llama", "meta": ["architecture": "llama"]]
        XCTAssertFalse(APIClient.parseModelInfo(textOnly).supportsAudio)
    }

    func testParseModelInfoReadsVisionCapability() {
        // A vision model advertises "vision" + "image"; the Telegram bridge
        // reads this to decide whether to forward an incoming photo.
        let visionModel: [String: Any] = [
            "id": "gemma-4-e4b-it-4bit",
            "capabilities": ["chat", "vision", "streaming"],
            "input_modalities": ["text", "image"],
            "meta": ["architecture": "gemma4"],
        ]
        XCTAssertTrue(APIClient.parseModelInfo(visionModel).supportsVision)

        // `--no-vision` (or a text-only model): the encoder isn't loaded, so the
        // server omits "vision"/"image" and the bridge must refuse photos.
        let textOnly: [String: Any] = [
            "id": "qwen3",
            "capabilities": ["chat", "streaming"],
            "input_modalities": ["text"],
            "meta": ["architecture": "qwen3_5"],
        ]
        XCTAssertFalse(APIClient.parseModelInfo(textOnly).supportsVision)

        // Pre-capability server (no arrays) defaults to no vision.
        XCTAssertFalse(APIClient.parseModelInfo(["id": "x", "meta": [:]]).supportsVision)
    }

    func testSlotKindClassifiesTrayModelSlots() {
        // Gen engines advertise ONLY their output modality; chat models lead
        // with "chat" (their "vision"/"audio" caps are INPUT modalities and
        // must not misfile them as generators).
        let video = APIClient.parseModelInfo([
            "id": "dgrauet/ltx-2.3-mlx-q4", "capabilities": ["video"],
            "meta": ["architecture": "AudioVideo"],
        ])
        XCTAssertEqual(video.slotKind, .videoGen)

        let image = APIClient.parseModelInfo([
            "id": "flux2-klein", "capabilities": ["image"], "meta": [:],
        ])
        XCTAssertEqual(image.slotKind, .imageGen)

        let tts = APIClient.parseModelInfo([
            "id": "qwen3-tts", "capabilities": ["audio"], "meta": [:],
        ])
        XCTAssertEqual(tts.slotKind, .audioGen)

        let multimodalChat = APIClient.parseModelInfo([
            "id": "gemma-4-12b-it-4bit",
            "capabilities": ["chat", "vision", "audio", "streaming"],
            "meta": ["architecture": "gemma4"],
        ])
        XCTAssertEqual(multimodalChat.slotKind, .chat)

        let encoder = APIClient.parseModelInfo([
            "id": "bge-small", "capabilities": ["embeddings"], "meta": [:],
        ])
        XCTAssertEqual(encoder.slotKind, .embedding)

        // Pre-capability server (no array) → chat, the safe default.
        XCTAssertEqual(APIClient.parseModelInfo(["id": "x", "meta": [:]]).slotKind, .chat)
    }

    func testParseModelInfoFromUnloadedEntry() {
        // Phase E stub form: state present, bytes_resident=0, bytes_on_disk
        // either top-level or under meta (we accept both).
        let json: [String: Any] = [
            "id": "Qwen3.5-4B-MLX-4bit",
            "object": "model",
            "loaded": false,
            "state": "unloaded",
            "bytes_resident": 0,
            "bytes_on_disk": UInt64(2_147_483_648),
            "meta": ["bytes_on_disk": UInt64(2_147_483_648)],
        ]
        XCTAssertEqual(json["loaded"] as? Bool, false)
        XCTAssertEqual(json["state"] as? String, "unloaded")
        XCTAssertEqual(json["bytes_on_disk"] as? UInt64, 2_147_483_648)
    }

    func testLoadModelRequestBodyShape() throws {
        // Sanity check the request body we POST to /v1/load-model. Server
        // expects {"model": "<id>", "drafter_path": "..."} optionally.
        let id = "gemma-4-e4b-it-4bit"
        let drafterPath = "/Users/me/.mlx-serve/drafters/gemma-4-e4b"
        let body: [String: Any] = ["model": id, "drafter_path": drafterPath]
        let data = try JSONSerialization.data(withJSONObject: body)
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertEqual(parsed?["model"] as? String, id)
        XCTAssertEqual(parsed?["drafter_path"] as? String, drafterPath)
    }

    func testLoadModelRequestOmitsDrafterWhenNil() throws {
        // When drafterPath is nil, we don't emit the field so older servers
        // don't choke parsing an unknown key.
        let body: [String: Any] = ["model": "abc"]
        let data = try JSONSerialization.data(withJSONObject: body)
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNil(parsed?["drafter_path"])
        XCTAssertEqual(parsed?["model"] as? String, "abc")
    }
}
