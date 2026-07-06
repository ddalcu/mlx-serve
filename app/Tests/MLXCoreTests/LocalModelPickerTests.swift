import XCTest
@testable import MLXCore

/// Chat-model pickers (tray menu, task sheet, auto-select) must only offer
/// models loadable as the server's primary CHAT model. Media checkpoints
/// (LTX "AudioVideo", FLUX/Krea/TTS) and non-chat models (Falconsai NSFW
/// classifier "vit", "bert" embeddings encoders) live under
/// `~/.mlx-serve/models` as gen-pane / RAG dependencies — they carry a
/// config.json + safetensors so discovery classifies them `.base`, but
/// selecting one in the tray and pressing Start can only fail (or serve
/// no chat endpoint at all).
final class LocalModelPickerTests: XCTestCase {
    private func model(_ modelType: String, kind: ModelKind = .base) -> LocalModel {
        LocalModel(
            id: "test:\(modelType)", name: modelType, path: "/tmp/\(modelType)",
            sizeFormatted: "1 GB", modelType: modelType, source: .mlxServe, kind: kind
        )
    }

    func testMediaAndNonChatModelsAreNotChatPickable() {
        // LTX-Video ("AudioVideo"), the Falconsai NSFW classifier ("vit"),
        // the other media archs, and embeddings-only encoders must never be
        // offered by a chat-model picker. Strings are the exact model_type
        // values from real ~/.mlx-serve/models checkpoints.
        for mt in ["AudioVideo", "vit", "flux2-klein-4b", "krea2_turbo", "qwen3_tts", "moss_tts_nano", "bert", "hunyuan3d_2_1", "hunyuan3d_2_1_paint", "acestep"] {
            XCTAssertFalse(model(mt).isChatPickable, "\"\(mt)\" must not be chat-pickable")
        }
    }

    func testChatModelsRemainPickable() {
        for mt in ["gemma4", "gemma3_text", "qwen3_5_moe", "diffusion_gemma", "gguf", "deepseek_v4"] {
            XCTAssertTrue(model(mt).isChatPickable, "\"\(mt)\" must stay chat-pickable")
        }
    }

    func testDrafterIsNeverChatPickable() {
        // Drafters pair with a base model via --drafter; not loadable alone.
        XCTAssertFalse(model("gemma4_assistant", kind: .drafter).isChatPickable)
    }

    func testUnparseableConfigIsNotChatPickable() {
        // parseConfigMetadata defaults modelType to "unknown" when config.json
        // is missing/garbled — such a dir can't load, so don't offer it.
        XCTAssertFalse(model("unknown").isChatPickable)
    }
}
