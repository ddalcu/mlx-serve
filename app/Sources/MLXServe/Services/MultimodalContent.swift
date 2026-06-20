import Foundation

/// Builds OpenAI-style multimodal `content` blocks (image_url / input_audio /
/// text) for a chat message. Pure and dependency-light so it can be unit-tested:
/// image preprocessing is injected as a closure (the app passes
/// `ImagePreprocessor.preprocess`), and audio is emitted straight from the
/// decoded PCM the server expects.
enum MultimodalContent {
    /// Returns a content-blocks array suitable for a `{"role":"user","content":[...]}`
    /// message. Images become `image_url` blocks (preprocessed pixels when the
    /// preprocessor succeeds, JPEG data-URL otherwise); audio becomes
    /// `input_audio` blocks carrying base64 float32-LE 16 kHz mono PCM; text is
    /// appended last when non-empty.
    static func build(
        text: String,
        images: [ChatImage],
        audio: [ChatAudio] = [],
        // When true (Qwen3-VL and other non-Gemma vision models), skip the
        // Gemma-specific square `x-mlx-pixels` preprocessing and send the raw
        // image so the server runs the model's own preprocessing (smart_resize +
        // merge-order patchify). Gemma keeps Swift-side bicubic preprocessing.
        serverPreprocess: Bool = false,
        preprocessImage: (Data) -> Data? = { ImagePreprocessor.preprocess($0) }
    ) -> [[String: Any]] {
        var blocks: [[String: Any]] = images.map { img in
            if !serverPreprocess, let pixelData = preprocessImage(img.data) {
                return [
                    "type": "image_url",
                    "image_url": [
                        "url": "data:image/x-mlx-pixels;base64,\(pixelData.base64EncodedString())"
                    ] as [String: Any],
                ]
            }
            // Fallback: send JPEG if preprocessing fails (server decodes + resizes).
            return [
                "type": "image_url",
                "image_url": ["url": img.base64URL] as [String: Any],
            ]
        }

        for clip in audio {
            blocks.append([
                "type": "input_audio",
                "input_audio": [
                    "data": clip.pcm.base64EncodedString(),
                    "format": "mlx_pcm_f32",
                ] as [String: Any],
            ])
        }

        if !text.isEmpty {
            blocks.append(["type": "text", "text": text])
        }
        return blocks
    }
}
