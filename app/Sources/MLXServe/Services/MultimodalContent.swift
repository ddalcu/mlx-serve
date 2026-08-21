import Foundation

/// Builds OpenAI-style multimodal `content` blocks (image_url / video_url /
/// input_audio / text) for a chat message. Pure and dependency-light so it can
/// be unit-tested: image preprocessing is injected as a closure (the app
/// passes `ImagePreprocessor.preprocess`), video frames are emitted straight
/// from `VideoPreprocessor.extractFrames`'s JPEG bytes, and audio is emitted
/// straight from the decoded PCM the server expects.
enum MultimodalContent {
    /// Whether the server should do the image preprocessing for this model.
    /// `x-mlx-pixels` is Gemma's square CHW format and ONLY Gemma can read it,
    /// so it is an allowlist of one: every other vision arch (Qwen3-VL,
    /// Muse-Glimmer, whatever lands next) gets the encoded image and runs its
    /// own resize + patchify server-side.
    static func wantsServerPreprocess(architecture: String) -> Bool {
        !architecture.hasPrefix("gemma")
    }

    /// Returns a content-blocks array suitable for a `{"role":"user","content":[...]}`
    /// message. Images become `image_url` blocks (preprocessed pixels when the
    /// preprocessor succeeds, JPEG data-URL otherwise); videos become
    /// `video_url` blocks carrying an ordered array of JPEG-data-URL frames
    /// (Qwen3-VL-family only — no video codec exists server-side, so frame
    /// extraction is done here, client-side); audio becomes `input_audio`
    /// blocks carrying base64 float32-LE 16 kHz mono PCM; text is appended
    /// last when non-empty.
    static func build(
        text: String,
        images: [ChatImage],
        videos: [ChatVideo] = [],
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

        for vid in videos {
            let frameUrls = vid.frames.map { "data:image/jpeg;base64,\($0.base64EncodedString())" }
            blocks.append([
                "type": "video_url",
                "video_url": ["frames": frameUrls] as [String: Any],
            ])
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
