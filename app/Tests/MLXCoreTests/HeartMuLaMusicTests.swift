import XCTest
@testable import MLXCore

/// HeartMuLa in the Music pane: lyric-conditioned from a tag line, no wordless mode, no tempo/key/language
/// fields and no reference audio. The family gates the FIELDS (values linger across a model switch and the
/// server names each a 400).
final class HeartMuLaMusicTests: XCTestCase {

    private let hm = MusicModelPreset.heartMuLa3B_8bit

    func testThePackDeclaresItsKnobSet() {
        for p in [MusicModelPreset.heartMuLa3B_8bit] {
            XCTAssertEqual(p.family, .heartmula, p.name)
            XCTAssertTrue(p.requiresLyrics, p.name)
            XCTAssertFalse(p.supportsInstrumental, p.name)
            XCTAssertFalse(p.supportsTempoAndKey, p.name)
            XCTAssertFalse(p.supportsMusicalMeta, p.name)
            XCTAssertFalse(p.supportsReferenceAudio, p.name)
            XCTAssertFalse(p.supportsSourceAudio, p.name)
            XCTAssertFalse(p.supportsScore, p.name)
            XCTAssertTrue(p.supportsAutoLength, "the model ends its own song: \(p.name)")
            XCTAssertTrue(p.supportsSampling, p.name)
            XCTAssertEqual(p.stepsRange, 4...50, p.name)
            XCTAssertEqual(p.durationRange, 5...360, p.name)
            XCTAssertTrue(MusicModelPreset.all.contains(p), p.name)
        }
        XCTAssertEqual(hm.repo, "ddalcu/HeartMuLa-3B-MLX-Serve-8bit")
    }

    func testRequestBodyCarriesNoFieldAnotherBackendOwns() {
        let req = MusicGenRequest(
            model: hm, prompt: "pop,piano", lyrics: "[Verse]\nla",
            instrumental: true, vocalLanguage: "ja", bpm: 96, keyscale: "C major",
            timesignature: "3", durationSeconds: 90, seed: 7, steps: 80)
        let body = MusicGenService.requestBody(req, modelName: "heartmula", refAudioB64: "AAAA", srcAudioB64: "AAAA")
        for key in ["instrumental", "bpm", "keyscale", "vocal_language", "timesignature", "ref_audio", "src_audio", "task", "cot"] {
            XCTAssertNil(body[key], "\(key) is another backend's field; the server names it a 400")
        }
        XCTAssertEqual(body["lyrics"] as? String, "[Verse]\nla")
        XCTAssertEqual(body["steps"] as? Int, 50, "clamped into the server's [4,50]")
        XCTAssertEqual(body["duration_seconds"] as? Int, 90 + MusicGenService.lengthGrace(90), "a set length is a soft cap")
        var auto = req
        auto.autoLength = true
        XCTAssertNil(MusicGenService.requestBody(auto, modelName: "heartmula")["duration_seconds"], "Auto lets the model end the song")
        for key in ["cfg_scale", "temperature", "top_k"] {
            XCTAssertNil(body[key], "\(key) left at the server default is omitted")
        }
    }

    func testSamplingKnobsTravelClampedOnlyToHeartMuLa() {
        var req = MusicGenRequest(model: hm, prompt: "pop", lyrics: "[Verse]\nla", durationSeconds: 90, autoLength: false)
        req.cfgScale = 99; req.temperature = 0; req.topK = 5000
        let body = MusicGenService.requestBody(req, modelName: "heartmula")
        XCTAssertEqual(body["cfg_scale"] as? Double, 10)
        XCTAssertEqual(body["temperature"] as? Double, 0.1)
        XCTAssertEqual(body["top_k"] as? Int, 1000)
        XCTAssertEqual(body["duration_seconds"] as? Int, 90 + MusicGenService.lengthGrace(90))
        XCTAssertTrue(MusicGenService.settingsText(req, resolvedSeed: 1, modelName: "m").contains("cfg_scale: 99.0"))
        req.model = .yue2_3B_8bit
        let other = MusicGenService.requestBody(req, modelName: "yue2")
        for key in ["cfg_scale", "temperature", "top_k"] { XCTAssertNil(other[key], key) }
    }

    func testTemplatesKeepOneShortTagPerCategory() {
        for s in MusicPrompt.builtinStyles(for: .heartmula) {
            let tags = s.body.split(separator: ",")
            XCTAssertLessThanOrEqual(tags.count, 8, s.title)
            XCTAssertTrue(tags.allSatisfy { $0.split(separator: " ").count <= 2 }, s.title)
        }
    }

    func testArchitectureRoutesToTheMusicPane() {
        XCTAssertTrue(isMediaModelType("heartmula"))
        XCTAssertEqual(MediaModality(modelType: "heartmula"), .music)
        let models = [ModelInfo(name: "someone/HeartMuLa-RL-3B", quantBits: 8, layers: 0, hiddenSize: 0, vocabSize: 0,
                                contextLength: 0, modelMaxTokens: 0, architecture: "heartmula", capabilities: ["audio", "music"])]
        XCTAssertEqual(CustomMediaModels.musicPreset(for: "someone/HeartMuLa-RL-3B", from: models)?.family, .heartmula)
        XCTAssertEqual(MusicPrompt.builtinLyrics(for: .heartmula), MusicPrompt.builtinLyrics(for: .yue2))
        for s in MusicPrompt.builtinStyles(for: .heartmula) {
            XCTAssertFalse(s.body.contains(", "), "tags are comma-separated without spaces: \(s.title)")
            XCTAssertEqual(s.body, s.body.lowercased(), s.title)
        }
    }

    func testBundleMarkersAreTheFilesTheServerNeeds() {
        let comp = hm.bundle.components[0]
        XCTAssertEqual(DownloadManager.requiredMediaMarker(modelType: "heartmula"), "codec.safetensors")
        let entries: [[String: Any]] = [
            ["path": "README.md", "type": "file", "size": 1000],
            ["path": "LICENSE", "type": "file", "size": 11357],
            ["path": "config.json", "type": "file", "size": 1788],
            ["path": "model.safetensors", "type": "file", "size": 3_700_000_000],
            ["path": "codec.safetensors", "type": "file", "size": 3_319_445_271],
            ["path": "tokenizer.json", "type": "file", "size": 9_085_657],
        ]
        let picked = DownloadManager.selectNeededFiles(from: entries, selection: comp.selection).map(\.0)
        for marker in comp.readyMarkers {
            XCTAssertTrue(picked.contains(marker), "readiness marker \(marker) not downloaded")
        }
        XCTAssertTrue(comp.readyMarkers.contains("codec.safetensors"))
    }
}
