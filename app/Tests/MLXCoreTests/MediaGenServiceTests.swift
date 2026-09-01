import XCTest
import AVFoundation
@testable import MLXCore

/// Tests for the unified media-generation path: image/audio/video now run
/// through the ONE main `mlx-serve` server (registry-hosted) instead of a
/// dedicated `NativeGenServer` subprocess. Covers the pure response-decode
/// contracts + the load→generate→unload residency default.
@MainActor
final class MediaGenServiceTests: XCTestCase {

    // MARK: - Image response decode (the /v1/images/generations contract)

    func testDecodePngB64ExtractsImage() throws {
        let pngBytes = Data([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x01, 0x02])
        let b64 = pngBytes.base64EncodedString()
        let body = try JSONSerialization.data(withJSONObject: ["data": [["b64_json": b64]]])
        let decoded = ImageGenService.decodePngB64(body)
        XCTAssertEqual(decoded, pngBytes)
    }

    func testDecodePngB64RejectsMalformed() {
        XCTAssertNil(ImageGenService.decodePngB64(Data("not json".utf8)))
        let noData = try! JSONSerialization.data(withJSONObject: ["error": "boom"])
        XCTAssertNil(ImageGenService.decodePngB64(noData))
        let emptyArr = try! JSONSerialization.data(withJSONObject: ["data": []])
        XCTAssertNil(ImageGenService.decodePngB64(emptyArr))
    }

    // MARK: - Video request body (the /v1/video/generations REQUEST contract)

    func testRequestBodyTwoStageCarriesPipelineAndGuidance() {
        // The confirmed bug: pipeline/cfg_scale/stg_scale were modeled in the UI
        // (VideoPipelineMode, VideoGenRequest) but never put in the HTTP body —
        // the Quality preset (cfg 3.0, twoStage) silently ran as unguided
        // one-stage. This pins the full wire shape for a .twoStage request.
        var req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 97, fps: 24, mode: .twoStage, steps: 30, cfgScale: 3.0)
        req.stgScale = 1.0
        let body = VideoGenService.requestBody(model: "ltx", prompt: "a prompt", request: req, firstFrameB64: nil)
        XCTAssertEqual(body["pipeline"] as? String, "two_stage")
        XCTAssertEqual(body["cfg_scale"] as? Double, 3.0)
        XCTAssertEqual(body["stg_scale"] as? Double, 1.0)
        XCTAssertEqual(body["steps"] as? Int, 30)
        // The pre-existing fields keep their shape.
        XCTAssertEqual(body["model"] as? String, "ltx")
        XCTAssertEqual(body["prompt"] as? String, "a prompt")
        XCTAssertEqual(body["num_frames"] as? Int, 97)
        XCTAssertEqual(body["height"] as? Int, 480)
        XCTAssertEqual(body["width"] as? Int, 704)
        XCTAssertEqual(body["seed"] as? Int, 42)
        // first_frame_image stays conditional — absent when there's no image.
        XCTAssertNil(body["first_frame_image"])
    }

    func testLivePreviewIsOptInAndCarriesItsShapeWhenAsked() {
        // A preview costs an x0 solve plus a host copy of the previewed frame on
        // EVERY step, so the default has to be the one nobody pays for: absent,
        // which is also the server's own default for any other client. The three
        // fields travel together — `preview_frames`/`preview_max_side` without
        // `preview` are read by nothing.
        func body(_ on: Bool) -> [String: Any] {
            var r = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                    numFrames: 97, fps: 24, mode: .oneStage, steps: 8, cfgScale: 1.0)
            r.livePreview = on

            return VideoGenService.requestBody(model: "m", prompt: "p", request: r,
                                               firstFrameB64: nil)
        }
        XCTAssertFalse(VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                       numFrames: 97, fps: 24, mode: .oneStage, steps: 8,
                                       cfgScale: 1.0).livePreview)
        XCTAssertFalse(VideoGenSettings().livePreview)
        XCTAssertNil(body(false)["preview"])
        XCTAssertNil(body(false)["preview_frames"])
        XCTAssertNil(body(false)["preview_max_side"])
        XCTAssertEqual(body(true)["preview"] as? Bool, true)
        XCTAssertEqual(body(true)["preview_frames"] as? Int, 1)
        XCTAssertEqual(body(true)["preview_max_side"] as? Int, 256)
    }

    func testLivePreviewSurvivesASaveLoadRoundTrip() {
        // The pane's decoder is hand-listed, so a field that is saved but never
        // decoded resets on every launch (the `bestQuality` class).
        var s = VideoGenSettings()
        s.livePreview = true
        let data = try! JSONEncoder().encode(s)
        let back = try! JSONDecoder().decode(VideoGenSettings.self, from: data)
        XCTAssertTrue(back.livePreview)
    }

    func testDiffusionDecoderFieldIsGatedOnThePacksOwnCapability() {
        // `vae_diffusion_decoder.safetensors` ships in the 8-bit LTX-2.5 pack
        // and NOT in the 4-bit one, so the toggle is per preset — and the state
        // survives a preset switch, which is exactly how a field reaches a
        // backend that answers 400 (the H3 `pipeline` class). The conv decoder
        // is the server's default, so the field is absent unless asked for.
        func body(_ model: VideoModelPreset, on: Bool) -> [String: Any] {
            var r = VideoGenRequest(model: model, prompt: "p", width: 384, height: 256,
                                    numFrames: 9, fps: 24, mode: .oneStage, steps: 8, cfgScale: 1.0)
            r.diffusionDecoder = on
            return VideoGenService.requestBody(model: "m", prompt: "p", request: r,
                                               firstFrameB64: nil)
        }
        XCTAssertTrue(VideoModelPreset.ltx25Q8.supportsDiffusionDecoder)
        XCTAssertFalse(VideoModelPreset.ltx25Q4.supportsDiffusionDecoder)
        XCTAssertFalse(VideoModelPreset.ltx23Q4.supportsDiffusionDecoder)
        XCTAssertEqual(body(.ltx25Q8, on: true)["decoder"] as? String, "diffusion")
        XCTAssertNil(body(.ltx25Q8, on: false)["decoder"])
        XCTAssertNil(body(.ltx25Q4, on: true)["decoder"])
        XCTAssertNil(body(.ltx23Q4, on: true)["decoder"])
    }

    func testRef2vaFieldsAreGatedOnTheModelsOwnCapability() {
        // Hiding a control is not the same as not sending its field — the class
        // that made every H3 request carry `pipeline` and 400. FL2VA has no
        // reference conditioning, so a request against it must carry no ref_*
        // field even when the state is populated (a preset switch leaves it).
        let payload = VideoRefPayloads(
            images: ["QQ==", "Qg=="],
            videos: [.init(frames: ["Rg==", "Rw=="], audio: "Uw==")],
            audios: ["VA=="]
        )
        var fl2va = VideoGenRequest(model: .minimaxH3, prompt: "p", width: 256, height: 256,
                                    numFrames: 124, fps: 24, mode: .oneStage, steps: 30, cfgScale: 1.0)
        fl2va.refImageSize = .max
        let fbody = VideoGenService.requestBody(model: "m", prompt: "p", request: fl2va,
                                                firstFrameB64: nil, refs: payload)
        XCTAssertNil(fbody["ref_images"])
        XCTAssertNil(fbody["ref_videos"])
        XCTAssertNil(fbody["ref_audios"])
        XCTAssertNil(fbody["ref_image_size"])

        var ref = VideoGenRequest(model: .minimaxH3Ref2VA, prompt: "p", width: 256, height: 256,
                                  numFrames: 124, fps: 24, mode: .oneStage, steps: 30, cfgScale: 1.0)
        let rbody = VideoGenService.requestBody(model: "m", prompt: "p", request: ref,
                                                firstFrameB64: nil, refs: payload)
        XCTAssertEqual(rbody["ref_images"] as? [String], ["QQ==", "Qg=="])
        XCTAssertEqual(rbody["ref_audios"] as? [String], ["VA=="])
        // A reference video is an OBJECT carrying its own soundtrack, so a
        // clip without audio cannot shift the pairing of the ones that have it.
        let vids = rbody["ref_videos"] as? [[String: Any]]
        XCTAssertEqual(vids?.count, 1)
        XCTAssertEqual(vids?.first?["frames"] as? [String], ["Rg==", "Rw=="])
        XCTAssertEqual(vids?.first?["audio"] as? String, "Uw==")
        // Default sizing is the server's own default, so it is not restated.
        XCTAssertNil(rbody["ref_image_size"])
        ref.refImageSize = .max
        let mbody = VideoGenService.requestBody(model: "m", prompt: "p", request: ref,
                                                firstFrameB64: nil, refs: payload)
        XCTAssertEqual(mbody["ref_image_size"] as? String, "max")

        // Empty payloads send nothing at all, on either pack.
        let ebody = VideoGenService.requestBody(model: "m", prompt: "p", request: ref,
                                                firstFrameB64: nil, refs: VideoRefPayloads())
        XCTAssertNil(ebody["ref_images"])
        XCTAssertNil(ebody["ref_videos"])
        XCTAssertNil(ebody["ref_audios"])

        // A video with no soundtrack omits the key rather than sending null.
        let silent = VideoRefPayloads(videos: [.init(frames: ["Rg=="], audio: nil)])
        let sbody = VideoGenService.requestBody(model: "m", prompt: "p", request: ref,
                                                firstFrameB64: nil, refs: silent)
        let svids = sbody["ref_videos"] as? [[String: Any]]
        XCTAssertEqual(svids?.count, 1)
        XCTAssertNil(svids?.first?["audio"])
    }

    func testRefVideoFrameCountSnapsToTheModelsOwnLadder() {
        // H3's frame ladder is 17k+5 and the server snaps DOWN, so extracting
        // frames the server will throw away is pure payload — a 124-frame clip
        // is ~10 MB of base64 before anything is generated.
        XCTAssertEqual(VideoGenService.refVideoFrameCount(available: 130, cap: 209), 124)
        XCTAssertEqual(VideoGenService.refVideoFrameCount(available: 124, cap: 209), 124)
        XCTAssertEqual(VideoGenService.refVideoFrameCount(available: 22, cap: 209), 22)
        XCTAssertEqual(VideoGenService.refVideoFrameCount(available: 21, cap: 209), 5)
        // The generation's own length is the ceiling: the server truncates a
        // longer reference before snapping it.
        XCTAssertEqual(VideoGenService.refVideoFrameCount(available: 400, cap: 124), 124)
        // Below the floor there is not one latent frame to condition on, so
        // the clip is refused rather than padded into something meaningless.
        XCTAssertEqual(VideoGenService.refVideoFrameCount(available: 4, cap: 209), 0)
        XCTAssertEqual(VideoGenService.refVideoFrameCount(available: 0, cap: 209), 0)
    }

    func testRefFrameTimesRunAtTheModelsFixedFrameRate() {
        // 24 fps is fixed for H3; the timestamps the model reads are derived
        // from the frame INDEX, so sampling at any other rate silently
        // stretches the reference's motion.
        let t = VideoGenService.refFrameTimes(count: 5, fps: 24)
        XCTAssertEqual(t.count, 5)
        XCTAssertEqual(t[0], 0.0, accuracy: 1e-9)
        XCTAssertEqual(t[1], 1.0 / 24.0, accuracy: 1e-9)
        XCTAssertEqual(t[4], 4.0 / 24.0, accuracy: 1e-9)
        XCTAssertTrue(VideoGenService.refFrameTimes(count: 0, fps: 24).isEmpty)
    }

    func testRef2vaPresetIsTheOnlyOneAdvertisingReferences() {
        // The factory takes the flag as a parameter precisely so the two FL2VA
        // presets cannot drift into claiming a capability their DiT lacks.
        XCTAssertTrue(VideoModelPreset.minimaxH3Ref2VA.supportsReferences)
        for p in VideoModelPreset.all where p.id != VideoModelPreset.minimaxH3Ref2VA.id {
            XCTAssertFalse(p.supportsReferences, "\(p.id) must not advertise references")
        }
        // REF2VA has no first/last-frame conditioning and FL2VA has no
        // references — two checkpoints, not two modes of one.
        XCTAssertTrue(VideoModelPreset.all.contains { $0.id == VideoModelPreset.minimaxH3Ref2VA.id })
    }

    func testTurboAndChainingAreFl2vaOnlyAndGateTheirFields() {
        // Capability side: both ride fl2va machinery (the LoRA is untested on
        // the REF2VA DiT; a reference has no keyframe row to chain through),
        // derived in the shared factory so the two FL2VA presets cannot drift.
        XCTAssertTrue(VideoModelPreset.minimaxH3.supportsTurbo)
        XCTAssertTrue(VideoModelPreset.minimaxH3Q4.supportsTurbo)
        XCTAssertFalse(VideoModelPreset.minimaxH3Ref2VA.supportsTurbo)
        XCTAssertTrue(VideoModelPreset.minimaxH3.supportsChainedWindows)
        XCTAssertFalse(VideoModelPreset.minimaxH3Ref2VA.supportsChainedWindows)
        for p in VideoModelPreset.all where p.backend != .minimaxH3 {
            XCTAssertFalse(p.supportsTurbo, "\(p.id) must not advertise turbo")
            XCTAssertFalse(p.supportsChainedWindows, "\(p.id) must not advertise chaining")
        }

        // Wire side: emitted only when engaged AND declared — hiding a control
        // is not the same as not sending its field (the `pipeline` class).
        var req = VideoGenRequest(model: .minimaxH3, prompt: "p", width: 960, height: 544,
                                  numFrames: 124, fps: 24, mode: .oneStage, steps: 8, cfgScale: 1.0)
        var body = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)
        XCTAssertNil(body["turbo"], "off by default")
        XCTAssertNil(body["chain_windows"], "a single window sends nothing")

        req.turbo = true
        req.chainWindows = 2
        body = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)
        XCTAssertEqual(body["turbo"] as? Bool, true)
        XCTAssertEqual(body["chain_windows"] as? Int, 2)

        // A preset switch leaves the state populated; the REF2VA pack must
        // never see either field (the server 400s chaining there by name).
        var ref = VideoGenRequest(model: .minimaxH3Ref2VA, prompt: "p", width: 960, height: 544,
                                  numFrames: 124, fps: 24, mode: .oneStage, steps: 30, cfgScale: 1.0)
        ref.turbo = true
        ref.chainWindows = 3
        body = VideoGenService.requestBody(model: "m", prompt: "p", request: ref, firstFrameB64: nil)
        XCTAssertNil(body["turbo"])
        XCTAssertNil(body["chain_windows"])
    }

    func testTurboBillsTheLoraBesideTheDiT() {
        // Turbo forces the recipe off (fast: false) and adds the resident
        // LoRA to the sampling term. On the REAL packs the staged load floor
        // (max(TE, DiT) + VAEs) already covers it at every servable geometry —
        // billing must still be monotone there — so the exact +lora mechanism
        // is pinned on a mutated preset whose sampling term dominates.
        let base = H3Plan.peakBytes(model: .minimaxH3, width: 960, height: 544,
                                    frames: 124, fast: false)
        let turboReal = H3Plan.peakBytes(model: .minimaxH3, width: 960, height: 544,
                                         frames: 124, fast: false, turbo: true)
        XCTAssertGreaterThanOrEqual(turboReal, base)

        var tiny = VideoModelPreset.minimaxH3
        tiny.stagedPeakGB = 1
        tiny.ditResidentGB = 1
        let b = H3Plan.peakBytes(model: tiny, width: 960, height: 544, frames: 124, fast: false)
        let t = H3Plan.peakBytes(model: tiny, width: 960, height: 544, frames: 124, fast: false, turbo: true)
        XCTAssertEqual(t - b, H3Plan.turboLoraBytes)
    }

    func testStreamStartFailureSurfacesTheServersOwnMessage() {
        // The server's named 400s exist to tell the user what to do ("this
        // pack has no turbo_lora.safetensors — download …"); the stream path
        // used to throw a hardcoded "stream start failed" without reading the
        // body, so the Failed card pointed at nothing (live 2026-08-06).
        let named = #"{"error":{"message":"this pack has no turbo_lora.safetensors — download it"}}"#
        XCTAssertEqual(APIError.errorDetail(fromBody: Data(named.utf8)),
                       "this pack has no turbo_lora.safetensors — download it")
        // A non-JSON body degrades to a trimmed snippet, and an empty one to
        // the generic line — never an empty detail.
        XCTAssertEqual(APIError.errorDetail(fromBody: Data("  plain text  ".utf8)), "plain text")
        XCTAssertEqual(APIError.errorDetail(fromBody: Data()), "stream start failed")
        // And the rendering carries a separator — "mlx-servestream start
        // failed" is what shipping without one looked like.
        let desc = APIError.badStatus(code: 400, detail: "x").errorDescription ?? ""
        XCTAssertTrue(desc.contains("mlx-serve: x"), desc)
    }

    func testH3FastRecipeDefaultOnWithOffSwitch() {
        // David's eyeball verdict on the same-seed 768p capstone pair: the
        // fast recipe (server-side step cache + attention broadcast, 2.83x)
        // is DEFAULT-ON, so the app sends NO field in the default case and
        // "fast": false only when the user opts into max quality.
        var req = VideoGenRequest(model: .minimaxH3, prompt: "p", width: 256, height: 256,
                                  numFrames: 56, fps: 24, mode: .oneStage, steps: 30, cfgScale: 1.0)
        var body = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)
        XCTAssertNil(body["fast"], "default must ride the server's fast default, not restate it")
        req.bestQuality = true
        body = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)
        XCTAssertEqual(body["fast"] as? Bool, false)
        // LTX has no fast recipe — the field never appears, toggled or not.
        var ltx = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 9, fps: 24, mode: .oneStage, steps: 8, cfgScale: 1.0)
        ltx.bestQuality = true
        let lbody = VideoGenService.requestBody(model: "m", prompt: "p", request: ltx, firstFrameB64: nil)
        XCTAssertNil(lbody["fast"])
    }

    // MARK: - Prompt guidance (the pane's examples / placeholder / hint / link)

    func testPromptFormatFollowsTheModelsOwnCapabilities() {
        // One chokepoint, derived from what the preset already declares — an
        // id-sniffing copy is how the pane ends up giving LTX advice on an H3
        // model, which is the state this replaced.
        XCTAssertEqual(VideoModelPreset.ltx23Q4.promptFormat, .ltx)
        XCTAssertEqual(VideoModelPreset.minimaxH3.promptFormat, .h3Base)
        XCTAssertEqual(VideoModelPreset.minimaxH3Q4.promptFormat, .h3Base)
        XCTAssertEqual(VideoModelPreset.minimaxH3Ref2VA.promptFormat, .h3Reference)
        // Every shipped preset resolves — a new one can't land without a format.
        for p in VideoModelPreset.all {
            XCTAssertEqual(p.promptFormat == .ltx, p.backend == .ltx, "\(p.id)")
        }
    }

    func testEveryH3ExampleCarriesItsFormatsSectionsInOrder() {
        // The format IS the feature: H3 was trained on labelled documents, and
        // an example that drops a section (or writes them out of order) teaches
        // the shape wrong. Order matters — the release's own guides fix it.
        for ex in H3PromptExamples.h3Base {
            XCTAssertTrue(H3PromptExamples.carriesSections(ex.body, H3PromptExamples.baseSections),
                          "base example '\(ex.title)' is missing a section or has them out of order")
        }
        for ex in H3PromptExamples.h3Reference {
            XCTAssertTrue(H3PromptExamples.carriesSections(ex.body, H3PromptExamples.referenceSections),
                          "reference example '\(ex.title)' is missing a section or has them out of order")
        }
        // retention_analysis is the only place a reference's ROLE is stated, so
        // every reference example must actually use one of the fixed markers.
        let markers = ["fully_preserved", "partially_preserved", "attribute_transfer",
                       "weak_reference", "fully_copy", "partially_copy", "reference -"]
        for ex in H3PromptExamples.h3Reference {
            XCTAssertTrue(markers.contains { ex.body.contains($0) },
                          "reference example '\(ex.title)' states no retention marker")
        }
    }

    func testNoLtxExampleLeaksIntoTheH3ListsOrViceVersa() {
        let ltxBodies = Set(H3PromptExamples.ltx.map(\.body))
        let h3Bodies = Set((H3PromptExamples.h3Base + H3PromptExamples.h3Reference).map(\.body))
        XCTAssertTrue(ltxBodies.isDisjoint(with: h3Bodies))
        // An LTX example is prose — it must carry none of H3's section labels,
        // or it would read as format guidance for a model that has no format.
        for ex in H3PromptExamples.ltx {
            for label in H3PromptExamples.referenceSections + H3PromptExamples.baseSections {
                XCTAssertFalse(ex.body.contains(label), "LTX example '\(ex.title)' carries \(label)")
            }
        }
        // And the base list must not carry the full-reference-only sections:
        // those name attachments an FL2VA pack cannot take.
        for ex in H3PromptExamples.h3Base {
            for label in ["subject_definitions:", "retention_analysis:", "detailed_description:"] {
                XCTAssertFalse(ex.body.contains(label), "base example '\(ex.title)' carries \(label)")
            }
        }
        XCTAssertEqual(H3PromptExamples.examples(for: .ltx).count, H3PromptExamples.ltx.count)
        XCTAssertEqual(H3PromptExamples.examples(for: .h3Base).count, H3PromptExamples.h3Base.count)
        XCTAssertEqual(H3PromptExamples.examples(for: .h3Reference).count, H3PromptExamples.h3Reference.count)
    }

    func testHintAndTipsLinkAreSelectedPerBackend() {
        // LTX keeps its own guidance verbatim, including the 15-word floor.
        XCTAssertEqual(H3PromptExamples.hint(for: .ltx, prompt: "a cat"),
                       "LTX-Video performs best with detailed 4–8 sentence prompts. Try Examples or Prompt tips above.")
        XCTAssertNil(H3PromptExamples.hint(for: .ltx, prompt: H3PromptExamples.ltx[0].body))
        XCTAssertNil(H3PromptExamples.hint(for: .ltx, prompt: ""), "an empty field shows the placeholder, not a warning")

        // H3 flags the missing FORMAT, which is the thing a user cannot guess —
        // and never quotes LTX's sentence advice at an H3 model.
        let bare = "a low aerial shot over a volcanic coastline at golden hour"
        let baseHint = H3PromptExamples.hint(for: .h3Base, prompt: bare)
        XCTAssertNotNil(baseHint)
        XCTAssertTrue(baseHint!.contains("integrated_multimodal_description:"))
        XCTAssertFalse(baseHint!.contains("LTX"))
        let refHint = H3PromptExamples.hint(for: .h3Reference, prompt: bare)
        XCTAssertNotNil(refHint)
        XCTAssertTrue(refHint!.contains("subject_definitions:"))
        XCTAssertFalse(refHint!.contains("LTX"))

        // A prompt already written in the format is never nagged.
        XCTAssertNil(H3PromptExamples.hint(for: .h3Base, prompt: H3PromptExamples.h3Base[0].body))
        XCTAssertNil(H3PromptExamples.hint(for: .h3Reference, prompt: H3PromptExamples.h3Reference[0].body))

        // "Prompt tips" must not send an H3 user to LTX's site.
        XCTAssertEqual(H3PromptExamples.tipsURL(for: .ltx).host, "docs.ltx.video")
        for f in [VideoPromptFormat.h3Base, .h3Reference] {
            let url = H3PromptExamples.tipsURL(for: f)
            XCTAssertNotEqual(url.host, "docs.ltx.video")
            // MiniMax's own guides — we do not ship a copy of them.
            XCTAssertTrue(url.absoluteString.contains("MiniMax-H3"), "\(url)")
        }

        // The placeholder NAMES the labels — it is the only surface that can.
        XCTAssertTrue(H3PromptExamples.placeholder(for: .h3Base).contains("integrated_multimodal_description:"))
        XCTAssertTrue(H3PromptExamples.placeholder(for: .h3Reference).contains("retention_analysis:"))
        XCTAssertFalse(H3PromptExamples.placeholder(for: .ltx).contains("integrated_multimodal_description:"))

        // It renders INSIDE the 110pt prompt editor, so it has a length budget:
        // the first draft listed one section per line and was clipped exactly
        // where the last labels were — the part that can't be guessed. ~320
        // chars is about six lines at the pane's minimum width; a longer
        // placeholder needs a taller editor, not a silent clip.
        for f in [VideoPromptFormat.ltx, .h3Base, .h3Reference] {
            let p = H3PromptExamples.placeholder(for: f)
            XCTAssertLessThanOrEqual(p.count, 320, "\(f) placeholder is too long for the prompt editor")
            XCTAssertFalse(p.contains("\n"), "\(f) placeholder must wrap, not hard-break — line breaks blow the height budget")
        }
    }

    func testH3ExamplesMatchTheMeasuredContextIRProfile() {
        // These bars come from MiniMax's OWN expansion stage, not from taste:
        // five H3-Context-IR expansions (2026-08-06) run ~1000-1530 chars of
        // body for a 5-second shot, while the hand-written examples they
        // replaced ran ~710. The main body IS the prompt, and the thin version
        // is what a transcription of the guide drifts toward.
        func body(_ text: String, after label: String, upTo next: String) -> String {
            guard let a = text.range(of: label), let b = text.range(of: next, range: a.upperBound..<text.endIndex)
            else { return "" }
            return String(text[a.upperBound..<b.lowerBound])
        }
        for ex in H3PromptExamples.h3Base {
            let main = body(ex.body, after: "integrated_multimodal_description:", upTo: "overall_soundscape:")
            XCTAssertGreaterThan(main.count, 700, "base example '\(ex.title)' body is too thin")
            let sound = body(ex.body, after: "overall_soundscape:", upTo: "non_diegetic_music:")
            XCTAssertGreaterThan(sound.count, 150, "base example '\(ex.title)' soundscape is too abstract")
            // Laid out like the reference examples: label on its own line,
            // blank line between fields. The production expansion stage emits
            // one inline run per field, which is a WALL in the prompt box for a
            // ~1500-char description; MiniMax's own written guide shows blank
            // lines, so this stays in-format and is the readable half of it.
            for label in H3PromptExamples.baseSections {
                XCTAssertTrue(ex.body.contains("\(label)\n"),
                              "base example '\(ex.title)' keeps \(label) inline with its value")
            }
            for label in H3PromptExamples.baseSections.dropFirst() {
                XCTAssertTrue(ex.body.contains("\n\n\(label)"),
                              "base example '\(ex.title)' has no blank line before \(label)")
            }
        }
        for ex in H3PromptExamples.h3Reference {
            let main = body(ex.body, after: "detailed_description:", upTo: "overall_soundscape:")
            XCTAssertGreaterThan(main.count, 1000, "reference example '\(ex.title)' detailed_description is too thin")
        }
        // Camera behaviour is NAMED (motion type), not just modified. Real
        // expansions say "static shot" / "slow push in"; the version this
        // replaced only ever said "with small amplitude at slow speed".
        let motion = ["static shot", "push in", "pull out", "tracking shot", "pedestal",
                      "zoom in", "zoom out", "pan left", "pan right", "truck", "tilt", "arc shot", "roll"]
        for ex in H3PromptExamples.h3Base + H3PromptExamples.h3Reference {
            let low = ex.body.lowercased()
            XCTAssertTrue(motion.contains { low.contains($0) },
                          "example '\(ex.title)' names no camera motion type")
        }
    }

    func testCarriesSectionsRejectsOutOfOrderAndMissingLabels() {
        // The helper is what the example guard rests on, so pin its two failure
        // modes directly rather than trusting it via the examples.
        let ok = "subject_definitions:\na\n\nsummary:\nb"
        XCTAssertTrue(H3PromptExamples.carriesSections(ok, ["subject_definitions:", "summary:"]))
        XCTAssertFalse(H3PromptExamples.carriesSections(ok, ["summary:", "subject_definitions:"]),
                       "reversed order must fail")
        XCTAssertFalse(H3PromptExamples.carriesSections("summary:\nb", ["subject_definitions:", "summary:"]),
                       "a missing label must fail")
    }

    func testRequestBodyPipelineModeMapping() {
        func pipeline(_ mode: VideoPipelineMode) -> String? {
            let req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                      numFrames: 9, fps: 24, mode: mode, steps: 8, cfgScale: 1.0)
            return VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)["pipeline"] as? String
        }
        XCTAssertEqual(pipeline(.oneStage), "one_stage")
        XCTAssertEqual(pipeline(.twoStage), "two_stage")
        XCTAssertEqual(pipeline(.twoStageHQ), "two_stage_hq")
    }

    func testRequestBodyCarriesLoraWhenSet() {
        var req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 9, fps: 24, mode: .oneStage, steps: 8, cfgScale: 1.0)
        req.loras = [LoraAdapter(path: "/tmp/style.safetensors", scale: 0.8)]
        let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)
        XCTAssertEqual(body["lora_paths"] as? [String], ["/tmp/style.safetensors"])
        XCTAssertEqual(body["lora_scales"] as? [Double], [0.8])
        // Stacking: a second adapter appends to both arrays, in order.
        req.loras.append(LoraAdapter(path: "/tmp/second.safetensors", scale: 1.2))
        let stacked = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)
        XCTAssertEqual(stacked["lora_paths"] as? [String], ["/tmp/style.safetensors", "/tmp/second.safetensors"])
        XCTAssertEqual(stacked["lora_scales"] as? [Double], [0.8, 1.2])
        // No LoRA → fields absent (missing lora_paths means detach server-side).
        req.loras = []
        let bare = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)
        XCTAssertNil(bare["lora_paths"])
        XCTAssertNil(bare["lora_scales"])
        // A half-filled row (no path chosen yet) is dropped, not sent as "".
        req.loras = [LoraAdapter(path: "", scale: 1.0)]
        let empty = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil)
        XCTAssertNil(empty["lora_paths"])
        XCTAssertNil(empty["lora_scales"])
    }

    func testCancelledErrorsMapToCancellationNotFailure() {
        // A user cancel surfaces from URLSession as URLError.cancelled, NOT
        // CancellationError — treating it as generic failure showed "Failed"
        // after every Cancel click.
        XCTAssertTrue(VideoGenService.isCancellation(CancellationError()))
        XCTAssertTrue(VideoGenService.isCancellation(URLError(.cancelled)))
        XCTAssertFalse(VideoGenService.isCancellation(URLError(.timedOut)))
        XCTAssertFalse(VideoGenService.isCancellation(APIError.badStatus(code: 500, detail: "x")))
    }

    func testResidencyEntryMatching() {
        // Discovered two-level id ("org/model") — the normal pull layout.
        XCTAssertTrue(VideoGenService.entryMatches(
            id: "dgrauet/ltx-2.3-mlx-q4", repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4"))
        // Path-registered ids: absolute path or bare basename.
        XCTAssertTrue(VideoGenService.entryMatches(
            id: "/x/models/dgrauet/ltx-2.3-mlx-q4", repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4"))
        XCTAssertTrue(VideoGenService.entryMatches(
            id: "ltx-2.3-mlx-q4", repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4"))
        // A different model never matches.
        XCTAssertFalse(VideoGenService.entryMatches(
            id: "google/gemma-3-12b-it-4bit", repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4"))
        XCTAssertFalse(VideoGenService.entryMatches(
            id: "org/other", repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: nil))
    }

    func testResidencyComputedFromModelsListNotProps() {
        // The live bug: GPU memory came from /props, which 503s on a headless
        // gen-only boot ("No default model configured") → "GPU memory 0 MB"
        // while 30 GB of LTX was resident. Residency must reduce over the
        // /v1/models snapshot (a no-model endpoint) instead.
        let ltx = APIClient.parseModelInfo([
            "id": "dgrauet/ltx-2.3-mlx-q4", "loaded": true,
            "bytes_resident": UInt64(31_801_302_892),
        ])
        let chatLoaded = APIClient.parseModelInfo([
            "id": "google/gemma-4-12b", "loaded": true,
            "bytes_resident": UInt64(8_000_000_000),
        ])
        let chatUnloaded = APIClient.parseModelInfo([
            "id": "google/gemma-3-4b", "loaded": false, "bytes_resident": UInt64(0),
        ])

        let r = VideoGenService.residency(
            from: [ltx, chatLoaded, chatUnloaded],
            repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4")
        XCTAssertTrue(r.loaded)
        XCTAssertEqual(r.bytesResident, 31_801_302_892)
        // GPU total sums every LOADED entry — unloaded stubs contribute nothing.
        XCTAssertEqual(r.gpuResidentBytes, 39_801_302_892)

        // Pane's model absent from the registry → not loaded, but the total
        // still reports who holds the GPU.
        let miss = VideoGenService.residency(
            from: [chatLoaded], repo: "dgrauet/ltx-2.3-mlx-q4", dirBasename: "ltx-2.3-mlx-q4")
        XCTAssertFalse(miss.loaded)
        XCTAssertEqual(miss.gpuResidentBytes, 8_000_000_000)
    }

    func testRequestBodyIncludesFirstFrameWhenPresent() {
        let req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 9, fps: 24, mode: .oneStage, steps: 8, cfgScale: 1.0)
        let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: "QUJD")
        XCTAssertEqual(body["first_frame_image"] as? String, "QUJD")
    }

    // MARK: - Audio-to-video request contract

    func testRequestBodyCarriesAudioWhenPresent() {
        let req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 97, fps: 24, mode: .twoStage, steps: 30, cfgScale: 3.0)
        let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req,
                                               firstFrameB64: nil, audioB64: "V0FW")
        XCTAssertEqual(body["audio"] as? String, "V0FW")
        // absent when there's no clip — the server treats presence as intent
        let none = VideoGenService.requestBody(model: "m", prompt: "p", request: req,
                                               firstFrameB64: nil, audioB64: nil)
        XCTAssertNil(none["audio"])
    }

    func testRequestBodyAudioForcesTwoStageAndReferenceGuidance() {
        // a2vid is two-stage only (the server 400s one_stage+audio). A Fast
        // (one-stage) preset with a clip attached must upgrade to two_stage AND
        // drop its one-stage guidance values (cfg 1.0 would run stage 1
        // unguided) so the server's reference defaults (cfg 3/7) apply.
        let req = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                  numFrames: 97, fps: 24, mode: .oneStage, steps: 12, cfgScale: 1.0)
        let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req,
                                               firstFrameB64: nil, audioB64: "V0FW")
        XCTAssertEqual(body["pipeline"] as? String, "two_stage")
        XCTAssertNil(body["cfg_scale"])
        XCTAssertNil(body["stg_scale"])
        // An explicit two-stage request keeps the user's guidance untouched.
        var hq = VideoGenRequest(model: .ltx23Q4, prompt: "p", width: 704, height: 480,
                                 numFrames: 97, fps: 24, mode: .twoStageHQ, steps: 15, cfgScale: 4.0)
        hq.stgScale = 0.5
        let hqBody = VideoGenService.requestBody(model: "m", prompt: "p", request: hq,
                                                 firstFrameB64: nil, audioB64: "V0FW")
        XCTAssertEqual(hqBody["pipeline"] as? String, "two_stage_hq")
        XCTAssertEqual(hqBody["cfg_scale"] as? Double, 4.0)
        XCTAssertEqual(hqBody["stg_scale"] as? Double, 0.5)
    }

    func testAudioFileToWavBase64TranscodesToPcm16Wav() throws {
        // Write a float32 WAV via AVAudioFile (any AVFoundation-readable format
        // works — this pins the transcode-to-PCM16-WAV contract the Zig server
        // parses), then round-trip through the helper.
        let dir = FileManager.default.temporaryDirectory
        let src = dir.appendingPathComponent("a2v_test_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: src) }
        let sr = 22050.0
        // Scope the writer: AVAudioFile flushes its header on dealloc (there is
        // no explicit close on this deployment target) — reading before the
        // writer dies sees an empty file.
        try autoreleasepool {
            let fmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sr, channels: 1, interleaved: false)!
            let file = try AVAudioFile(forWriting: src, settings: fmt.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
            let n: AVAudioFrameCount = 22050 // 1 s
            let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: n)!
            buf.frameLength = n
            for i in 0..<Int(n) {
                buf.floatChannelData![0][i] = 0.4 * sin(2.0 * .pi * 440.0 * Float(i) / Float(sr))
            }
            try file.write(from: buf)
        }

        let b64 = VideoGenService.audioFileToWavBase64(path: src.path)
        let wav = try XCTUnwrap(b64.flatMap { Data(base64Encoded: $0) })
        // RIFF/WAVE with a PCM16 fmt chunk at the source rate.
        XCTAssertEqual(String(data: wav.prefix(4), encoding: .ascii), "RIFF")
        XCTAssertEqual(String(data: wav.subdata(in: 8..<12), encoding: .ascii), "WAVE")
        let audioFormat = wav.subdata(in: 20..<22).withUnsafeBytes { $0.load(as: UInt16.self) }
        let bits = wav.subdata(in: 34..<36).withUnsafeBytes { $0.load(as: UInt16.self) }
        let rate = wav.subdata(in: 24..<28).withUnsafeBytes { $0.load(as: UInt32.self) }
        XCTAssertEqual(audioFormat, 1) // PCM
        XCTAssertEqual(bits, 16)
        XCTAssertEqual(rate, 22050)
        XCTAssertGreaterThan(wav.count, 44 + 20000) // ~1 s of PCM16 mono
        // Unreadable path → nil, never a throw/crash.
        XCTAssertNil(VideoGenService.audioFileToWavBase64(path: "/nonexistent/clip.m4a"))
    }

    func testFramesCoveringAudioDurationSnapsUpOnLadder() {
        // Attaching a clip auto-suggests a frame count that COVERS it: the
        // smallest 8N+1 ladder value ≥ duration*fps, capped at the model max.
        let m = VideoModelPreset.ltx23Q4
        XCTAssertEqual(m.framesCovering(durationSeconds: 2.0), 49)   // 48 frames → 49
        XCTAssertEqual(m.framesCovering(durationSeconds: 0.1), m.frameOptions.first) // tiny clip → floor
        XCTAssertEqual(m.framesCovering(durationSeconds: 3600), m.frameOptions.last) // longer than cap → max
        XCTAssertNil(m.framesCovering(durationSeconds: 0))           // no/empty clip → no suggestion
    }

    // MARK: - Video response decode (the /v1/video/generations contract)

    func testDecodeFramesParsesRgb8Body() {
        let frames = 2, w = 2, h = 2
        let rgb = Data(repeating: 7, count: frames * w * h * 3)
        let obj: [String: Any] = [
            "format": "rgb8", "frames": frames, "height": h, "width": w,
            "fps": 24, "data": rgb.base64EncodedString(),
        ]
        let decoded = VideoGenService.decodeFrames(obj)
        XCTAssertEqual(decoded?.frames, frames)
        XCTAssertEqual(decoded?.rgb.count, frames * w * h * 3)
    }

    func testDecodeFramesRejectsSizeMismatch() {
        // rgb byte count must equal frames*h*w*3, else the body is corrupt.
        let obj: [String: Any] = [
            "format": "rgb8", "frames": 2, "height": 2, "width": 2,
            "data": Data(repeating: 1, count: 8).base64EncodedString(),  // wrong size
        ]
        XCTAssertNil(VideoGenService.decodeFrames(obj))
    }

    func testDecodeFramesParsesOptionalAudioTrack() {
        let frames = 2, w = 2, h = 2
        let rgb = Data(repeating: 7, count: frames * w * h * 3)
        let pcm = Data(repeating: 3, count: 320 * 2 * 2)  // 320 stereo frames, s16le
        let obj: [String: Any] = [
            "format": "rgb8", "frames": frames, "height": h, "width": w, "fps": 24,
            "data": rgb.base64EncodedString(),
            "audio_format": "pcm_s16le", "audio_sample_rate": 16000, "audio_channels": 2,
            "audio_data": pcm.base64EncodedString(),
        ]
        let decoded = VideoGenService.decodeFrames(obj)
        XCTAssertEqual(decoded?.audioPCM?.count, pcm.count)
        XCTAssertEqual(decoded?.audioSampleRate, 16000)
        XCTAssertEqual(decoded?.audioChannels, 2)
    }

    func testDecodeFramesAudioAbsentLeavesPcmNil() {
        // A video-only body (no audio fields) must still decode, with no audio.
        let obj: [String: Any] = [
            "format": "rgb8", "frames": 1, "height": 2, "width": 2, "fps": 24,
            "data": Data(repeating: 7, count: 12).base64EncodedString(),
        ]
        let decoded = VideoGenService.decodeFrames(obj)
        XCTAssertNotNil(decoded)
        XCTAssertNil(decoded?.audioPCM)
    }

    func testRefPayloadsBuildFromREALFilesOnDisk() throws {
        // Every other ref2va test constructs `VideoRefPayloads` by hand, so the
        // half a user actually exercises — picked file on disk → decode →
        // base64 on the wire — was covered nowhere. That half is the one with
        // AVAssetImageGenerator, JPEG re-encoding and audio extraction in it.
        let tmp = FileManager.default.temporaryDirectory
        let stem = UUID().uuidString

        // A real clip WITH a soundtrack: 2 s so the ladder has room (5 frames
        // needs ~0.21 s, the next rung 22 needs ~0.92 s).
        let fps = 24, w = 32, h = 32, nf = 48
        let rgb = Data(repeating: 90, count: nf * w * h * 3)
        let sr = 16000, ch = 2, aFrames = sr * 2
        var pcm = Data(count: aFrames * ch * 2)
        pcm.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Int16.self)
            for i in 0..<aFrames { let v = Int16(1500.0 * sin(Double(i) * 0.05)); p[i*2] = v; p[i*2+1] = v }
        }
        let clip = tmp.appendingPathComponent("mlxserve-ref-\(stem).mp4")
        try VideoGenService.writeMP4(rgb: rgb, frames: nf, width: w, height: h, fps: fps, to: clip,
                                     audioPCM: pcm, audioSampleRate: sr, audioChannels: ch)

        // A real PNG.
        let png = tmp.appendingPathComponent("mlxserve-ref-\(stem).png")
        let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: 64, pixelsHigh: 64,
                                   bitsPerSample: 8, samplesPerPixel: 3, hasAlpha: false, isPlanar: false,
                                   colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        try XCTUnwrap(rep.representation(using: .png, properties: [:])).write(to: png)
        defer { for u in [clip, png] { try? FileManager.default.removeItem(at: u) } }

        var req = VideoGenRequest(model: .minimaxH3Ref2VA, prompt: "p", width: 960, height: 544,
                                  numFrames: 124, fps: 24, mode: .oneStage, steps: 30, cfgScale: 1.0)
        req.refImagePaths = [png.path]
        req.refVideoPaths = [clip.path]
        req.refAudioPaths = [clip.path]      // audio is extracted from any AV-readable file

        let refs = try XCTUnwrap(VideoGenService.refPayloads(for: req), "a readable picked file must not fail the build")

        // Image rides through UNCHANGED — the server resizes, so re-encoding here
        // would only throw away what the "max" sizing mode exists to keep.
        XCTAssertEqual(refs.images.count, 1)
        XCTAssertEqual(Data(base64Encoded: refs.images[0])?.prefix(4), Data([0x89, 0x50, 0x4E, 0x47]), "not a PNG")

        // Clip → JPEG frames on the server's own 17k+5 ladder, plus its sound.
        XCTAssertEqual(refs.videos.count, 1)
        let v = refs.videos[0]
        XCTAssertGreaterThanOrEqual(v.frames.count, 5)
        XCTAssertEqual(v.frames.count % 17, 5, "frame count \(v.frames.count) is off the 17k+5 ladder the server snaps to")
        XCTAssertEqual(Data(base64Encoded: v.frames[0])?.prefix(2), Data([0xFF, 0xD8]), "frames must be JPEG, not PNG")
        let sound = try XCTUnwrap(v.audio, "the clip has a soundtrack, so it must ride as the video's own audio")
        XCTAssertEqual(Data(base64Encoded: sound)?.prefix(4), Data("RIFF".utf8), "soundtrack is not a WAV")

        XCTAssertEqual(refs.audios.count, 1)
        XCTAssertEqual(Data(base64Encoded: refs.audios[0])?.prefix(4), Data("RIFF".utf8))

        // And the wire body actually carries them.
        let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req, firstFrameB64: nil, refs: refs)
        XCTAssertEqual((body["ref_images"] as? [String])?.count, 1)
        XCTAssertEqual((body["ref_videos"] as? [[String: Any]])?.count, 1)
        XCTAssertEqual((body["ref_audios"] as? [String])?.count, 1)

        // Same picked files against an FL2VA preset resolve to NOTHING — the
        // capability gate, proven on real input rather than on a struct.
        var fl = req; fl.model = .minimaxH3
        let none = try XCTUnwrap(VideoGenService.refPayloads(for: fl))
        XCTAssertTrue(none.images.isEmpty && none.videos.isEmpty && none.audios.isEmpty)
    }

    func testRefPayloadsFailRatherThanSilentlyDroppingAnUnreadableFile() throws {
        // "Generated, but quietly without your reference" is the worst outcome:
        // it looks like the feature not working rather than the file not being
        // readable, and it costs a full generation to find out.
        var req = VideoGenRequest(model: .minimaxH3Ref2VA, prompt: "p", width: 960, height: 544,
                                  numFrames: 124, fps: 24, mode: .oneStage, steps: 30, cfgScale: 1.0)
        req.refImagePaths = ["/nonexistent/\(UUID().uuidString).png"]
        XCTAssertNil(VideoGenService.refPayloads(for: req))

        var bad = VideoGenRequest(model: .minimaxH3Ref2VA, prompt: "p", width: 960, height: 544,
                                  numFrames: 124, fps: 24, mode: .oneStage, steps: 30, cfgScale: 1.0)
        bad.refVideoPaths = ["/nonexistent/\(UUID().uuidString).mp4"]
        XCTAssertNil(VideoGenService.refPayloads(for: bad))
    }

    func testWriteMP4WithAudioProducesAnAudioTrack() async throws {
        let frames = 3, w = 16, h = 16, fps = 24
        let rgb = Data(repeating: 120, count: frames * w * h * 3)
        // 0.25s of a quiet tone, 16 kHz stereo s16le.
        let sr = 16000, ch = 2, nFrames = sr / 4
        var pcm = Data(count: nFrames * ch * 2)
        pcm.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Int16.self)
            for i in 0..<nFrames {
                let v = Int16(2000.0 * sin(Double(i) * 0.2))
                p[i * 2] = v; p[i * 2 + 1] = v
            }
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-audiomux-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: fps, to: url,
                                     audioPCM: pcm, audioSampleRate: sr, audioChannels: ch)

        let asset = AVURLAsset(url: url)
        let videoTracks = try await asset.loadTracks(withMediaType: .video)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        XCTAssertEqual(videoTracks.count, 1, "expected one video track")
        XCTAssertEqual(audioTracks.count, 1, "audio track missing — mux did not add sound")
    }

    func testWriteMP4WithAudioDoesNotDeadlockAtRealisticScale() throws {
        // A multi-input AVAssetWriter deadlocks when every video frame is pushed
        // before any audio: the muxer stops accepting video (isReadyForMoreMediaData
        // stays false) to bound how far video can lead the still-empty audio track,
        // while the audio is only appended AFTER the video loop — which never ends.
        // Toy-scale clips (a few tiny frames) stay under the muxer's backpressure
        // window and falsely pass, so this reproduces at the ~97-frame scale a real
        // LTX clip hits. A deadlock surfaces here as a wait() timeout, not a hang.
        let frames = 97, w = 256, h = 256, fps = 24
        let rgb = Data(repeating: 120, count: frames * w * h * 3)
        let sr = 16000, ch = 2, nAudio = sr * frames / fps
        var pcm = Data(count: nAudio * ch * 2)
        pcm.withUnsafeMutableBytes { raw in
            let p = raw.bindMemory(to: Int16.self)
            for i in 0..<nAudio {
                let v = Int16(2000.0 * sin(Double(i) * 0.2))
                p[i * 2] = v; p[i * 2 + 1] = v
            }
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-deadlock-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        let done = expectation(description: "writeMP4 completes (no mux deadlock)")
        let muxError = MuxErrorBox()
        Thread.detachNewThread {
            do {
                try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: fps, to: url,
                                             audioPCM: pcm, audioSampleRate: sr, audioChannels: ch)
            } catch { muxError.value = error }
            done.fulfill()
        }
        wait(for: [done], timeout: 30)
        XCTAssertNil(muxError.value, "writeMP4 threw while muxing audio")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "no mp4 written")
    }

    /// Tiny boxed error holder so the detached mux thread can hand a failure back
    /// to the (sendable-checked) test closure.
    private final class MuxErrorBox: @unchecked Sendable { var value: Error? }

    func testWriteMP4WithSubFramePCMCompletesAtRealisticScale() throws {
        // A non-empty PCM payload smaller than one audio frame (3 bytes < the
        // 4-byte stereo s16 frame) yields zero appendable frames. appendAudio's
        // `guard numFrames > 0 else { return }` used to bail WITHOUT marking the
        // audio input finished — leaving a starved, never-finished sibling input
        // that wedges the video loop (same multi-input AVAssetWriter
        // backpressure class as the append-order deadlock above). Realistic
        // frame count so the backpressure window is actually exceeded; a
        // deadlock surfaces as a wait() timeout, not a hang.
        let frames = 97, w = 256, h = 256, fps = 24
        let rgb = Data(repeating: 120, count: frames * w * h * 3)
        let pcm = Data([1, 2, 3])  // non-empty, sub-frame
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-subframe-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }

        let done = expectation(description: "writeMP4 completes with sub-frame PCM")
        let muxError = MuxErrorBox()
        Thread.detachNewThread {
            do {
                try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: fps, to: url,
                                             audioPCM: pcm, audioSampleRate: 16000, audioChannels: 2)
            } catch { muxError.value = error }
            done.fulfill()
        }
        wait(for: [done], timeout: 30)
        XCTAssertNil(muxError.value, "writeMP4 threw on sub-frame PCM")
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "no mp4 written")
    }

    func testWriteMP4WithZeroAudioChannelsSkipsAudio() async throws {
        // audio_channels is SERVER-controlled: 0 must not divide-by-zero
        // (bytesPerFrame = 2 * channels) or wedge the mux — the audio input is
        // skipped entirely for invalid channels/sampleRate.
        let frames = 3, w = 16, h = 16
        let rgb = Data(repeating: 90, count: frames * w * h * 3)
        let pcm = Data(repeating: 1, count: 3200)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-zerochan-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: 24, to: url,
                                     audioPCM: pcm, audioSampleRate: 16000, audioChannels: 0)
        let asset = AVURLAsset(url: url)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        XCTAssertEqual(audioTracks.count, 0, "invalid channel count must skip the audio track")
    }

    func testDecodeFramesDropsAudioWithInvalidChannels() {
        // Same server-controlled field at the decode layer: a body claiming
        // audio_channels 0 parses (video is fine) but the audio is dropped.
        let obj: [String: Any] = [
            "format": "rgb8", "frames": 1, "height": 2, "width": 2, "fps": 24,
            "data": Data(repeating: 7, count: 12).base64EncodedString(),
            "audio_format": "pcm_s16le", "audio_sample_rate": 16000, "audio_channels": 0,
            "audio_data": Data(repeating: 3, count: 64).base64EncodedString(),
        ]
        let decoded = VideoGenService.decodeFrames(obj)
        XCTAssertNotNil(decoded)
        XCTAssertNil(decoded?.audioPCM, "audio with 0 channels must be dropped")
    }

    func testWriteMP4WithoutAudioHasNoAudioTrack() async throws {
        let frames = 2, w = 16, h = 16
        let rgb = Data(repeating: 90, count: frames * w * h * 3)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mlxserve-noaudio-\(UUID().uuidString).mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        try VideoGenService.writeMP4(rgb: rgb, frames: frames, width: w, height: h, fps: 24, to: url)
        let asset = AVURLAsset(url: url)
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        XCTAssertEqual(audioTracks.count, 0)
    }

    // MARK: - Model resolution (moved from NativeGenServer to ServerManager)

    func testResolveModelDirMissingRepoIsNil() {
        XCTAssertNil(ServerManager.resolveModelDir(repo: "nonexistent-owner/definitely-not-a-real-model-xyz"))
    }

    /// Regression: Mage-Flow ships the diffusers layout — `model_index.json` +
    /// weight subdirs, NO root config.json. `resolveModelDir` gated on
    /// config.json alone, so a fully-downloaded Mage-Flow read as "not
    /// downloaded" and every gen service threw `.modelMissing`. It must accept
    /// the diffusers root marker too, while still rejecting a bare empty folder.
    func testResolveModelDirAcceptsDiffusersModelIndexLayout() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "resolvedir-\(UUID().uuidString)"
        defer { try? fm.removeItem(atPath: root) }

        // Diffusers layout (model_index.json, NO config.json) → resolves.
        let mageDir = (root as NSString).appendingPathComponent("microsoft/Mage-Flow-Turbo")
        try fm.createDirectory(atPath: mageDir, withIntermediateDirectories: true)
        fm.createFile(atPath: (mageDir as NSString).appendingPathComponent("model_index.json"), contents: Data("{}".utf8))
        XCTAssertEqual(ServerManager.resolveModelDir(repo: "microsoft/Mage-Flow-Turbo", modelsRoot: root), mageDir)

        // Standard MLX layout (config.json) → still resolves.
        let stdDir = (root as NSString).appendingPathComponent("acme/std")
        try fm.createDirectory(atPath: stdDir, withIntermediateDirectories: true)
        fm.createFile(atPath: (stdDir as NSString).appendingPathComponent("config.json"), contents: Data("{}".utf8))
        XCTAssertEqual(ServerManager.resolveModelDir(repo: "acme/std", modelsRoot: root), stdDir)

        // A bare empty folder (no marker) → nil (not downloaded).
        let emptyDir = (root as NSString).appendingPathComponent("acme/empty")
        try fm.createDirectory(atPath: emptyDir, withIntermediateDirectories: true)
        XCTAssertNil(ServerManager.resolveModelDir(repo: "acme/empty", modelsRoot: root))
    }

    /// FLUX.2-klein 9B (`mlx-community/flux2-klein-9b-4bit`) ships the weight
    /// subdirs and NO root json at all — not `config.json`, not
    /// `model_index.json`. `resolveModelDir` re-checked those two markers after
    /// `existingModelDir` had already answered the same question, so a complete
    /// download threw `.modelMissing` at generate time. One predicate, one
    /// answer: the duplicate marker list is what let the two sites drift.
    func testResolveModelDirAcceptsTheConfiglessWeightSubdirLayout() throws {
        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "resolvedir9b-\(UUID().uuidString)"
        defer { try? fm.removeItem(atPath: root) }

        let dir = (root as NSString).appendingPathComponent("mlx-community/flux2-klein-9b-4bit")
        for sub in ["transformer", "vae", "text_encoder", "tokenizer"] {
            try fm.createDirectory(atPath: (dir as NSString).appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        fm.createFile(atPath: (dir as NSString).appendingPathComponent("transformer/0.safetensors"), contents: Data([0, 1, 2]))
        XCTAssertEqual(ServerManager.resolveModelDir(repo: "mlx-community/flux2-klein-9b-4bit", modelsRoot: root), dir)
    }

    // MARK: - Residency default

    func testKeepResidentDefaultsOff() {
        // Decision: load→generate→unload by default; "Keep loaded" is opt-in.
        let img = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 4)
        XCTAssertFalse(img.keepResident)
        let vid = VideoGenRequest(model: .ltx23Q4, prompt: "x", width: 384, height: 256, numFrames: 9, fps: 24, mode: .oneStage, steps: 6, cfgScale: 1.0)
        XCTAssertFalse(vid.keepResident)
        let aud = AudioGenRequest(model: .qwen3TTS06B, text: "x")
        XCTAssertFalse(aud.keepResident)
    }

    // MARK: - Per-model capabilities (the Advanced panel is gated on these)

    /// Every flag mirrors a server-side fact. A control the backend ignores is
    /// worse than a missing one, and Mage-Flow shipped with five of them: CFG,
    /// negative prompt, conditioning rebalance, LoRA and variation strength all
    /// showed for a model that answers 400 or silently drops them.
    func testMageFlowDeclaresTheCapabilitiesItActuallyHas() {
        for p in [ImageModelPreset.mageFlowTurbo, .mageFlowEditTurbo] {
            XCTAssertFalse(p.supportsImg2Img, "\(p.id): no VAE-encoder variation path (server 400s)")
            XCTAssertFalse(p.supportsLoRA, "\(p.id): no LoRA path (0 matched modules -> 400)")
            XCTAssertEqual(p.condWeightCount, 0, "\(p.id): single final hidden state, no layers to tap")
            XCTAssertTrue(p.stepsAreFixed, "\(p.id): distilled 4-step schedule")
            XCTAssertEqual(p.fixedSteps, 4)
            // Distillation-fixed means every tier is the same; a tier that only
            // buys time is the no-op class in another costume.
            for q in QualityPreset.allCases {
                XCTAssertEqual(p.settings(q).steps, 4, "\(p.id) \(q): tiers must not sell steps")
            }
        }
        // Only the Edit checkpoint edits; only it can do reference edits.
        XCTAssertTrue(ImageModelPreset.mageFlowEditTurbo.supportsReferenceEdit)
        XCTAssertFalse(ImageModelPreset.mageFlowTurbo.supportsReferenceEdit)
        // The backends that DO have these paths keep them.
        for p in [ImageModelPreset.flux2Klein4B_Q4, .krea2Turbo] {
            XCTAssertTrue(p.supportsImg2Img)
            XCTAssertTrue(p.supportsLoRA)
            XCTAssertFalse(p.stepsAreFixed)
            XCTAssertGreaterThan(p.condWeightCount, 0)
        }
    }

    /// The 8-bit mirrors are the SAME architecture at half the download, so they
    /// reuse their bf16 sibling's variant — quantization is a property of the
    /// checkpoint, not of what the model can do. If they ever diverge on a
    /// capability flag, one of the two is lying to the UI.
    func testMageFlow8BitPresetsMatchTheirBf16Siblings() {
        let pairs = [
            (ImageModelPreset.mageFlowTurbo, ImageModelPreset.mageFlowTurbo8bit),
            (ImageModelPreset.mageFlowEditTurbo, ImageModelPreset.mageFlowEditTurbo8bit),
        ]
        for (bf16, q8) in pairs {
            XCTAssertEqual(q8.variant, bf16.variant, "\(q8.id): same architecture, same variant")
            XCTAssertEqual(q8.configName, bf16.configName)
            XCTAssertEqual(q8.supportsReferenceEdit, bf16.supportsReferenceEdit)
            XCTAssertEqual(q8.supportsImg2Img, bf16.supportsImg2Img)
            XCTAssertEqual(q8.supportsLoRA, bf16.supportsLoRA)
            XCTAssertEqual(q8.stepsAreFixed, bf16.stepsAreFixed)
            XCTAssertEqual(q8.fixedSteps, bf16.fixedSteps)
            XCTAssertEqual(q8.resolutions, bf16.resolutions)
            XCTAssertNotEqual(q8.id, bf16.id, "distinct ids or the picker collapses them")
            XCTAssertLessThan(q8.approxDownloadGB, bf16.approxDownloadGB, "\(q8.id): must be the smaller one")
            XCTAssertTrue(q8.repo.hasPrefix("ddalcu/"), "\(q8.id): our mirror, not microsoft's")
        }
        // The engine gates EDIT capability on the directory name containing
        // "mage-flow-edit" (dirIsEdit in src/mage_flow.zig), and the download
        // dir is the repo id — so the edit mirror's name is load-bearing.
        let editDir = ImageModelPreset.mageFlowEditTurbo8bit.repo
            .split(separator: "/").last.map(String.init) ?? ""
        XCTAssertTrue(editDir.lowercased().contains("mage-flow-edit"),
                      "\(editDir) would come up in text-to-image mode")
        XCTAssertFalse(
            (ImageModelPreset.mageFlowTurbo8bit.repo.split(separator: "/").last.map(String.init) ?? "")
                .lowercased().contains("mage-flow-edit"),
            "the txt2img mirror must NOT match dirIsEdit")
        // Both are in the catalog, cheapest-first ordering intact.
        let all = ImageModelPreset.all
        XCTAssertTrue(all.contains { $0.id == ImageModelPreset.mageFlowTurbo8bit.id })
        XCTAssertTrue(all.contains { $0.id == ImageModelPreset.mageFlowEditTurbo8bit.id })
        for (i, p) in all.enumerated() where i > 0 {
            XCTAssertGreaterThanOrEqual(p.approxDownloadGB, all[i - 1].approxDownloadGB,
                                        "catalog must stay ordered cheapest → heaviest")
        }
    }

    func testMageFlowResolutionsCoverTheNativeRange() {
        let r = ImageModelPreset.mageFlowTurbo.resolutions
        // The model card claims native 512-2048 on any aspect, "including
        // extreme 4:1" — and measured cost tracks megapixels, not shape, so
        // there's no reason to hide the panoramas.
        XCTAssertTrue(r.contains { $0.width == 2048 && $0.height == 2048 }, "missing the 2048 ceiling")
        XCTAssertTrue(r.contains { $0.width == 2048 && $0.height == 512 }, "missing the 4:1 panorama")
        XCTAssertTrue(r.contains { $0.width == 512 && $0.height == 2048 }, "missing the 1:4 tall shape")
        for o in r {
            XCTAssertEqual(o.width % 16, 0, "\(o.label): VAE is /16")
            XCTAssertEqual(o.height % 16, 0, "\(o.label): VAE is /16")
            XCTAssertTrue((512...2048).contains(o.width) && (512...2048).contains(o.height),
                          "\(o.label) is outside the model's native 512-2048 range")
        }
    }

    func testMatchSourceIsOfferedOnlyWhenEditingAndSurvivesModeChanges() {
        let edit = ImageModelPreset.mageFlowEditTurbo
        // Editing: "Match source" leads, because an editor that changes the
        // geometry of its input is wrong by default.
        XCTAssertEqual(edit.resolutionOptions(editMode: true).first, .matchSource)
        XCTAssertFalse(edit.resolutionOptions(editMode: false).contains(.matchSource))
        // A txt2img model never offers it (nothing to match).
        XCTAssertFalse(ImageModelPreset.mageFlowTurbo.resolutionOptions(editMode: true).contains(.matchSource))
        // Leaving edit mode with "Match source" selected must re-point the
        // picker instead of leaving it on an off-menu value.
        XCTAssertEqual(edit.validResolution(.matchSource, editMode: false), edit.defaultResolution)
        XCTAssertEqual(edit.validResolution(.matchSource, editMode: true), .matchSource)
        // Switching models drops a resolution the new model doesn't offer.
        let panorama = ResolutionOption(width: 2048, height: 512, label: "x")
        XCTAssertEqual(ImageModelPreset.flux2Klein4B_Q4.validResolution(panorama, editMode: false),
                       ImageModelPreset.flux2Klein4B_Q4.defaultResolution)
    }

    // MARK: - Prompt examples (an editor's repertoire IS its prompt vocabulary)

    func testPromptExamplesMatchTheModeAndTheModel() {
        // No source image → text-to-image starters, whatever the model.
        for p in [ImageModelPreset.mageFlowEditTurbo, .mageFlowTurbo, .flux2Klein4B_Q4, .krea2Turbo] {
            XCTAssertEqual(p.promptExamples(editing: false), ImagePromptExamples.textToImage, "\(p.id)")
        }
        // Editing on Mage-Flow-Edit → the full published repertoire.
        let mage = ImageModelPreset.mageFlowEditTurbo.promptExamples(editing: true)
        XCTAssertEqual(mage, ImagePromptExamples.mageFlowEdit)
        XCTAssertTrue(mage.contains { $0.name == "Control maps" })
        XCTAssertTrue(mage.contains { $0.name == "Restore" })
        // Editing on another in-context editor → generic instructions only. We
        // verified control maps and restoration on Mage-Flow, not on FLUX;
        // offering them there would be advertising, not a feature.
        let flux = ImageModelPreset.flux2Klein4B_Q4.promptExamples(editing: true)
        XCTAssertEqual(flux, ImagePromptExamples.genericEdit)
        XCTAssertFalse(flux.contains { $0.name == "Control maps" })
        // A model that can't edit never shows edit instructions, even if the
        // pane somehow asks for them.
        XCTAssertEqual(ImageModelPreset.mageFlowTurbo.promptExamples(editing: true), ImagePromptExamples.textToImage)
    }

    /// These strings are the model's OWN published phrasings, and the exact ones
    /// verified live against the port (a depth map, canny edges, a
    /// white-background cutout). An editor keys on the wording: shortening
    /// "Generate a grayscale monocular depth map… closer regions brighter and
    /// farther regions darker." to "make a depth map" returns a grey photo, not
    /// a depth map. Rewording them is a quality regression, so pin them.
    func testControlMapPromptsAreTheModelsOwnWordingVerbatim() {
        let maps = ImagePromptExamples.mageFlowEdit.first { $0.name == "Control maps" }!
        func body(_ title: String) -> String { maps.examples.first { $0.title == title }!.body }
        XCTAssertEqual(body("Depth map"),
            "Generate a grayscale monocular depth map of this image. Represent relative distance at pixel level, with closer regions brighter and farther regions darker.")
        XCTAssertEqual(body("Canny edges"),
            "Convert this image into a clean black-and-white Canny edge map showing only the important object and scene contours.")
        XCTAssertEqual(body("Segmentation map"),
            "Generate a semantic segmentation map that assigns clearly different flat colors to the main subject, other foreground objects, and the background regions.")
        let content = ImagePromptExamples.mageFlowEdit.first { $0.name == "Content" }!
        XCTAssertEqual(content.examples.first { $0.title == "Cut out the subject" }!.body,
            "Extract the main foreground subject from the image and isolate it on a clean pure white background. Preserve its shape, identity, texture, and fine boundary details.")
    }

    func testEveryExampleGroupIsUsableInAMenu() {
        for lib in [ImagePromptExamples.textToImage, ImagePromptExamples.mageFlowEdit, ImagePromptExamples.genericEdit] {
            var seenGroups = Set<String>()
            for g in lib {
                XCTAssertFalse(g.examples.isEmpty, "\(g.name): an empty submenu is a dead end")
                XCTAssertTrue(seenGroups.insert(g.name).inserted, "duplicate group \(g.name)")
                var seenTitles = Set<String>()
                for e in g.examples {
                    // SwiftUI keys the menu rows on `title`; a duplicate silently
                    // collapses two entries into one.
                    XCTAssertTrue(seenTitles.insert(e.title).inserted, "\(g.name): duplicate title \(e.title)")
                    XCTAssertFalse(e.title.isEmpty)
                    XCTAssertGreaterThan(e.body.count, 12, "\(e.title): too short to be a useful starting point")
                }
            }
        }
    }

    // MARK: - Image request body (img2img + rebalance + LoRA)

    func testImageRequestJsonOmitsSizeForMatchSource() {
        // width/height 0 = "Match source": the server must not see a `size`, so
        // an edit comes back at the source's own resolution (the reference
        // pipeline's max_size default) instead of being re-gridded to a bucket.
        var req = ImageGenRequest(model: .mageFlowEditTurbo, prompt: "x", width: 0, height: 0, steps: 4)
        req.editMode = true
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 3)
        XCTAssertNil(json["size"], "match-source must omit size entirely")
        XCTAssertEqual(json["steps"] as? Int, 4)
        // A real size is still sent verbatim.
        let sized = ImageGenRequest(model: .mageFlowEditTurbo, prompt: "x", width: 2048, height: 512, steps: 4)
        XCTAssertEqual(ImageGenService.requestJson(for: sized, modelName: "m", seed: 3)["size"] as? String, "2048x512")
    }


    func testImageRequestJsonDefaultsOmitImg2ImgRebalanceAndLora() {
        let req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 4)
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 7)
        XCTAssertEqual(json["prompt"] as? String, "x")
        XCTAssertEqual(json["seed"] as? Int, 7)
        // No behavior change for plain text-to-image requests.
        XCTAssertNil(json["image"])
        XCTAssertNil(json["strength"])
        XCTAssertNil(json["cond_gain"])
        XCTAssertNil(json["cond_weights"])
        XCTAssertNil(json["lora_paths"])
    }

    func testImageRequestJsonIncludesImg2ImgFields() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("img2img-src-\(UUID().uuidString).png")
        let bytes = Data([0x89, 0x50, 0x4E, 0x47, 1, 2, 3, 4])
        try bytes.write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        var req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 8)
        req.initImagePath = tmp.path
        req.strength = 0.45
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertEqual(json["image"] as? String, bytes.base64EncodedString())
        XCTAssertEqual(json["strength"] as? Double, 0.45)
    }

    func testImageRequestJsonMissingSourceFileDropsImg2Img() {
        var req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 8)
        req.initImagePath = "/definitely/not/a/real/file.png"
        req.strength = 0.4
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertNil(json["image"])
        XCTAssertNil(json["strength"])
    }

    func testImageRequestJsonIncludesRebalanceAndLora() {
        var req = ImageGenRequest(model: .krea2Turbo, prompt: "x", width: 1024, height: 1024, steps: 8)
        req.condGain = 1.5
        req.condWeightsText = "1, 1 1 1 1 1 0.5 1 1 1 1 2"
        req.loras = [
            LoraAdapter(path: "/tmp/style.safetensors", scale: 1.0),
            LoraAdapter(path: "/tmp/character.safetensors", scale: 0.6),
        ]
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertEqual(json["cond_gain"] as? Double, 1.5)
        let w = json["cond_weights"] as? [Double]
        XCTAssertEqual(w?.count, 12)
        XCTAssertEqual(w?[6], 0.5)
        XCTAssertEqual(w?[11], 2)
        XCTAssertEqual(json["lora_paths"] as? [String], ["/tmp/style.safetensors", "/tmp/character.safetensors"])
        XCTAssertEqual(json["lora_scales"] as? [Double], [1.0, 0.6])
    }

    func testImageRequestJsonDropsHalfFilledLoraRows() {
        // A row added by tapping "+" but never given a path must never reach
        // the server as an empty string — it's silently dropped.
        var req = ImageGenRequest(model: .krea2Turbo, prompt: "x", width: 1024, height: 1024, steps: 8)
        req.loras = [LoraAdapter(path: "", scale: 1.0), LoraAdapter(path: "/tmp/ok.safetensors", scale: 0.9)]
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertEqual(json["lora_paths"] as? [String], ["/tmp/ok.safetensors"])
        XCTAssertEqual(json["lora_scales"] as? [Double], [0.9])
    }

    func testParseCondWeightsAcceptsCommasAndSpacesRejectsGarbage() {
        XCTAssertEqual(ImageGenRequest.parseCondWeights("1,2,3"), [1, 2, 3])
        XCTAssertEqual(ImageGenRequest.parseCondWeights(" 0.5  1\t-2 "), [0.5, 1, -2])
        XCTAssertEqual(ImageGenRequest.parseCondWeights("1, 2,, 3"), [1, 2, 3])
        XCTAssertNil(ImageGenRequest.parseCondWeights("1,x,3"))
        XCTAssertNil(ImageGenRequest.parseCondWeights(""))
        XCTAssertNil(ImageGenRequest.parseCondWeights("  "))
    }

    func testCondWeightCountFollowsBackend() {
        // FLUX taps encoder layers 9/18/27 → 3 weights; Krea taps 12 layers.
        XCTAssertEqual(ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 4).condWeightCount, 3)
        XCTAssertEqual(ImageGenRequest(model: .krea2Turbo, prompt: "x", width: 1024, height: 1024, steps: 8).condWeightCount, 12)
    }

    // MARK: - Instruction edit mode (FLUX.2 in-context reference conditioning)

    func testImageRequestJsonEditModeSendsModeAndOmitsStrength() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("edit-src-\(UUID().uuidString).png")
        try Data([1, 2, 3]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        var req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "make the hair blue", width: 1024, height: 1024, steps: 8)
        req.initImagePath = tmp.path
        req.editMode = true
        req.strength = 0.4
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertEqual(json["mode"] as? String, "edit")
        XCTAssertNotNil(json["image"])
        // Edit conditions on the clean reference — strength does not apply.
        XCTAssertNil(json["strength"])
    }

    func testImageRequestJsonVariationModeOmitsModeField() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("var-src-\(UUID().uuidString).png")
        try Data([1, 2, 3]).write(to: tmp)
        defer { try? FileManager.default.removeItem(at: tmp) }

        var req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 8)
        req.initImagePath = tmp.path
        req.editMode = false
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertNil(json["mode"]) // default server behavior = variation
        XCTAssertNotNil(json["strength"])
    }

    func testImageRequestJsonEditModeIncludesRefImages() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let src = tmpDir.appendingPathComponent("edit-src-\(UUID().uuidString).png")
        try Data([1, 2, 3]).write(to: src)
        defer { try? FileManager.default.removeItem(at: src) }
        let refBytes = Data([9, 8, 7, 6])
        let ref = tmpDir.appendingPathComponent("edit-ref-\(UUID().uuidString).png")
        try refBytes.write(to: ref)
        defer { try? FileManager.default.removeItem(at: ref) }

        var req = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "replace the face in image 1 with the face from image 2", width: 1024, height: 1024, steps: 8)
        req.initImagePath = src.path
        req.editMode = true
        req.refImagePaths = [ref.path, "/definitely/not/a/real/ref.png"] // missing file skipped
        let json = ImageGenService.requestJson(for: req, modelName: "m", seed: 1)
        XCTAssertEqual(json["mode"] as? String, "edit")
        XCTAssertEqual(json["ref_images"] as? [String], [refBytes.base64EncodedString()])
    }

    func testImageRequestJsonRefImagesOmittedInVariationAndTextToImage() throws {
        let tmpDir = FileManager.default.temporaryDirectory
        let ref = tmpDir.appendingPathComponent("var-ref-\(UUID().uuidString).png")
        try Data([1]).write(to: ref)
        defer { try? FileManager.default.removeItem(at: ref) }

        // Variation mode: extra references have no meaning (the server 400s).
        let src = tmpDir.appendingPathComponent("var-src-\(UUID().uuidString).png")
        try Data([2]).write(to: src)
        defer { try? FileManager.default.removeItem(at: src) }
        var vreq = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 8)
        vreq.initImagePath = src.path
        vreq.editMode = false
        vreq.refImagePaths = [ref.path]
        XCTAssertNil(ImageGenService.requestJson(for: vreq, modelName: "m", seed: 1)["ref_images"])

        // Text-to-image (no source at all): refs are meaningless too.
        var treq = ImageGenRequest(model: .flux2Klein4B_Q4, prompt: "x", width: 1024, height: 1024, steps: 8)
        treq.editMode = true
        treq.refImagePaths = [ref.path]
        XCTAssertNil(ImageGenService.requestJson(for: treq, modelName: "m", seed: 1)["ref_images"])
    }

    func testSupportsReferenceEditFollowsVariant() {
        // Editing is a trained FLUX.2 capability; Krea doesn't have it.
        XCTAssertTrue(ImageModelPreset.flux2Klein4B_Q4.supportsReferenceEdit)
        XCTAssertFalse(ImageModelPreset.krea2Turbo.supportsReferenceEdit)
    }
}

// MARK: - MiniMax-H3 declared capabilities

extension MediaGenServiceTests {

    /// H3 is not LTX-shaped, and the pane gates on the PRESET rather than on a
    /// model id. Mage-Flow shipped five dead image controls before its preset
    /// started declaring capabilities; this pins the video equivalent.
    func testMiniMaxH3DeclaresTheCapabilitiesItActuallyHas() {
        let h3 = VideoModelPreset.minimaxH3

        // No guidance pass (CFG-distilled), no pipeline modes. The server
        // answers a NAMED 400 for each, so offering them would be a control
        // that can only fail.
        XCTAssertFalse(h3.supportsCFG)
        XCTAssertFalse(h3.supportsPipelineModes)
        // Adapters it DOES take: the server resolves `lora_paths` against H3's
        // own module names and sums them with the Turbo distillation.
        XCTAssertTrue(h3.supportsLoRA)
        XCTAssertTrue(VideoModelPreset.minimaxH3Ref2VA.supportsLoRA)

        // It writes its own soundtrack jointly with the frames, so there is no
        // audio INPUT to condition on.
        XCTAssertFalse(h3.supportsAudioInput)
        XCTAssertTrue(h3.generatesAudio)

        // LTX keeps every capability — the defaults must not have moved.
        let ltx = VideoModelPreset.ltx23Q4
        XCTAssertTrue(ltx.supportsLoRA)
        XCTAssertTrue(ltx.supportsCFG)
        XCTAssertTrue(ltx.supportsPipelineModes)
        XCTAssertTrue(ltx.supportsAudioInput)
    }

    /// H3's ladder is 17k+5, LTX's is 8N+1. Offering a count off the ladder
    /// means the server snaps it and the clip is a different length than the
    /// one requested — which silently desynchronizes an external audio mux.
    func testMiniMaxH3FrameLadderIs17kPlus5() {
        let h3 = VideoModelPreset.minimaxH3
        XCTAssertFalse(h3.frameOptions.isEmpty)
        for n in h3.frameOptions {
            XCTAssertEqual(n % 17, 5, "\(n) is not on the 17k+5 ladder")
            XCTAssertLessThanOrEqual(n, h3.maxFrames)
        }
        // Floor is the ENGINE's 5, not the reference node's trained-range
        // start of 124: off-distribution is a reason to warn (see
        // `H3ReachableRangeTests`), not a reason to make a one-second test
        // clip — or the server's own default length of 56 — unreachable.
        XCTAssertEqual(h3.frameOptions.first, 5)
        XCTAssertTrue(h3.frameOptions.contains(56))
        XCTAssertTrue(h3.frameOptions.contains(124))
        // Ceiling is the top of the model's own ladder: MiniMax states 4-15 s
        // at 24 fps, and 362 frames (15.1 s) is a run we have done in ONE
        // generation. It used to stop at 209 — the rap demo's length, i.e. the
        // longest clip we had shipped a verdict on — behind a comment calling
        // 362 "untested-by-us" that was already out of date. What actually
        // bounds length is memory and time, and both are modelled now
        // (`H3PlanningTests`), so the picker offers the range and says the cost.
        XCTAssertEqual(h3.frameOptions.last, 362)
        XCTAssertTrue(h3.frameOptions.contains(209))
        XCTAssertEqual(h3.maxFrames, 362)
        // No quality TIER may pick the top of the ladder: a preset is a
        // default, and 362 frames at 1344x768 is a five-hour job that nobody
        // should start by choosing "Quality". The slider reaches it; the tiers
        // stop at the length we have shipped a verdict on.
        for q in QualityPreset.allCases {
            XCTAssertLessThanOrEqual(h3.settings(q).numFrames, 209,
                                     "\(q) defaults past the validated length")
        }
        // 15.1 s at 24 fps — the top of MiniMax's stated 4-15 s range.
        XCTAssertEqual(Double(h3.frameOptions.last!) / Double(h3.fps), 15.08, accuracy: 0.05)

        // Every quality preset's frame count must land ON the ladder, or the
        // Frames picker renders blank for that tier.
        for q in QualityPreset.allCases {
            let n = h3.settings(q).numFrames
            XCTAssertTrue(h3.frameOptions.contains(n),
                          "\(q) frame count \(n) is off the ladder")
        }

        // LTX's ladder is a DIFFERENT rule and must not have been changed.
        for n in VideoModelPreset.ltx23Q4.frameOptions {
            XCTAssertEqual(n % 8, 1, "\(n) is not on LTX's 8N+1 ladder")
        }
    }

    /// Every H3 canvas must be a multiple of 32: the DiT patchifies 2x2 over a
    /// 16x-compressed latent, so an off-grid size cannot be expressed.
    func testMiniMaxH3ResolutionsAreOn32PixelGrid() {
        for r in VideoModelPreset.minimaxH3.resolutions {
            XCTAssertEqual(r.width % 32, 0, "width \(r.width) is off the 32 grid")
            XCTAssertEqual(r.height % 32, 0, "height \(r.height) is off the 32 grid")
        }
    }

    /// 960x544 is the long-form canvas, and a relative speed label is a claim
    /// about pixel count that has to stay true when the list grows.
    ///
    /// Field measurement on an M5 Max (2026-08-04, our engine): 362 frames —
    /// the top of the 17k+5 ladder — in ONE generation at 139.6 s/step, and
    /// 2.9x faster than 1344x768 at a matched 124 frames while costing 1.98x
    /// fewer pixels. Adding it below 768x768 makes the old "fastest" label on
    /// that entry false, which is exactly the drift this guard catches.
    func testMiniMaxH3ResolutionLabelsMatchTheirPixelCost() {
        let rs = VideoModelPreset.minimaxH3.resolutions
        XCTAssertTrue(rs.contains { $0.id == "960x544" }, "the long-form canvas is missing")
        // The claim is about PIXEL COST, so the comparison is too — a rotation
        // costs exactly the same, and the 9:16 portrait added in #177 ties the
        // 16:9 landscape at 522,240. Comparing ids instead made that tie read
        // as drift ("544x960 claims fastest but 960x544 has fewer pixels")
        // while BOTH labels were true. The guard still catches the real thing:
        // a genuinely cheaper canvas added below one still labelled fastest.
        let cheapestPixels = rs.map { $0.width * $0.height }.min()!
        XCTAssertEqual(cheapestPixels, 960 * 544, "the long-form canvas is no longer the floor")
        for r in rs where r.label.lowercased().contains("fastest") {
            XCTAssertEqual(r.width * r.height, cheapestPixels,
                           "\(r.id) claims fastest but a cheaper canvas exists")
        }
        // Every canvas stays inside the model's own area cap (768*1344): past
        // it the server's own normalize would scale the request down.
        for r in rs {
            XCTAssertLessThanOrEqual(r.width * r.height, 768 * 1344, "\(r.id) exceeds the area cap")
        }
    }
}

extension MediaGenServiceTests {

    /// Every video preset's backend must be recognized as a MEDIA type, or the
    /// browser and picker treat it as a chat model. This is the client half of
    /// the same duplication that made `/v1/load-model` answer 400 "unsupported
    /// model_type" for MiniMax-H3: the server-side dispatcher knew the type and
    /// discovery did not.
    func testEveryVideoBackendIsRecognizedAsAMediaType() {
        XCTAssertTrue(isMediaModelType("minimax_h3"))
        XCTAssertTrue(isMediaModelType("AudioVideo"))   // LTX
        // A chat arch must NOT be, or it would be routed to a media engine.
        for t in ["gemma4", "qwen3", "llama", "deepseek_v4"] {
            XCTAssertFalse(isMediaModelType(t), "\(t) must not read as media")
        }
    }
}

extension MediaGenServiceTests {

    /// The request must carry only what the backend declares it can honor.
    /// REGRESSION: `pipeline` was added to the body unconditionally, so every
    /// MiniMax-H3 generation shipped a field that backend has no concept of and
    /// the server refused it — the pane's controls were hidden, but the request
    /// builder had not been told.
    func testRequestOmitsFieldsTheBackendDoesNotSupport() {
        let h3 = VideoGenRequest(
            model: .minimaxH3, prompt: "a cat", width: 256, height: 256,
            numFrames: 56, fps: 24, mode: .oneStage, steps: 30,
            cfgScale: 1.0, loras: [LoraAdapter(path: "/tmp/some.safetensors")])
        // audioB64 present: a stale in-memory clip from an earlier LTX pick
        // must never reach a backend that generates its own soundtrack.
        let body = VideoGenService.requestBody(
            model: "m", prompt: "a cat", request: h3,
            firstFrameB64: nil, audioB64: "QUJD")

        XCTAssertNil(body["pipeline"], "H3 has no pipeline modes")
        XCTAssertNil(body["cfg_scale"], "H3 is CFG-distilled")
        XCTAssertNil(body["stg_scale"])
        XCTAssertNil(body["audio"], "H3 takes no audio input")
        // Adapters DO travel — H3 resolves them against its own module names.
        XCTAssertEqual(body["lora_paths"] as? [String], ["/tmp/some.safetensors"])
        // The fields every backend needs must still be there. `preview` is NOT
        // one of them — it is the pane's own opt-in (see
        // testLivePreviewIsOptInAndCarriesItsShapeWhenAsked).
        for k in ["model", "prompt", "num_frames", "height", "width", "steps", "seed"] {
            XCTAssertNotNil(body[k], "\(k) must always be sent")
        }

        // LTX keeps all of them — the gating must not have narrowed it.
        let ltx = VideoGenRequest(
            model: .ltx23Q4, prompt: "a cat", width: 704, height: 480,
            numFrames: 97, fps: 24, mode: .oneStage, steps: 12,
            cfgScale: 3.0, loras: [LoraAdapter(path: "/tmp/some.safetensors")])
        let lbody = VideoGenService.requestBody(
            model: "m", prompt: "a cat", request: ltx,
            firstFrameB64: nil, audioB64: nil)
        XCTAssertNotNil(lbody["pipeline"])
        XCTAssertNotNil(lbody["cfg_scale"])
        XCTAssertEqual(lbody["lora_paths"] as? [String], ["/tmp/some.safetensors"])
        XCTAssertEqual(lbody["lora_scales"] as? [Double], [1.0])
    }
}
