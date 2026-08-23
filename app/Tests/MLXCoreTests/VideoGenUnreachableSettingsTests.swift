import XCTest
@testable import MLXCore

/// Five server settings the video pane could not reach (#243). Each is a
/// finished server feature whose control was missing, so the failure mode is
/// silent: the request simply never carries the field and the server applies
/// its default. Two of them (`stg_scale`, `chain_windows`) were already IN the
/// request and already sent — they just had no control, which is the same bug
/// wearing a different hat, so those two are pinned by a source scan.
final class VideoGenUnreachableSettingsTests: XCTestCase {

    private func videoGenViewSource() throws -> String {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // MLXCoreTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // app root
            .appendingPathComponent("Sources/MLXServe/Views/VideoGenView.swift")
        return try String(contentsOf: url, encoding: .utf8)
    }

    private func req(_ model: VideoModelPreset, mode: VideoPipelineMode = .oneStage) -> VideoGenRequest {
        VideoGenRequest(model: model, prompt: "p", width: 384, height: 256,
                        numFrames: 124, fps: 24, mode: mode, steps: 16, cfgScale: 1.0)
    }

    // MARK: - H3 last frame

    func testLastFrameImageReachesTheRequestOnlyWhereDeclared() {
        // fl2va is first-LAST frame to video+audio. The server pins the last
        // keyframe as a center-COVER anchor; without the field we ship half
        // the partition the pack is named for. ref2va has no keyframe row to
        // anchor, so it must stay absent even when the state is populated (a
        // preset switch leaves the picked file behind — the `pipeline`-on-H3
        // class). LTX pins the last latent frame on both pipelines (#260).
        XCTAssertTrue(VideoModelPreset.minimaxH3.supportsLastFrame)
        XCTAssertTrue(VideoModelPreset.minimaxH3Q4.supportsLastFrame)
        XCTAssertFalse(VideoModelPreset.minimaxH3Ref2VA.supportsLastFrame)
        XCTAssertTrue(VideoModelPreset.ltx25Q8.supportsLastFrame)
        XCTAssertTrue(VideoModelPreset.ltx25Q4.supportsLastFrame)
        XCTAssertTrue(VideoModelPreset.ltx23Q4.supportsLastFrame)

        for model in [VideoModelPreset.minimaxH3, .minimaxH3Ref2VA, .ltx25Q8] {
            let body = VideoGenService.requestBody(model: "m", prompt: "p", request: req(model),
                                                   firstFrameB64: nil, lastFrameB64: "TA==")
            if model.supportsLastFrame {
                XCTAssertEqual(body["last_frame_image"] as? String, "TA==",
                               "\(model.id) declares last-frame support but drops the field")
            } else {
                XCTAssertNil(body["last_frame_image"],
                             "\(model.id) has no last-frame anchor — the field must not be sent")
            }
        }
        // Absent when no image is attached, like first_frame_image.
        let bare = VideoGenService.requestBody(model: "m", prompt: "p", request: req(.minimaxH3),
                                               firstFrameB64: nil, lastFrameB64: nil)
        XCTAssertNil(bare["last_frame_image"])
    }

    func testLastFrameHasAControlBesideFirstFrame() throws {
        let source = try videoGenViewSource()
        XCTAssertTrue(source.contains("lastFrameImageURL"),
                      "The video pane needs a last-frame well; the server anchor is unreachable without one")
        XCTAssertTrue(source.contains("lastFrameImagePath: "),
                      "The picked last frame must reach VideoGenRequest, or the well is decorative")
    }

    // MARK: - H3 chained windows

    func testChainedWindowsHasAControl() throws {
        // Already in the request and already sent by the service — it was the
        // CONTROL that never existed, so `chainWindows` sat at 1 forever.
        let source = try videoGenViewSource()
        XCTAssertTrue(source.contains("chainWindows"),
                      "chain_windows is wired end to end but no view sets it — long clips are unreachable")
    }

    func testChainedWindowsStaysGatedOnThePartition() {
        var fl2va = req(.minimaxH3)
        fl2va.chainWindows = 3
        XCTAssertEqual(VideoGenService.requestBody(model: "m", prompt: "p", request: fl2va,
                                                   firstFrameB64: nil)["chain_windows"] as? Int, 3)
        var ref = req(.minimaxH3Ref2VA)
        ref.chainWindows = 3
        XCTAssertNil(VideoGenService.requestBody(model: "m", prompt: "p", request: ref,
                                                 firstFrameB64: nil)["chain_windows"])
    }

    // MARK: - LTX spatio-temporal guidance

    func testStgScaleHasASlider() throws {
        // Declared, persisted and SENT with no way to set it — the worst shape
        // of this bug, because the wire looks correct.
        let source = try videoGenViewSource()
        XCTAssertTrue(source.contains("\"STG scale\""),
                      "stg_scale is sent on every LTX request but has no slider — it is pinned at whatever is in storage")
    }

    // MARK: - LTX stage-2 refine steps

    func testStage2StepsIsSentOnlyWhenSetAndOnlyTwoStage() {
        // 0 means "all 3" server-side, so 0 is Auto and stays absent — the
        // server's default is the absent-field behavior everywhere else in
        // this body. One-stage has no refine pass to size.
        var auto = req(.ltx25Q8, mode: .twoStage)
        auto.stage2Steps = 0
        XCTAssertNil(VideoGenService.requestBody(model: "m", prompt: "p", request: auto,
                                                 firstFrameB64: nil)["stage2_steps"])
        var set = req(.ltx25Q8, mode: .twoStage)
        set.stage2Steps = 2
        XCTAssertEqual(VideoGenService.requestBody(model: "m", prompt: "p", request: set,
                                                   firstFrameB64: nil)["stage2_steps"] as? Int, 2)
        var oneStage = req(.ltx25Q8, mode: .oneStage)
        oneStage.stage2Steps = 2
        XCTAssertNil(VideoGenService.requestBody(model: "m", prompt: "p", request: oneStage,
                                                 firstFrameB64: nil)["stage2_steps"],
                     "one-stage has no refine pass — the field would be a no-op the server still parses")
        // H3 has no pipeline modes at all.
        var h3 = req(.minimaxH3)
        h3.stage2Steps = 2
        XCTAssertNil(VideoGenService.requestBody(model: "m", prompt: "p", request: h3,
                                                 firstFrameB64: nil)["stage2_steps"])
    }

    func testStage2StepsHasAControl() throws {
        let source = try videoGenViewSource()
        XCTAssertTrue(source.contains("stage2Steps"),
                      "The two-stage refine pass has no step control")
    }

    // MARK: - LTX audio guidance

    func testAudioGuidanceIsSentOnlyWithAnAttachedClip() {
        // cfg_audio_scale scales the AUDIO guider, which only exists on the
        // a2vid path. Sending it without a clip would set a knob on a guider
        // that never runs.
        var withAudio = req(.ltx25Q8, mode: .twoStage)
        withAudio.cfgAudioScale = 5.0
        XCTAssertEqual(VideoGenService.requestBody(model: "m", prompt: "p", request: withAudio,
                                                   firstFrameB64: nil,
                                                   audioB64: "UklGRg==")["cfg_audio_scale"] as? Double, 5.0)
        XCTAssertNil(VideoGenService.requestBody(model: "m", prompt: "p", request: withAudio,
                                                 firstFrameB64: nil)["cfg_audio_scale"],
                     "no clip means no audio guider to scale")
        // A one-stage preset with a clip is upgraded to two_stage and DROPS its
        // guidance so the server's reference defaults apply — the audio scale
        // is guidance and must drop with the rest.
        var upgraded = req(.ltx25Q8, mode: .oneStage)
        upgraded.cfgAudioScale = 5.0
        let ubody = VideoGenService.requestBody(model: "m", prompt: "p", request: upgraded,
                                                firstFrameB64: nil, audioB64: "UklGRg==")
        XCTAssertEqual(ubody["pipeline"] as? String, "two_stage")
        XCTAssertNil(ubody["cfg_audio_scale"])
        XCTAssertNil(ubody["cfg_scale"])
    }

    func testAudioGuidanceHasAControl() throws {
        let source = try videoGenViewSource()
        XCTAssertTrue(source.contains("cfgAudioScale"),
                      "Audio-to-video always runs the default audio guidance — the scale has no control")
    }
}
