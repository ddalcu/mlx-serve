import XCTest
@testable import MLXCore

/// The one-tap video templates: what each sends to FastH3 Omni, and the text
/// rules around the lines a chat model writes into it.
final class MediaTemplateTests: XCTestCase {

    private func member(_ name: String, _ kind: CastKind, looks: String = "") -> CastMember {
        CastMember(name: name, kind: kind, voice: CastVoice.default(for: kind), looks: looks, photo: "\(name).jpg")
    }

    /// A fill with every slot answered: the template's own defaults plus a cast member per cast slot.
    private func filled(_ t: MediaTemplate, as kind: CastKind = .girl) -> TemplateFill {
        var fill = t.defaultFill()
        for slot in t.slots where slot.kind == .cast {
            fill.cast[slot.key] = member(slot.key.capitalized, kind)
        }
        return fill
    }

    private func lines(_ t: MediaTemplate, _ fill: TemplateFill) -> [String: String] {
        Dictionary(uniqueKeysWithValues: t.lineKeys(fill).map { ($0, "Line for \($0) here.") })
    }

    func testEveryDocumentCarriesTheSixReferenceSectionsInOrder() {
        for t in MediaTemplate.allCases where t.kind == .video {
            let doc = t.document(filled(t), lines: lines(t, filled(t)))
            var cursor = doc.startIndex
            for section in H3PromptExamples.referenceSections {
                guard let r = doc.range(of: section, range: cursor..<doc.endIndex) else {
                    return XCTFail("\(t.id): \(section) missing or out of order")
                }
                cursor = r.upperBound
            }
        }
    }

    func testEveryCastPhotoIsAPictureTheDocumentNamesInOrder() {
        for t in MediaTemplate.allCases where t.kind == .video {
            let fill = filled(t)
            let cast = t.cast(fill)
            XCTAssertFalse(cast.isEmpty, "\(t.id) renders from at least one photo")
            let doc = t.document(fill, lines: lines(t, fill))
            for i in 1...cast.count {
                XCTAssertTrue(doc.contains("<Picture \(i)>"), "\(t.id) never names <Picture \(i)>")
                XCTAssertTrue(doc.contains("<Subject \(i)> (appears in [Shot 1]"), "\(t.id) states no role for <Subject \(i)>")
            }
            XCTAssertFalse(doc.contains("<Picture \(cast.count + 1)>"), "\(t.id) names a picture it does not send")
        }
    }

    func testEveryLineIsSpokenInsideADialogueTag() {
        for t in MediaTemplate.allCases where t.kind == .video {
            let fill = filled(t)
            let doc = t.document(fill, lines: lines(t, fill))
            for key in t.lineKeys(fill) {
                XCTAssertTrue(doc.contains("<d>[English] Line for \(key) here.</d>"), "\(t.id) drops line \(key)")
            }
        }
    }

    func testPetTalkWithOnePetHasOneHostAndOneLine() {
        var fill = filled(.pets)
        fill.cast["b"] = nil
        XCTAssertEqual(MediaTemplate.pets.lineKeys(fill), ["a"])
        XCTAssertEqual(MediaTemplate.pets.cast(fill).count, 1)
        let doc = MediaTemplate.pets.document(fill, lines: ["a": "Hello there."])
        XCTAssertFalse(doc.contains("<Subject 2>"))
        XCTAssertNil(MediaTemplate.pets.missing(fill), "the second pet is optional")
    }

    func testARequiredPhotoOrBlankAnswerIsNamedBeforeAnythingRuns() {
        var fill = MediaTemplate.santa.defaultFill()
        XCTAssertEqual(MediaTemplate.santa.missing(fill), "Who is it for?")
        fill.cast["child"] = member("Leo", .boy)
        fill.answers["wish"] = "   "
        XCTAssertEqual(MediaTemplate.santa.missing(fill), "What are they hoping for?")
    }

    func testTheUsersWordsCannotBreakTheDocument() {
        var fill = filled(.hug)
        fill.answers["whisper"] = "I <d>love</d> you\n\"always\""
        let doc = MediaTemplate.hug.document(fill, lines: MediaTemplate.hug.ownLines(fill))
        XCTAssertTrue(doc.contains("<d>[English] I dlove/d you always</d>"))
        XCTAssertEqual(doc.components(separatedBy: "<d>").count, 2, "exactly one dialogue tag")
    }

    func testTheChildsNameReachesSantasBriefAndHisStarterLine() {
        var fill = filled(.santa)
        fill.cast["child"] = member("Leo", .boy)
        XCTAssertTrue(MediaTemplate.santa.ask(fill).contains("Leo"))
        XCTAssertTrue(MediaTemplate.santa.fallbackLines(fill)["santa"]?.contains("Leo") == true)
        XCTAssertTrue(MediaTemplate.santa.document(fill, lines: ["santa": "Hi Leo."]).contains("the boy in <Picture 1>"))
    }

    func testLinesParseFromAReplyWithChatterAroundTheObject() {
        let reply = "Sure!\n```json\n{\"a\": \"It smelled like jerky.\", \"b\": \"You have the brain of a sponge.\"}\n```"
        XCTAssertEqual(TemplateText.parseLines(reply, keys: ["a", "b"]),
                       ["a": "It smelled like jerky.", "b": "You have the brain of a sponge."])
        XCTAssertNil(TemplateText.parseLines(reply, keys: ["a", "c"]), "a missing key is no answer")
        XCTAssertNil(TemplateText.parseLines("{\"a\": \"  \"}", keys: ["a"]), "a blank line is no answer")
        XCTAssertNil(TemplateText.parseLines("no json here", keys: ["a"]))
    }

    func testTenSecondsIsTheFrameCountOmniWasTrainedOn() {
        XCTAssertEqual(TemplateText.frames(seconds: 10), 243)
        XCTAssertEqual(TemplateText.frames(seconds: 5), 124)
        for s in 1...15 { XCTAssertEqual((TemplateText.frames(seconds: s) - 5) % 17, 0) }
    }

    func testTheWordBudgetFollowsTheClipLength() {
        XCTAssertEqual(TemplateText.wordBudget(seconds: 10), 26)
        XCTAssertEqual(TemplateText.words(in: ["a": "one two", "b": " three "]), 3)
    }

    func testEveryCanvasIsOnTheThirtyTwoPixelGrid() {
        for shape in ShareShape.allCases {
            XCTAssertEqual(shape.width % 32, 0)
            XCTAssertEqual(shape.height % 32, 0)
        }
        XCTAssertLessThan(ShareShape.story.width, ShareShape.story.height)
    }

    func testAMacThatCannotHoldTheModelIsNeverOfferedTheDownload() {
        let gb: UInt64 = 1 << 30
        XCTAssertEqual(TemplateReadiness.of(onDisk: false, neededBytes: 60 * gb, usableBytes: 40 * gb, seconds: 240),
                       .tooBig(neededGB: 60))
        XCTAssertEqual(TemplateReadiness.of(onDisk: false, neededBytes: 60 * gb, usableBytes: 200 * gb, seconds: 240),
                       .needsDownload)
        XCTAssertEqual(TemplateReadiness.of(onDisk: true, neededBytes: 60 * gb, usableBytes: 200 * gb, seconds: 240),
                       .ready(seconds: 240))
    }

    /// A Mac on the network with the pack makes the video when this one cannot; this Mac wins when it can.
    func testAnotherMacWithTheModelMakesItWhenThisOneCannot() {
        let gb: UInt64 = 1 << 30
        let peer = "FastH3-Omni-MLX-Serve-8bit@studio"
        XCTAssertEqual(TemplateReadiness.of(onDisk: false, neededBytes: 60 * gb, usableBytes: 40 * gb, seconds: 240, peer: peer),
                       .onPeer(peer))
        XCTAssertEqual(TemplateReadiness.of(onDisk: false, neededBytes: 60 * gb, usableBytes: 200 * gb, seconds: 240, peer: peer),
                       .onPeer(peer))
        XCTAssertEqual(TemplateReadiness.of(onDisk: true, neededBytes: 60 * gb, usableBytes: 200 * gb, seconds: 240, peer: peer),
                       .ready(seconds: 240))
        func lan(_ name: String) -> ModelInfo {
            ModelInfo(name: name, quantBits: 0, layers: 0, hiddenSize: 0, vocabSize: 0,
                      contextLength: 0, modelMaxTokens: 0, capabilities: ["video"], lanPeer: "studio")
        }
        XCTAssertEqual(TemplateReadiness.peer(for: .pets, in: [lan("Other-Video@studio"), lan(peer)]),
                       peer, "only a peer serving the template's own pack")
        XCTAssertNil(TemplateReadiness.peer(for: .pets, in: []))
    }

    @MainActor
    func testARenderOnAnotherMacCarriesItsRoute() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tpl-route-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let cast = CastStore(rootDir: dir)
        let fill = filled(.pets)
        XCTAssertNil(TemplateRunner.request(.pets, fill: fill, lines: [:], shape: .story, seed: 1, cast: cast, peer: nil).lanModelId)
        XCTAssertEqual(TemplateRunner.request(.pets, fill: fill, lines: [:], shape: .story, seed: 1, cast: cast, peer: "m@studio").lanModelId,
                       "m@studio")
    }

    /// Captions need the speech model and the aligner that times its words, never the diarizer.
    func testCaptionsNameTheSpeechModelsTheyStillNeed() {
        func pack(_ type: String) -> LocalModel {
            LocalModel(id: type, name: type, path: "/m/\(type)", sizeFormatted: "1 GB", modelType: type, source: .mlxServe, kind: .base)
        }
        let asr = pack("qwen3_asr"), aligner = pack("qwen3_forced_aligner")
        XCTAssertEqual(CaptionModels.of([], engine: .combo), .download([TrayTranscript.asrRepo]))
        XCTAssertEqual(CaptionModels.of([asr], engine: .combo), .download([DownloadManager.alignerRepo]))
        XCTAssertEqual(CaptionModels.of([asr, aligner], engine: .combo), .ready)
        XCTAssertEqual(CaptionModels.of([asr, aligner], engine: .nemotron), .notConverted)
        XCTAssertEqual(CaptionModels.of([pack("nemo_speechlm")], engine: .nemotron), .ready)
    }

    func testAReactionIsAFiveSecondSquareMemeWithItsTextOnTopAndNoCaptions() {
        let t = MediaTemplate.reaction
        XCTAssertEqual(t.seconds, 5)
        XCTAssertEqual(TemplateText.frames(seconds: t.seconds), 124)
        XCTAssertEqual(t.defaultShape, .square)
        XCTAssertFalse(t.captions, "a one-word reaction gets meme text, not captions")
        XCTAssertTrue(t.makesGIF)
        XCTAssertNil(t.defaultFill().answers["top"], "optional meme text starts empty, not on its example")
    }

    func testEachReactionSaysItsOneLineOrNothing() {
        var fill = filled(.reaction, as: .dog)
        for (reaction, keys) in [(Reaction.mindBlown, ["line"]), (.eyeRoll, []), (.sayLess, ["line"])] {
            fill.answers["reaction"] = reaction.label
            XCTAssertEqual(MediaTemplate.reaction.lineKeys(fill), keys, reaction.label)
            XCTAssertEqual(Array(MediaTemplate.reaction.ownLines(fill).keys), keys)
        }
        fill.answers["reaction"] = Reaction.sayLess.label
        XCTAssertTrue(MediaTemplate.reaction.document(fill, lines: MediaTemplate.reaction.ownLines(fill))
            .contains("<d>[English] Say less.</d>"))
        fill.answers["reaction"] = Reaction.mindBlown.label
        XCTAssertTrue(MediaTemplate.reaction.document(fill, lines: MediaTemplate.reaction.ownLines(fill))
            .contains("paws rise"), "a dog's paws go to its head, not hands")
    }

    func testADancePartyTakesOneToFourOfAnyoneAndPutsThePetsOnTheirHindLegs() {
        var fill = MediaTemplate.party.defaultFill()
        XCTAssertEqual(MediaTemplate.party.missing(fill), "Who dances out front?")
        fill.cast["a"] = member("Max", .dog)
        XCTAssertNil(MediaTemplate.party.missing(fill))
        XCTAssertTrue(MediaTemplate.party.document(fill, lines: [:]).contains("the whole crowd dances along behind"))
        fill.cast["b"] = member("Kim", .man)
        fill.cast["d"] = member("Luna", .cat)
        let doc = MediaTemplate.party.document(fill, lines: [:])
        XCTAssertEqual(MediaTemplate.party.cast(fill).map(\.name), ["Max", "Kim", "Luna"])
        XCTAssertTrue(doc.contains("<Subject 2> and <Subject 3> dance right behind"))
        XCTAssertTrue(doc.contains("<Subject 1> and <Subject 3> dancing upright on their hind legs"), "the man stays on his feet")
        XCTAssertFalse(doc.contains("<Picture 4>"), "a skipped blank does not leave a gap in the numbering")
        XCTAssertTrue(MediaTemplate.party.lineKeys(fill).isEmpty)
    }

    /// The bandana a photo shows is lost unless the document says to keep it.
    func testWhatTheyLookLikeIsKeptNotJustDescribed() {
        var fill = filled(.reaction)
        fill.cast["who"] = member("Max", .dog, looks: "red bandana")
        let doc = MediaTemplate.reaction.document(fill, lines: MediaTemplate.reaction.ownLines(fill))
        let retention = doc.components(separatedBy: "retention_analysis:")[1].components(separatedBy: "detailed_description:")[0]
        XCTAssertTrue(retention.contains("including red bandana"))
    }

    func testListsJoinInEnglishWhateverTheMacsLanguage() {
        XCTAssertEqual(TemplateText.list(["a"]), "a")
        XCTAssertEqual(TemplateText.list(["a", "b"]), "a and b")
        XCTAssertEqual(TemplateText.list(["a", "b", "c"]), "a, b and c")
    }

    /// Every gallery card shows what it makes: a clip, or a still for a picture or a song.
    func testEveryTemplateShipsAGallerySample() {
        let resources = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../Sources/MLXServe/Resources").standardizedFileURL
        for t in MediaTemplate.allCases {
            let found = ["mp4", "jpg"].contains {
                FileManager.default.fileExists(atPath: resources.appendingPathComponent("\(t.sampleName).\($0)").path)
            }
            XCTAssertTrue(found, "no gallery sample for \(t)")
        }
    }

    /// Every option of every picker shows how it looks: a clip per option ships in the app.
    func testEveryChoiceHasAPreviewClip() {
        let resources = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .appendingPathComponent("../../Sources/MLXServe/Resources").standardizedFileURL
        XCTAssertEqual(MediaTemplate.reaction.sampleName(choice: "Mind blown"), "template-reaction-mind-blown")
        for t in MediaTemplate.allCases {
            for slot in t.slots {
                guard case .choice(let options) = slot.kind else { continue }
                for o in options {
                    let file = resources.appendingPathComponent(t.sampleName(choice: o) + ".mp4")
                    XCTAssertTrue(FileManager.default.fileExists(atPath: file.path), "no preview for \(t) / \(o)")
                }
            }
        }
    }

    /// The stage plays the selected option's clip, and follows a switch.
    func testTheStageShowsTheSelectedOption() {
        var fill = MediaTemplate.pets.defaultFill()
        XCTAssertEqual(MediaTemplate.pets.choiceSample(fill), "template-pets-podcast")
        fill.answers["format"] = PetFormat.couch.label
        XCTAssertEqual(MediaTemplate.pets.choiceSample(fill), "template-pets-couch-confession")
        XCTAssertEqual(MediaTemplate.reaction.choiceSample(MediaTemplate.reaction.defaultFill()), "template-reaction-mind-blown")
        XCTAssertNil(MediaTemplate.santa.choiceSample(MediaTemplate.santa.defaultFill()), "no options, no option clip")
    }

    // MARK: - Image and song templates

    func testTheGalleryHasThreeSectionsAndEveryTemplateBelongsToOne() {
        XCTAssertEqual(TemplateKind.allCases.map(\.title), ["Video", "Image", "Audio"])
        XCTAssertEqual(MediaTemplate.allCases.filter { $0.kind == .image }, [.portrait, .headshot, .poster, .figure])
        XCTAssertEqual(MediaTemplate.allCases.filter { $0.kind == .audio }, [.birthday, .theme, .ballad, .roast])
        XCTAssertEqual(MediaTemplate.allCases.filter { $0.kind == .video }.count, 5)
    }

    /// The picture keeps who is in the photo; the meme text is ours, never the model's.
    @MainActor
    func testAnImageTemplateEditsTheirPhotoAndKeepsWhoTheyAre() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tpl-img-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let cast = CastStore(rootDir: dir)
        var fill = MediaTemplate.portrait.defaultFill()
        fill.cast["who"] = member("Max", .dog, looks: "red bandana")
        fill.answers["top"] = "his royal highness"
        let prompt = MediaTemplate.portrait.imagePrompt(fill)
        XCTAssertTrue(prompt.contains("the same dog from the photo"))
        XCTAssertTrue(prompt.contains("red bandana"))
        XCTAssertFalse(prompt.lowercased().contains("royal highness"), "the meme text is burned on, not drawn by the model")
        let r = TemplateRunner.imageRequest(.portrait, fill: fill, shape: .story, seed: 3, cast: cast, peer: nil)
        XCTAssertEqual(r.model.repo, MediaTemplate.imageModel.repo)
        XCTAssertTrue(r.editMode)
        XCTAssertEqual(r.initImagePath, cast.photoURL(try XCTUnwrap(fill.cast["who"])).path)
        XCTAssertEqual([r.width, r.height], [576, 1024])
        XCTAssertNil(MediaTemplate.portrait.choiceSample(fill))
    }

    @MainActor
    func testASongIsSungFromTheLyricsTheChatModelWrote() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tpl-song-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        var fill = MediaTemplate.birthday.defaultFill()
        fill.cast["who"] = member("Sarah", .woman)
        XCTAssertEqual(MediaTemplate.birthday.lineKeys(fill), ["title", "lyrics"])
        XCTAssertTrue(MediaTemplate.birthday.ask(fill).contains("Sarah"))
        XCTAssertTrue(MediaTemplate.birthday.fallbackLines(fill)["lyrics"]?.contains("Sarah") == true)
        let reply = #"{"title": "Sarah's Day", "lyrics": "[verse]\nWake up Sarah\n\n[chorus]\nHappy birthday"}"#
        let lines = try XCTUnwrap(TemplateText.parseLines(reply, keys: ["title", "lyrics"]))
        XCTAssertEqual(lines["lyrics"], "[verse]\nWake up Sarah\n\n[chorus]\nHappy birthday", "lyrics keep their line breaks")
        let r = TemplateRunner.songRequest(.birthday, fill: fill, lines: lines, seed: 3, peer: "m@studio")
        XCTAssertEqual(r.lyrics, lines["lyrics"])
        XCTAssertEqual(r.durationSeconds, MediaTemplate.birthday.seconds)
        XCTAssertEqual(r.lanModelId, "m@studio")
        XCTAssertFalse(r.prompt.isEmpty)
    }

    func testEachKindIsMadeOnItsOwnModelAndAskedOfAPeerByItsOwnCapability() {
        func lan(_ name: String, _ cap: String) -> ModelInfo {
            ModelInfo(name: name, quantBits: 0, layers: 0, hiddenSize: 0, vocabSize: 0,
                      contextLength: 0, modelMaxTokens: 0, capabilities: [cap], lanPeer: "studio")
        }
        let image = (MediaTemplate.imageModel.repo as NSString).lastPathComponent + "@studio"
        let models = [lan(image, "image"), lan("Other@studio", "music")]
        XCTAssertEqual(TemplateReadiness.peer(for: .poster, in: models), image)
        XCTAssertNil(TemplateReadiness.peer(for: .ballad, in: models))
        XCTAssertNil(TemplateReadiness.peer(for: .santa, in: models))
    }

    func testAnActionFigureKeepsTheirFaceInItsBox() {
        var fill = MediaTemplate.figure.defaultFill()
        fill.cast["who"] = member("Luna", .cat)
        let prompt = MediaTemplate.figure.imagePrompt(fill)
        XCTAssertTrue(prompt.contains("the same cat from the photo"))
        XCTAssertTrue(prompt.contains("action figure"))
    }

    func testARoastRapTeasesThemKindly() {
        var fill = MediaTemplate.roast.defaultFill()
        fill.cast["who"] = member("Kim", .man)
        fill.answers["tease"] = "always being late"
        let ask = MediaTemplate.roast.ask(fill)
        XCTAssertTrue(ask.contains("Kim") && ask.contains("always being late") && ask.contains("family friendly"))
        XCTAssertTrue(MediaTemplate.roast.songStyle(fill).contains("hip-hop"))
        XCTAssertTrue(MediaTemplate.roast.fallbackLines(fill)["lyrics"]?.contains("Kim") == true)
    }

    // MARK: - Review fixes

    /// A replaced photo or a removed member is seen at Make, not the copy picked earlier.
    func testTheFillFollowsTheCastAsItIsNow() {
        let max = member("Max", .dog)
        var renamed = max
        renamed.photo = "new.jpg"
        let fill = TemplateFill(cast: ["a": max, "b": member("Gone", .cat)], answers: ["topic": "socks"])
        let now = fill.resolved(in: [renamed])
        XCTAssertEqual(now.cast["a"]?.photo, "new.jpg")
        XCTAssertNil(now.cast["b"], "a member removed since is dropped, so the form asks for them again")
        XCTAssertEqual(now.answers, fill.answers)
    }

    func testEveryLineTheTemplateSpeaksMustBeWritten() {
        var fill = filled(.pets)
        fill.cast["b"] = nil
        XCTAssertFalse(MediaTemplate.pets.linesMissing(fill, ["a": "Hi."]))
        fill.cast["b"] = member("B", .cat)
        XCTAssertTrue(MediaTemplate.pets.linesMissing(fill, ["a": "Hi."]), "a speaker added after writing has no line")
        XCTAssertTrue(MediaTemplate.pets.linesMissing(fill, ["a": "Hi.", "b": "  "]))
        XCTAssertFalse(MediaTemplate.party.linesMissing(filled(.party), [:]))
    }

    /// The song's video must not share a name with the Music tab's sidecar for the song.
    @MainActor
    func testASongsVideoDoesNotClaimTheMusicTabsSidecar() {
        let wav = URL(fileURLWithPath: "/m/2026-11-19_song.wav")
        let video = TemplateRunner.songVideoURL(wav)
        XCTAssertEqual(video.pathExtension, "mp4")
        XCTAssertNotEqual(VideoGenService.sidecarPath(forVideo: video.path), "/m/2026-11-19_song.txt")
    }

    /// Another take of a video whose cast member is gone does nothing, and keeps the video showing.
    @MainActor
    func testAnotherTakeWithAMissingCastMemberKeepsTheResult() throws {
        _ = NSApplication.shared
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tpl-take-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let clip = dir.appendingPathComponent("c.mp4")
        FileManager.default.createFile(atPath: clip.path, contents: Data("x".utf8))
        let app = AppState(startServerAtLaunch: false)
        let run = TemplateRunner(history: TemplateHistory(rootDir: dir.appendingPathComponent("h")))
        run.restore(TemplateHistoryEntry(template: "reaction", clip: clip.path, cast: ["who": UUID()],
                                         answers: ["reaction": "Eye roll"], lines: [:], shape: "square", seed: 1), cast: app.cast)
        run.anotherTake(app: app)
        XCTAssertEqual(run.phase, .done)
        XCTAssertEqual(run.result, clip)
    }

    /// Any member of the cast fills any blank, and the words fit what they are.
    func testEveryBlankTakesAnyoneInTheCast() {
        for t in MediaTemplate.allCases {
            for kind in [CastKind.dog, .boy] {
                let fill = filled(t, as: kind)
                XCTAssertNil(t.missing(fill), "\(t) refused a \(kind)")
                let lines = Dictionary(uniqueKeysWithValues: t.lineKeys(fill).map { ($0, "Hello there.") })
                let words = [t.document(fill, lines: lines), t.ask(fill), t.imagePrompt(fill)].joined(separator: " ")
                if kind == .dog {
                    XCTAssertFalse(words.contains("hair and clothing"), "\(t) gives a dog hair and clothing")
                    XCTAssertFalse(words.contains("a child") || words.contains("the child"), "\(t) calls a dog a child")
                } else {
                    XCTAssertFalse(words.contains("ears perked"), "\(t) perks a boy's ears")
                    XCTAssertFalse(words.contains("talking-pet"), "\(t) calls a boy a pet")
                }
            }
        }
    }

    /// A roast's cover shows them as a rapper, still themselves; the other songs keep their photo.
    func testARoastRapsCoverMakesThemARapper() {
        var fill = MediaTemplate.roast.defaultFill()
        fill.cast["who"] = member("Luna", .cat)
        let prompt = MediaTemplate.roast.songCoverPrompt(fill)
        XCTAssertTrue(prompt?.contains("the same cat from the photo") == true)
        XCTAssertTrue(prompt?.contains("rapper") == true)
        XCTAssertNil(MediaTemplate.birthday.songCoverPrompt(fill))
    }

    /// Advanced Editor hands the full pane exactly what the template would have made.
    @MainActor
    func testTheAdvancedEditorGetsWhatTheTemplateWouldMake() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tpl-adv-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let cast = CastStore(rootDir: dir)
        var fill = MediaTemplate.figure.defaultFill()
        fill.cast["who"] = member("Max", .dog)
        let image = TemplateRunner.imageSettings(.figure, fill: fill, shape: .story, seed: 9, cast: cast, peer: nil)
        XCTAssertEqual(image.modelId, MediaTemplate.imageModel.id)
        XCTAssertEqual(image.prompt, MediaTemplate.figure.imagePrompt(fill))
        XCTAssertEqual(image.sourcePath, cast.photoURL(try XCTUnwrap(fill.cast["who"])).path)
        XCTAssertTrue(image.editMode)
        XCTAssertEqual([image.customWidth, image.customHeight], [576, 1024])
        XCTAssertEqual(image.seed, 9)

        let lines = ["title": "T", "lyrics": "[verse]\nHi"]
        let song = TemplateRunner.musicSettings(.roast, fill: fill, lines: lines, seed: 9, peer: "m@studio")
        XCTAssertEqual(song.modelId, LanPick.persisted(lanModel: "m@studio", presetId: MediaTemplate.songModel.id))
        XCTAssertEqual(song.prompt, MediaTemplate.roast.songStyle(fill))
        XCTAssertEqual(song.lyrics, "[verse]\nHi")
        XCTAssertEqual(song.durationSeconds, MediaTemplate.roast.seconds)
        XCTAssertFalse(song.autoLength)
    }
}
