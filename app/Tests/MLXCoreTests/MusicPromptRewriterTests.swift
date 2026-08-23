import XCTest
@testable import MLXCore

/// The rewrite asks the chat model to match the CURRENT family's examples:
/// a one-line ACE-Step caption must not come back as a Music 3 three-block
/// caption, and lyrics must keep the section-tag grammar the server parses.
final class MusicPromptRewriterTests: XCTestCase {

    func testStyleRewriteShowsOnlyTheCurrentFamilysExamples() {
        let ace = MusicPromptRewriter.request(.style, text: "sad piano", family: .acestep,
                                              other: "", instrumental: true, language: "en")
        XCTAssertTrue(ace.system.contains(MusicPrompt.builtinStyles[0].body))
        XCTAssertFalse(ace.system.contains("Global Metadata"))
        XCTAssertTrue(ace.user.contains("sad piano"))
        XCTAssertTrue(ace.user.lowercased().contains("instrumental"))

        let m3 = MusicPromptRewriter.request(.style, text: "sad piano", family: .minimaxMusic3,
                                             other: "", instrumental: false, language: "en")
        XCTAssertTrue(m3.system.contains("Global Metadata"))
        XCTAssertFalse(m3.system.contains(MusicPrompt.builtinStyles[0].body))
    }

    func testLyricsRewriteCarriesSectionTagsAndTheStylePrompt() {
        let r = MusicPromptRewriter.request(.lyrics, text: "a song about rain", family: .acestep,
                                            other: "dreamy synthwave", instrumental: false, language: "fr")
        XCTAssertTrue(r.system.contains("[Chorus]"))
        XCTAssertTrue(r.system.contains(MusicPrompt.builtinLyrics[0].body))
        XCTAssertTrue(r.user.contains("dreamy synthwave"))
        XCTAssertTrue(r.user.contains("French"))
    }
}
