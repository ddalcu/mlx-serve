import XCTest
@testable import MLXCore

/// What a tool-call card shows, derived from the structured record instead of
/// from the summary text the engine assembled to be read back.
final class ToolCallDisplayTests: XCTestCase {

    // MARK: - Arguments

    /// A model writes `path` before `content` for a reason, and
    /// `JSONSerialization` hands back an unordered dictionary — so the order is
    /// recovered from the raw JSON. Alphabetical would put a file body above
    /// the path it was written to.
    func testArgumentsKeepTheOrderTheModelSentThem() {
        let json = #"{"path":"a.txt","content":"hello","append":"true"}"#
        XCTAssertEqual(ToolCallDisplay.arguments(fromJSON: json).map(\.name),
                       ["path", "content", "append"])
    }

    /// A key that also occurs inside a VALUE must not decide the order.
    func testAKeyMentionedInsideAValueDoesNotMoveIt() {
        let json = #"{"command":"grep -r \"path\" .","path":"src"}"#
        XCTAssertEqual(ToolCallDisplay.arguments(fromJSON: json).map(\.name), ["command", "path"])
    }

    /// Newlines are what make the panel unreadable: a file body would turn a
    /// two-line card into a two-screen one.
    func testEveryKindOfLineBreakBecomesASpace() {
        let json = "{\"content\":\"one\\ntwo\\r\\nthree\\tfour\"}"
        XCTAssertEqual(ToolCallDisplay.arguments(fromJSON: json).first?.value, "one two three four")
    }

    func testRunsOfWhitespaceCollapse() {
        XCTAssertEqual(ToolCallDisplay.flatten("a   \n\n   b"), "a b")
    }

    /// The panel summarises what was asked; the file itself is one readFile
    /// away.
    func testALongValueIsCutWithAnEllipsis() {
        let value = ToolCallDisplay.flatten(String(repeating: "x", count: 5000))
        XCTAssertTrue(value.hasSuffix("…"))
        XCTAssertLessThan(value.count, 5000)
    }

    /// Numbers and booleans reach us as JSON types, not strings — a card that
    /// showed only strings would drop `startLine` and `recursive`.
    func testNonStringValuesAreShown() {
        let json = #"{"startLine":42,"recursive":true,"missing":null}"#
        let values = Dictionary(uniqueKeysWithValues:
            ToolCallDisplay.arguments(fromJSON: json).map { ($0.name, $0.value) })
        XCTAssertEqual(values["startLine"], "42")
        XCTAssertEqual(values["recursive"], "true")
        XCTAssertEqual(values["missing"], "null")
    }

    func testMalformedJSONYieldsNothingRatherThanCrashing() {
        XCTAssertTrue(ToolCallDisplay.arguments(fromJSON: "{not json").isEmpty)
        XCTAssertTrue(ToolCallDisplay.arguments(fromJSON: "").isEmpty)
    }

    // MARK: - The name

    /// Qualified: the test target declares its own `SerializedToolCall` (in
    /// `AgentHarnessTests`), which shadows the app's inside this module.
    func testTheTitleComesFromTheStructuredCall() {
        let calls = [MLXCore.SerializedToolCall(id: "1", name: "writeFile", arguments: "{}")]
        XCTAssertEqual(ToolCallDisplay.title(calls: calls, summary: "ignored"), "writeFile")
    }

    /// A history written before the calls were recorded still has
    /// `**name**(args)` to read.
    func testTheTitleFallsBackToTheSummaryText() {
        XCTAssertEqual(ToolCallDisplay.title(calls: [], summary: "**readFile**(path: a.txt)"),
                       "readFile")
    }

    // MARK: - How the name reads

    /// `<server>__<tool>` is a wire format, not something to read. The card
    /// shows it as a path — the same two halves `MCPManager` dispatches on.
    func testAnMCPToolNameReadsAsAPath() {
        XCTAssertEqual(ToolCallDisplay.displayName("perry-memory__get_context"),
                       "perry-memory/get_context")
    }

    func testABuiltInNameIsUntouched() {
        XCTAssertEqual(ToolCallDisplay.displayName("writeFile"), "writeFile")
    }

    /// The split is at the FIRST `__` because that is where dispatch splits:
    /// `namespacedName` collapses `__` inside the SERVER id, so a later one can
    /// only belong to the tool. Rendering it any other way would name a tool
    /// the call never went to.
    func testOnlyTheFirstSeparatorSplits() {
        XCTAssertEqual(ToolCallDisplay.displayName("srv__do__thing"), "srv/do__thing")
    }

    /// Half a name is not a namespace: an empty side means this is not the
    /// `<server>__<tool>` shape, and inventing a slash would be a lie about
    /// where it ran.
    func testAnEmptyHalfIsNotANamespace() {
        XCTAssertEqual(ToolCallDisplay.displayName("__orphan"), "__orphan")
        XCTAssertEqual(ToolCallDisplay.displayName("orphan__"), "orphan__")
    }

    // MARK: - browse, whose action is really its name

    func testBrowseCarriesItsActionAsPartOfItsName() {
        let args = ToolCallDisplay.arguments(fromJSON: #"{"action":"click","selector":"button"}"#)
        XCTAssertEqual(ToolCallDisplay.variant(toolName: "browse", arguments: args), "click")
    }

    func testAToolWithNoVariantHasNone() {
        let args = ToolCallDisplay.arguments(fromJSON: #"{"path":"a.txt"}"#)
        XCTAssertNil(ToolCallDisplay.variant(toolName: "readFile", arguments: args))
    }

    /// The browser keeps its page between calls, so five of the seven actions
    /// send no URL at all — the interesting argument is whichever one the
    /// action actually uses.
    func testBrowseShowsWhicheverArgumentItsActionUses() {
        func headline(_ json: String) -> String? {
            ToolCallDisplay.headline(toolName: "browse",
                                     arguments: ToolCallDisplay.arguments(fromJSON: json))
        }
        XCTAssertEqual(headline(#"{"action":"navigate","url":"example.com"}"#), "example.com")
        XCTAssertEqual(headline(#"{"action":"click","selector":"button.submit"}"#), "button.submit")
        XCTAssertEqual(headline(#"{"action":"executeJS","script":"document.title"}"#), "document.title")
        XCTAssertNil(headline(#"{"action":"screenshot"}"#))
    }

    /// An MCP tool is namespaced `<server>__<tool>`, so a server exposing a
    /// familiar name gets the same treatment as the built-in.
    func testAnMCPToolIsLookedUpUnderItsBareName() {
        let args = ToolCallDisplay.arguments(fromJSON: #"{"path":"a.txt"}"#)
        XCTAssertEqual(ToolCallDisplay.headline(toolName: "files__readFile", arguments: args),
                       "a.txt")
    }

    /// A tool with no rule shows nothing extra — which is what every tool did
    /// before this existed, so a NEW tool degrades to the old behaviour rather
    /// than to a wrong guess.
    func testAnUnknownToolHasNoHeadline() {
        let args = ToolCallDisplay.arguments(fromJSON: #"{"whatever":"value"}"#)
        XCTAssertNil(ToolCallDisplay.headline(toolName: "somethingNew", arguments: args))
    }

    // MARK: - What came of it

    func testWriteFileReportsWhatItWrote() {
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "writeFile",
                                                      result: "Wrote 1053 characters to a.md"),
                       "1053 chars")
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "writeFile",
                                                      result: "Appended 42 characters to a.md"),
                       "42 chars")
    }

    /// readFile numbers every line it returns, and the model works in lines —
    /// `editFile` takes a startLine. The metadata header a large file carries
    /// is not one of them.
    func testReadFileCountsLinesAndNotItsOwnHeader() {
        let plain = "1| one\n2| two\n3| three"
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "readFile", result: plain),
                       "3 lines")

        let withHeader = "[File: a.swift | Lines: 1-2 of 900 | 40000 bytes]\n1| one\n2| two"
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "readFile", result: withHeader),
                       "2 lines")
    }

    /// Line mode reports the range it replaced AFTER clamping to the file, so
    /// a model asking for 4-999 in a 50-line file reports the truth.
    func testEditFileCountsTheReplacedRange() {
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "editFile",
                                                      result: "Edited a.swift (replaced lines 4-9)"),
                       "6 lines")
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "editFile",
                                                      result: "Edited a.swift (replaced lines 4-4)"),
                       "1 line")
    }

    /// Text mode (find/replace) returns "Edited x" and nothing else: `find` can
    /// be part of one line, so there is no line count to give. Silence beats a
    /// number that is not there.
    func testEditFileInTextModeReportsNothing() {
        XCTAssertNil(ToolCallDisplay.resultHeadline(toolName: "editFile", result: "Edited a.swift"))
    }

    /// The separator is what tells a match from the context lines around it:
    /// `path:line:text` is a hit, `path-line-text` is context.
    func testSearchFilesCountsMatchesAndFilesButNotContext() {
        let output = """
        src/a.swift:12:let x = TODO
        src/a.swift-13-  next line
        src/a.swift:40:// TODO again
        src/b.swift:3:TODO
        """
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "searchFiles", result: output),
                       "3 occurrences in 2 files")
    }

    func testSearchFilesSaysWhenThereWasNothing() {
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "searchFiles",
                                                      result: "No matches found for 'zzz'"),
                       "nothing found")
    }

    func testASingleHitIsSingular() {
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "searchFiles",
                                                      result: "a.swift:1:x"),
                       "1 occurrence in 1 file")
    }

    /// The listing stops at 200 and says so on a line of its own — which is not
    /// an entry, and means the real number is unknown.
    func testListFilesReportsWhenItHitItsCeiling() {
        let entries = (1...200).map { "file\($0).txt" }.joined(separator: "\n")
        XCTAssertEqual(ToolCallDisplay.resultHeadline(
            toolName: "listFiles", result: entries + "\n[... truncated at 200 entries]"),
                       "200+ files found")
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "listFiles", result: "a.txt\nb.txt"),
                       "2 files found")
        XCTAssertEqual(ToolCallDisplay.resultHeadline(toolName: "listFiles",
                                                      result: "No files found in src"),
                       "nothing found")
    }

    // MARK: - Which call started which process

    /// `processHandles` is collected across the whole ROUND and pinned to one
    /// message, so nothing in the model records which call started which
    /// process. The handle is in the result text — in all three spellings the
    /// tool produces.
    func testTheHandleIsRecoveredFromEveryFormOfTheMessage() {
        let host = "Started in background as bg1 (pid 4242). It keeps running"
        let sandbox = "Started in the SANDBOX background (isolated Linux guest) as bg2 (guest pid 14256)."
        let adopted = "Still running after 30s — now managed in the background as bg7 (pid 99), NOT killed."

        XCTAssertEqual(ToolCallDisplay.backgroundHandle(inResult: host), "bg1")
        XCTAssertEqual(ToolCallDisplay.backgroundHandle(inResult: sandbox), "bg2")
        XCTAssertEqual(ToolCallDisplay.backgroundHandle(inResult: adopted), "bg7")
    }

    /// A foreground command names no handle, and neither does the tool's advice
    /// about how to stop one — a kill button attached to a mention would kill
    /// something the call never started.
    func testAResultWithNoProcessNamesNoHandle() {
        XCTAssertNil(ToolCallDisplay.backgroundHandle(inResult: "[cwd: /x]\nhello\n"))
        XCTAssertNil(ToolCallDisplay.backgroundHandle(
            inResult: #"stop it with killProcess {"handle": "bg1"}"#))
    }

    // MARK: - The result body

    func testTheResultDropsTheMarkerTheSummaryWasBuiltWith() {
        XCTAssertEqual(ToolCallDisplay.resultBody("**shell** → Killed bg1."), "Killed bg1.")
    }

    /// A summary that does not carry the marker is shown as it is rather than
    /// silently emptied.
    func testASummaryWithoutTheMarkerSurvives() {
        XCTAssertEqual(ToolCallDisplay.resultBody("**plain**"), "plain")
        XCTAssertEqual(ToolCallDisplay.resultBody("no markers here"), "no markers here")
    }
}
