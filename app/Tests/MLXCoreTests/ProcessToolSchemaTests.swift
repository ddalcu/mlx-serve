import XCTest
@testable import MLXCore

/// Validates the REAL `AgentPrompt.toolDefinitionsJSON` (the one sent to the
/// model) after adding the background-process tools — complements the
/// hardcoded-snapshot ToolKeyOrderTests.
final class ProcessToolSchemaTests: XCTestCase {

    private func defs() -> [[String: Any]] {
        let data = AgentPrompt.toolDefinitionsJSON.data(using: .utf8)!
        return try! JSONSerialization.jsonObject(with: data) as! [[String: Any]]
    }

    private func byName() -> [String: [String: Any]] {
        Dictionary(uniqueKeysWithValues: defs().compactMap { d -> (String, [String: Any])? in
            guard let fn = d["function"] as? [String: Any], let n = fn["name"] as? String else { return nil }
            return (n, fn)
        })
    }

    func testToolDefinitionsJSONIsValid() {
        XCTAssertNotNil(try? JSONSerialization.jsonObject(with: AgentPrompt.toolDefinitionsJSON.data(using: .utf8)!))
    }

    func testToolCountIncludesNewProcessTools() {
        // 11 base (shell, cwd, writeFile, readFile, editFile, searchFiles,
        // listFiles, browse, webSearch, saveMemory, createTask) + 3 process
        // (killProcess, readProcessOutput, listProcesses) + 4 media
        // (generate_image, generate_speech, generate_music, generate_video) = 18.
        XCTAssertEqual(defs().count, 18)
    }

    /// The four media-generation tools, each with exactly one required argument
    /// and its optional knobs. Speech and music are separate because their
    /// arguments have nothing in common.
    func testMediaGenerationToolsPresent() {
        let tools = byName()
        let required = ["generate_image": "prompt", "generate_speech": "text",
                        "generate_music": "prompt", "generate_video": "prompt"]
        for (n, key) in required {
            XCTAssertNotNil(tools[n], "\(n) missing from schema")
            XCTAssertEqual((tools[n]!["parameters"] as! [String: Any])["required"] as! [String], [key])
        }
        XCTAssertNil(tools["generate_audio"], "generate_audio split into speech + music")
    }

    /// Every optional knob a media tool advertises must be one `MediaToolArgs`
    /// actually reads — an argument the parser ignores is a control that does
    /// nothing, spent out of a small model's budget.
    func testMediaToolOptionalArgumentsAreOnesWeRead() {
        let tools = byName()
        let expected = ["generate_image": Set(["prompt", "size"]),
                        "generate_speech": Set(["text", "speed"]),
                        "generate_music": Set(["prompt", "lyrics", "duration_seconds",
                                               "bpm", "keyscale", "time_signature", "vocal_language"]),
                        "generate_video": Set(["prompt", "seconds", "size"])]
        for (n, keys) in expected {
            let props = (tools[n]!["parameters"] as! [String: Any])["properties"] as! [String: Any]
            XCTAssertEqual(Set(props.keys), keys, "\(n) advertises the wrong argument set")
        }
    }

    @MainActor
    func testEveryMediaToolShowsACallTheModelCanCopy() {
        for n in ["generate_image", "generate_speech", "generate_music", "generate_video"] {
            let example = AgentEngine.toolExample(for: n)
            XCTAssertTrue(example.contains("{"), "\(n) needs an Example: in its description")
            XCTAssertTrue(example.contains("prompt") || example.contains("text"),
                          "\(n) example must show its required key: \(example)")
        }
    }

    func testShellHasOptionalRunInBackgroundParam() {
        let shell = byName()["shell"]!
        let params = shell["parameters"] as! [String: Any]
        let props = params["properties"] as! [String: Any]
        XCTAssertNotNil(props["run_in_background"], "shell must advertise run_in_background")
        // Opt-in: only `command` stays required.
        XCTAssertEqual(params["required"] as! [String], ["command"])
    }

    func testNewToolsPresentWithCorrectRequired() {
        let tools = byName()
        for n in ["killProcess", "readProcessOutput", "listProcesses"] {
            XCTAssertNotNil(tools[n], "\(n) missing from schema")
        }
        XCTAssertEqual((tools["killProcess"]!["parameters"] as! [String: Any])["required"] as! [String], ["handle"])
        XCTAssertEqual((tools["readProcessOutput"]!["parameters"] as! [String: Any])["required"] as! [String], ["handle"])
        XCTAssertEqual((tools["listProcesses"]!["parameters"] as! [String: Any])["required"] as! [String], [])
    }

    @MainActor
    func testHandleToolsHaveAHandleExample() {
        // toolExample relies on an "Example: " marker in each description.
        // (listProcesses' example is literally `{}`, which toolExample can't tell
        // from "no marker" — so only the handle-taking tools are checked here.)
        for n in ["killProcess", "readProcessOutput"] {
            XCTAssertTrue(AgentEngine.toolExample(for: n).contains("handle"),
                          "\(n) needs a handle Example: in its description")
        }
    }

    func testToolDefinitionsParsedIncludesNewTools() {
        let names = AgentPrompt.toolDefinitions.compactMap { ($0["function"] as? [String: Any])?["name"] as? String }
        XCTAssertTrue(names.contains("killProcess"))
        XCTAssertTrue(names.contains("readProcessOutput"))
        XCTAssertTrue(names.contains("listProcesses"))
    }

    /// The path-omission class: weak models routinely emit a `writeFile` with a
    /// huge `content` value and NO `path` (the big blob crowds out the small
    /// required param). The cheap lever is the description — it must FOREGROUND
    /// `path` (name it before `content`) so an autoregressive model commits the
    /// path before the body, and the example a model copies must show `path`.
    @MainActor
    func testWriteFileDescriptionForegroundsPath() {
        let writeFile = byName()["writeFile"]!
        let params = writeFile["parameters"] as! [String: Any]
        // Required contract is unchanged.
        XCTAssertEqual(params["required"] as! [String], ["path", "content"])

        let desc = writeFile["description"] as! String
        let firstPath = desc.range(of: "path")!.lowerBound
        let firstContent = desc.range(of: "content")!.lowerBound
        XCTAssertLessThan(firstPath, firstContent,
            "writeFile description must name `path` before `content` (path-omission steer)")

        // The copy-paste example must carry an explicit path.
        XCTAssertTrue(AgentEngine.toolExample(for: "writeFile").contains("\"path\""),
            "writeFile Example: must show a path the model can copy")
    }
}
