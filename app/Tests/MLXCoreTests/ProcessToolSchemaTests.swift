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
        // 11 existing (shell, cwd, writeFile, readFile, editFile, searchFiles,
        // listFiles, browse, webSearch, saveMemory, createTask) + 3 new = 14.
        XCTAssertEqual(defs().count, 14)
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
}
