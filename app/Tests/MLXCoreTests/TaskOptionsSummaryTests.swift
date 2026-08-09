import XCTest
@testable import MLXCore

/// Progressive disclosure's one hazard is a setting that IS set but invisible.
/// The summary line is the mitigation, so these pin what it must say.
final class TaskOptionsSummaryTests: XCTestCase {

    /// All defaults ⇒ nothing to say. Not an empty string: "Options" with a
    /// blank line under it reads as a section that failed to load.
    func testAllDefaultsProduceNoLine() {
        XCTAssertNil(TaskOptionsSummary.text(agentName: nil, modelName: nil, useMCP: false))
    }

    /// The VALUE, not the field name — the line exists to answer "what did I
    /// set?" without expanding the row.
    func testItNamesTheChosenValues() {
        XCTAssertEqual(
            TaskOptionsSummary.text(agentName: "Chef", modelName: nil, useMCP: false),
            "Chef")
        XCTAssertEqual(
            TaskOptionsSummary.text(agentName: nil, modelName: "gemma4-31b", useMCP: false),
            "gemma4-31b")
        XCTAssertEqual(
            TaskOptionsSummary.text(agentName: nil, modelName: nil, useMCP: true),
            "MCP")
    }

    /// Order is the order of the controls, so the line reads as a summary of
    /// the section rather than an arbitrary set.
    func testEverythingSetReadsInControlOrder() {
        XCTAssertEqual(
            TaskOptionsSummary.text(agentName: "Chef", modelName: "gemma4-31b", useMCP: true),
            "Chef · gemma4-31b · MCP")
    }

    /// A field cleared to empty/whitespace is not a choice — claiming it would
    /// advertise a customization that isn't there.
    func testBlankValuesAreNotCustomizations() {
        XCTAssertNil(TaskOptionsSummary.text(agentName: "", modelName: "   ", useMCP: false))
        XCTAssertEqual(
            TaskOptionsSummary.text(agentName: "  ", modelName: "gemma4-31b", useMCP: false),
            "gemma4-31b")
    }

    /// Names are echoed as written — an agent called "MCP" or a model with odd
    /// casing must not be re-cased by the summary.
    func testValuesAreEchoedVerbatim() {
        XCTAssertEqual(
            TaskOptionsSummary.text(agentName: "  Chef  ", modelName: nil, useMCP: false),
            "Chef",
            "surrounding whitespace is trimmed, but the name itself is untouched")
        XCTAssertEqual(
            TaskOptionsSummary.text(agentName: "iPhone helper", modelName: nil, useMCP: false),
            "iPhone helper")
    }
}
