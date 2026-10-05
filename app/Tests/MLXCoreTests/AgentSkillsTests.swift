import XCTest
@testable import MLXCore

final class AgentSkillsTests: XCTestCase {

    private func tempDir() throws -> String {
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("agent-skills-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(atPath: dir) }
        return dir
    }

    // Twin of the Zig `skill install` test: missing files are written, an edited one is kept, the agent's dir links to it.
    func testInstallWritesMissingKeepsEditedAndLinks() throws {
        let root = try tempDir()
        let skill = root + "/skills/mlx-serve"
        try FileManager.default.createDirectory(atPath: skill, withIntermediateDirectories: true)
        try "edited".write(toFile: skill + "/SKILL.md", atomically: true, encoding: .utf8)

        AgentSkills.install(agentId: "pi", root: root)
        AgentSkills.install(agentId: "claude", root: root)

        XCTAssertEqual(try String(contentsOfFile: root + "/pi/skills/mlx-serve/SKILL.md", encoding: .utf8), "edited")
        XCTAssertEqual(try String(contentsOfFile: root + "/claude/plugin/skills/mlx-serve/SKILL.md", encoding: .utf8), "edited")
        XCTAssertTrue(try String(contentsOfFile: skill + "/media.md", encoding: .utf8).contains("/v1/images/generations"))
        XCTAssertTrue(FileManager.default.fileExists(atPath: root + "/claude/plugin/.claude-plugin/plugin.json"))
    }

    func testCodexGetsNoSkillLinkAndNoDedicatedDir() throws {
        XCTAssertNil(AgentSkills.linkPath(agentId: "codex"))
        let root = try tempDir()
        AgentSkills.install(agentId: "codex", root: root)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root + "/codex"))
    }

    // The update menu restores every shipped skill file, backing up the edited ones where no scan finds them as skills.
    func testRefreshBacksUpEditsOutsideTheSkillsDir() throws {
        let root = try tempDir()
        AgentSkills.install(agentId: "pi", root: root)
        XCTAssertTrue(AgentSkills.isOutdated(root: root), "the flat built-ins are not installed yet")
        _ = AgentSkills.refresh(root: root, stamp: "a")
        XCTAssertFalse(AgentSkills.isOutdated(root: root))

        try "edited".write(toFile: root + "/skills/mlx-serve/SKILL.md", atomically: true, encoding: .utf8)
        let backup = try XCTUnwrap(AgentSkills.refresh(root: root, stamp: "b"))
        XCTAssertFalse(backup.hasPrefix(root + "/skills/"))
        XCTAssertEqual(try String(contentsOfFile: backup + "/mlx-serve/SKILL.md", encoding: .utf8), "edited")
        XCTAssertFalse(AgentSkills.isOutdated(root: root))
    }

    func testFolderSkillIsIndexedWithItsPathAndLoadsOnInvoke() throws {
        let dir = try tempDir()
        try FileManager.default.createDirectory(atPath: dir + "/game", withIntermediateDirectories: true)
        try "---\nname: game\ndescription: wire games up\n---\nBODY-game".write(
            toFile: dir + "/game/SKILL.md", atomically: true, encoding: .utf8)
        let mgr = SkillManager(skillsDir: dir)

        let plain = mgr.matchingSkills(for: "hook my game up")
        XCTAssertTrue(plain.contains("game (wire games up)"), plain)
        XCTAssertTrue(plain.contains(dir + "/game/SKILL.md"), plain)
        XCTAssertFalse(plain.contains("BODY-game"))
        XCTAssertTrue(mgr.matchingSkills(for: "/game go").contains("BODY-game"))
    }

    func testReadFileReachesSkillFilesAndNothingElseOutside() async throws {
        let workspace = try tempDir()
        let skills = try tempDir()
        try "SKILL-BODY".write(toFile: skills + "/SKILL.md", atomically: true, encoding: .utf8)
        let gate = FileToolSandboxGate(sandboxEnabled: { false }, pinnedWorkspace: { (nil, nil) }, ensureMounted: { _ in })
        let handler = ReadFileHandler(gate: gate, readableRoots: [skills])

        let out = try await handler.execute(parameters: ["path": skills + "/SKILL.md"], workingDirectory: workspace)
        XCTAssertTrue(out.contains("SKILL-BODY"))
        do {
            _ = try await handler.execute(parameters: ["path": skills + "/../x"], workingDirectory: workspace)
            XCTFail("a path escaping the skills dir must stay confined")
        } catch {
            XCTAssertTrue("\(error)".contains("outside the workspace"), "\(error)")
        }
    }
}
