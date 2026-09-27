import Foundation

/// The mlx-serve agent skill: one folder under `~/.mlx-serve/skills`, linked
/// into each launched agent's own skills dir. Twin of Zig `launch.installSkill`.
enum AgentSkills {
    static let name = "mlx-serve"
    static let defaultRoot = NSString(string: "~/.mlx-serve").expandingTildeInPath
    /// Claude Code has no skills dir we own; `--plugin-dir` loads this plugin.
    static let claudePluginDir = "claude/plugin"
    static let claudePluginManifest = "{\"name\": \"mlx-serve\", \"description\": \"Skills for the local mlx-serve server\"}\n"

    /// The app bundle's copy (app/build.sh), else the repo's `skills/` in dev and tests.
    static func sourceDir() -> URL? {
        let repo = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("skills/\(name)")
        return [Bundle.main.resourceURL?.appendingPathComponent("agent-skills/\(name)"), repo]
            .compactMap { $0 }
            .first { FileManager.default.fileExists(atPath: $0.path) }
    }

    static func files() -> [(name: String, content: String)] {
        guard let src = sourceDir(),
              let names = try? FileManager.default.contentsOfDirectory(atPath: src.path) else { return [] }
        return names.filter { $0.hasSuffix(".md") }.sorted().compactMap { n in
            (try? String(contentsOf: src.appendingPathComponent(n), encoding: .utf8)).map { (n, $0) }
        }
    }

    /// Where an agent discovers skills in its dedicated config dir; opencode
    /// reads `skills.paths` from its inline config instead.
    static func linkPath(agentId: String) -> String? {
        switch agentId {
        case "pi", "omp", "codex", "hermes": return "\(agentId)/skills/\(name)"
        case "claude": return "\(claudePluginDir)/skills/\(name)"
        default: return nil
        }
    }

    /// Install the skill wherever it is missing and link it into the agent's
    /// dir. Never overwrites: edits stick until the user picks "Update".
    static func install(agentId: String, root: String = defaultRoot) {
        let fm = FileManager.default
        let skill = "\(root)/skills/\(name)"
        try? fm.createDirectory(atPath: skill, withIntermediateDirectories: true)
        for f in files() where !fm.fileExists(atPath: "\(skill)/\(f.name)") {
            try? f.content.write(toFile: "\(skill)/\(f.name)", atomically: true, encoding: .utf8)
        }
        guard let link = linkPath(agentId: agentId) else { return }
        if agentId == "claude" {
            let dir = "\(root)/\(claudePluginDir)/.claude-plugin"
            if !fm.fileExists(atPath: "\(dir)/plugin.json") {
                try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
                try? claudePluginManifest.write(toFile: "\(dir)/plugin.json", atomically: true, encoding: .utf8)
            }
        }
        let linkAbs = "\(root)/\(link)"
        try? fm.createDirectory(atPath: (linkAbs as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
        try? fm.createSymbolicLink(atPath: linkAbs, withDestinationPath: skill)
    }

    /// Every skill file the app ships, relative to `<root>/skills`.
    static func shipped() -> [(path: String, content: String)] {
        files().map { ("\(name)/\($0.name)", $0.content) }
            + SkillManager.builtinSkills.map { ($0.file, $0.body) }
    }

    static func isOutdated(root: String = defaultRoot) -> Bool {
        shipped().contains { (try? String(contentsOfFile: "\(root)/skills/\($0.path)", encoding: .utf8)) != $0.content }
    }

    /// Restore every shipped skill file that differs. Edited copies go to
    /// `<root>/skills-backup-<stamp>`, outside the skills dir so no agent picks
    /// them up as skills. Returns the backup dir, nil when nothing was edited.
    @discardableResult
    static func refresh(root: String = defaultRoot, stamp: String) -> String? {
        let fm = FileManager.default
        let backup = "\(root)/skills-backup-\(stamp)"
        var backedUp = false
        for f in shipped() {
            let path = "\(root)/skills/\(f.path)"
            let old = try? String(contentsOfFile: path, encoding: .utf8)
            guard old != f.content else { continue }
            if let old {
                let dest = "\(backup)/\(f.path)"
                try? fm.createDirectory(atPath: (dest as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
                try? old.write(toFile: dest, atomically: true, encoding: .utf8)
                backedUp = true
            }
            try? fm.createDirectory(atPath: (path as NSString).deletingLastPathComponent, withIntermediateDirectories: true)
            try? f.content.write(toFile: path, atomically: true, encoding: .utf8)
        }
        return backedUp ? backup : nil
    }
}
