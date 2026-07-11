import XCTest
@testable import MLXCore

/// CLASS GUARD — "a host binary spawn that breaks the sandbox".
///
/// Spawning a binary from `/bin`, `/usr/bin`, `/usr/sbin` or `/opt/homebrew` is
/// a defect in two independent ways:
///
///  1. App Sandbox the binary is unreachable,
///     so the feature silently dies in the container.
///  2. With the Agent Sandbox ON, it runs on the HOST — escaping the very
///     confinement the shield icon promises the user.
///
/// Phase 0 removed four of these (`rg`/`grep`, `tar`, `lsof`/`ps`, `vm_stat`)
/// in favor of `FileSearcher`, `TarReader` and `SystemMetrics`. The rest are
/// tracked below with an explicit disposition. Adding a new one — or a new
/// spawn in a file not on this list — fails this test.
///
/// When a file's disposition is discharged (gated out under `MAS_BUILD`, or
/// moved into the guest), delete its entry. The list only shrinks.
final class HostEscapeAuditTests: XCTestCase {

    /// file name → why its spawns are still here.
    private static let known: [String: String] = [
        "CLIInstaller.swift":
            "osascript + login shell to symlink into /usr/local/bin. Runtime-gated on the compile-time BuildFeatures.cliInstaller flag (false under MAS_BUILD); UI hidden with it.",
        "CLILauncher.swift":
            "login zsh to detect+launch claude/pi/opencode. Runtime-gated on the compile-time BuildFeatures.cliLauncher flag (false under MAS_BUILD); UI hidden with it.",
        "UpdateChecker.swift":
            "hdiutil/ditto to swap the installed .app. Runtime-gated on the compile-time BuildFeatures.selfUpdate flag (false under MAS_BUILD); UI hidden with it.",
        "MCPSpawner.swift":
            "HostMCPSpawner's login zsh, used only when the Agent Sandbox is OFF. Compiled out under MAS_BUILD.",
        "ProcessRegistry.swift":
            "login zsh for the agent's host shell tool. Guest-only under MAS_BUILD (Phase 4).",
    ]

    /// Matches a string literal naming an absolute path into a system bin dir.
    private static let spawnPattern = #""(/bin/|/usr/bin/|/usr/sbin/|/opt/homebrew/)"#

    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // MLXCoreTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // app
            .appendingPathComponent("Sources/MLXServe")
    }

    func testNoNewHostBinarySpawns() throws {
        let fm = FileManager.default
        let regex = try NSRegularExpression(pattern: Self.spawnPattern)

        var offenders: [String: [String]] = [:]
        let walker = try XCTUnwrap(fm.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil))
        for case let url as URL in walker where url.pathExtension == "swift" {
            let text = try String(contentsOf: url, encoding: .utf8)
            for (number, rawLine) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let line = String(rawLine)
                let trimmed = line.trimmingCharacters(in: .whitespaces)
                if trimmed.hasPrefix("//") { continue } // prose, not code
                // Indices must come from `line` itself, not from the Substring.
                let range = NSRange(line.startIndex..., in: line)
                guard regex.firstMatch(in: line, range: range) != nil else { continue }
                offenders[url.lastPathComponent, default: []].append("\(number + 1): \(trimmed)")
            }
        }

        // Guard against a vacuous pass: the walk must have seen the real tree.
        XCTAssertGreaterThan(try fileCount(), 50, "source walk found too few files")

        let unexpected = offenders.keys.filter { Self.known[$0] == nil }.sorted()
        XCTAssertTrue(unexpected.isEmpty, """
            New host-binary spawn(s) introduced in: \(unexpected.joined(separator: ", "))

            \(unexpected.flatMap { name in offenders[name]!.map { "  \(name):\($0)" } }.joined(separator: "\n"))

            A spawned /bin, /usr/bin, /usr/sbin or /opt/homebrew binary is unreachable
            inside the App Sandbox container AND escapes the Agent Sandbox onto the host.
            Use FileSearcher / TarReader / SystemMetrics, or route the command into the
            guest. If it is genuinely unavoidable, add the file to `known` with a reason.
            """)
    }

    /// Entries in `known` that no longer spawn anything must be deleted, or the
    /// list rots into a permanent excuse.
    func testKnownEscapeListHasNoStaleEntries() throws {
        let fm = FileManager.default
        let regex = try NSRegularExpression(pattern: Self.spawnPattern)

        var stale: [String] = []
        for name in Self.known.keys.sorted() {
            let url = sourcesRoot.appendingPathComponent("Services/\(name)")
            guard let text = try? String(contentsOf: url, encoding: .utf8) else {
                stale.append("\(name) (file no longer exists)")
                continue
            }
            let code = text.split(separator: "\n", omittingEmptySubsequences: false)
                .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }
                .joined(separator: "\n")
            if regex.firstMatch(in: code, range: NSRange(code.startIndex..., in: code)) == nil {
                stale.append("\(name) (no spawns left — remove it from `known`)")
            }
        }
        XCTAssertTrue(stale.isEmpty, stale.joined(separator: "\n"))
    }

    private func fileCount() throws -> Int {
        let walker = FileManager.default.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil)
        var n = 0
        for case let url as URL in walker! where url.pathExtension == "swift" { n += 1 }
        return n
    }
}
