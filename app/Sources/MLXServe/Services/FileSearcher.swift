import Foundation

// MARK: - Glob

/// Simple glob matching supporting `*` (within a path component), `**` (across
/// components) and `?`. Extracted from `ListFilesHandler` so `FileSearcher`
/// can share it; semantics are unchanged.
enum Glob {
    static func matches(_ path: String, pattern: String) -> Bool {
        var regex = "^"
        var i = pattern.startIndex
        while i < pattern.endIndex {
            let c = pattern[i]
            if c == "*" {
                let next = pattern.index(after: i)
                if next < pattern.endIndex && pattern[next] == "*" {
                    regex += ".*"
                    i = pattern.index(after: next)
                    // Skip trailing slash after **
                    if i < pattern.endIndex && pattern[i] == "/" {
                        i = pattern.index(after: i)
                    }
                    continue
                } else {
                    regex += "[^/]*"
                }
            } else if c == "?" {
                regex += "[^/]"
            } else if c == "." {
                regex += "\\."
            } else {
                regex += String(c)
            }
            i = pattern.index(after: i)
        }
        regex += "$"
        return path.range(of: regex, options: .regularExpression) != nil
    }
}

// MARK: - FileSearcher

/// In-process recursive content search — the `searchFiles` agent tool.
///
/// Replaces a `rg`/`grep` shell-out that (a) only found ripgrep at two
/// hardcoded Homebrew paths, (b) silently dropped the `context` parameter on
/// the `grep` fallback, (c) descended into `.git`, (d) emitted grep's
/// `Binary file … matches` notice, and (e) could not run at all inside the App
/// Sandbox or the Agent Sandbox guest, where neither binary is reachable.
///
/// Output shape mirrors `rg -n --no-heading`: `path:line:text` for a match,
/// `path-line-text` for a context line.
enum FileSearcher {

    /// Directories never descended into. Mirrors ripgrep's defaults (hidden
    /// entries are skipped wholesale) plus the usual heavy build trees, which
    /// gitignore would exclude for `rg` but `grep -r` happily walked.
    static let prunedDirectories: Set<String> = [
        "node_modules", "__pycache__", "zig-cache", "zig-out",
        ".build", "dist", "build", "target", "venv",
    ]

    /// Bytes inspected when classifying a file as binary. A NUL in this window
    /// means binary — the same heuristic grep and ripgrep use.
    static let binarySniffBytes = 8192

    /// Files larger than this are skipped rather than read into memory.
    static let maxFileBytes = 16 * 1024 * 1024

    struct Options {
        /// Absolute directory or file to search. Callers confine this first.
        var root: String
        var pattern: String
        /// Glob filter. A pattern with no `/` matches the filename at any depth.
        var include: String?
        var contextLines: Int = 0
        var maxResults: Int = 100
    }

    struct Line: Equatable {
        let path: String
        let number: Int
        let text: String
        let isMatch: Bool
    }

    /// A pattern that fails to compile as a regex is searched for literally.
    /// The old shell-out passed the pattern straight to `rg`/`grep`, so an
    /// agent writing `foo(bar)` got a regex-error instead of its match.
    private static func makeMatcher(_ pattern: String) -> (String) -> Bool {
        if let re = try? NSRegularExpression(pattern: pattern) {
            return { line in
                re.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) != nil
            }
        }
        return { $0.contains(pattern) }
    }

    static func search(_ options: Options) -> [Line] {
        guard options.maxResults > 0 else { return [] }
        let matcher = makeMatcher(options.pattern)
        let context = max(0, min(options.contextLines, 10))
        var out: [Line] = []

        for file in candidateFiles(root: options.root, include: options.include) {
            if out.count >= options.maxResults { break }
            guard let lines = readTextLines(file) else { continue }

            // Collect matching line indices, then expand by the context window.
            // Overlapping windows merge; `emitted` keeps each line printed once.
            var emitted = Set<Int>()
            for (idx, text) in lines.enumerated() where matcher(text) {
                let lo = max(0, idx - context)
                let hi = min(lines.count - 1, idx + context)
                for j in lo...hi where !emitted.contains(j) {
                    emitted.insert(j)
                    out.append(Line(path: file, number: j + 1, text: lines[j], isMatch: j == idx))
                    if out.count >= options.maxResults { break }
                }
                if out.count >= options.maxResults { break }
            }
        }
        // A context line emitted before its match was seen keeps `isMatch`
        // false; sort restores file/line order after any such backfill.
        return out.sorted { ($0.path, $0.number) < ($1.path, $1.number) }
    }

    /// `rg -n --no-heading` shape: `:` before a match line, `-` before context.
    static func render(_ lines: [Line], pattern: String) -> String {
        guard !lines.isEmpty else { return "No matches found for '\(pattern)'" }
        return lines.map { l in
            let sep = l.isMatch ? ":" : "-"
            return "\(l.path)\(sep)\(l.number)\(sep)\(l.text)"
        }.joined(separator: "\n")
    }

    // MARK: - Walk

    /// Depth-first, alphabetically sorted (so `maxResults` truncation is
    /// deterministic), never following symlinks, never entering hidden or
    /// pruned directories.
    static func candidateFiles(root: String, include: String?) -> [String] {
        let fm = FileManager.default
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: root, isDirectory: &isDir) else { return [] }
        if !isDir.boolValue {
            return matchesInclude(root, include) ? [root] : []
        }

        var files: [String] = []
        var stack = [root]
        while let dir = stack.popLast() {
            let entries = (try? fm.contentsOfDirectory(atPath: dir))?.sorted() ?? []
            // Forward pass so FILES are visited alphabetically — truncation at
            // `maxResults` happens in visit order, so a reversed collection
            // here would keep zzz.txt and drop aaa.txt.
            var subdirs: [String] = []
            for name in entries {
                let full = (dir as NSString).appendingPathComponent(name)
                let url = URL(fileURLWithPath: full)
                let rv = try? url.resourceValues(forKeys: [.isSymbolicLinkKey, .isDirectoryKey])
                if rv?.isSymbolicLink == true { continue }
                if rv?.isDirectory == true {
                    if name.hasPrefix(".") || prunedDirectories.contains(name) { continue }
                    subdirs.append(full)
                } else {
                    if name.hasPrefix(".") { continue }
                    if matchesInclude(full, include) { files.append(full) }
                }
            }
            // Reversed: popLast() then yields the alphabetically-first subdir.
            stack.append(contentsOf: subdirs.reversed())
        }
        return files
    }

    private static func matchesInclude(_ path: String, _ include: String?) -> Bool {
        guard let include, !include.isEmpty else { return true }
        // Gitignore semantics: a pattern without a separator matches the
        // basename at any depth (`-g '*.swift'` finds `src/a.swift`).
        let target = include.contains("/") ? path : (path as NSString).lastPathComponent
        return Glob.matches(target, pattern: include)
    }

    /// nil when the file is binary, unreadable, or over `maxFileBytes`.
    static func readTextLines(_ path: String) -> [String]? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? Int, size <= maxFileBytes else { return nil }
        guard let data = FileManager.default.contents(atPath: path) else { return nil }
        if data.prefix(binarySniffBytes).contains(0) { return nil }
        guard let text = String(data: data, encoding: .utf8)
                ?? String(data: data, encoding: .isoLatin1) else { return nil }
        // `split(omittingEmptySubsequences: false)` keeps blank lines, so line
        // numbers line up with what an editor shows.
        var lines = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        if lines.last == "" { lines.removeLast() } // trailing newline isn't a line
        return lines
    }
}
