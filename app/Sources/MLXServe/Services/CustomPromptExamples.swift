import Foundation

/// User-provided prompt examples loaded from `~/.mlx-serve/examples/`.
/// Drop a `.txt` file into the appropriate subdirectory — each file becomes
/// a named group in the Examples menu, and each non-empty line becomes a
/// separate clickable prompt (auto-titled from its first few words).
///
/// Directory layout:
///   ~/.mlx-serve/examples/
///     video/          → VideoGenView's Examples menu
///     image/          → ImageGenView's Examples menu (text-to-image mode)
///     image-edit/     → ImageGenView's Examples menu (edit mode)
///
/// File format (one prompt per line):
///   ~/.mlx-serve/examples/image/PartiPrompts.txt:
///     A portrait of a kangaroo wearing a purple hoodie
///     A blue jay standing on a basket of rainbow macarons
///     A corgi wearing a pirate hat and eyepatch
struct CustomPromptExamples {
    let title: String
    let body: String

    /// Load every `.txt` file from `~/.mlx-serve/examples/<subdirectory>/`.
    /// Each file becomes its own group (filename minus extension = group name).
    /// Each non-empty line becomes a prompt with an auto-generated title from
    /// its first few words. Returns empty when the directory doesn't exist
    /// or contains no valid prompt files.
    static func load(for category: String) -> [ImagePromptExampleGroup] {
        let dir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".mlx-serve/examples/\(category)")

        guard let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        else { return [] }

        return files
            .filter { $0.pathExtension.lowercased() == "txt" }
            .compactMap { file -> ImagePromptExampleGroup? in
                guard let content = try? String(contentsOf: file, encoding: .utf8),
                      !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                else { return nil }

                let prompts = content
                    .components(separatedBy: .newlines)
                    .map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty }
                    .map { line -> ImagePromptExample in
                        let title = Self.autoTitle(from: line)
                        return ImagePromptExample(title: title, body: line)
                    }

                guard !prompts.isEmpty else { return nil }

                let groupName = file.deletingPathExtension().lastPathComponent
                return ImagePromptExampleGroup(name: groupName, examples: prompts)
            }
            .sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    /// Generate a short title from the first few words of a prompt line.
    /// Caps at ~40 characters, adds ellipsis if truncated.
    private static func autoTitle(from line: String) -> String {
        let words = line.split(separator: " ")
        let preview = words.prefix(6).joined(separator: " ")
        if preview.count > 40 {
            return String(preview.prefix(40)).trimmingCharacters(in: .whitespaces) + "…"
        }
        if words.count > 6 {
            return preview + "…"
        }
        return preview
    }
}
