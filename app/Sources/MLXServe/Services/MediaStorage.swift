import Foundation

/// Output roots for natively-generated media. All three modalities (image,
/// audio, video) are produced by the embedded `mlx-serve` engine — there is no
/// Python venv anymore; this is just where the results are written.
enum MediaStorage {
    static let imagesRoot: String = make("images")
    static let videosRoot: String = make("videos")
    static let audiosRoot: String = make("audio")
    static let musicRoot: String = make("music")
    static let models3dRoot: String = make("models3d")
    /// Upscale/restore output. The folder is named for the FEATURE the user
    /// pressed ("Upscale") and the endpoint it comes from
    /// (`/v1/images/upscales`), not for the model's internal word for the job.
    ///
    /// RENAMING AN OUTPUT FOLDER LOSES THE OUTPUTS unless something moves
    /// them: the history shelf lists what it finds under this path, so files
    /// written to the old `restored/` would simply stop existing as far as the
    /// app is concerned. `adoptLegacy` moves them across, once.
    static let upscalesRoot: String = adoptLegacy("restored", into: make("upscales"))

    /// Move a previous release's output folder into `dest` and remove it.
    /// Per-file, skipping anything already present, so a half-finished move is
    /// resumable and can never overwrite a newer file with an older one.
    private static func adoptLegacy(_ old: String, into dest: String) -> String {
        let fm = FileManager.default
        let legacy = NSString(string: "~/.mlx-serve/generations/\(old)").expandingTildeInPath
        var isDir: ObjCBool = false
        guard fm.fileExists(atPath: legacy, isDirectory: &isDir), isDir.boolValue else { return dest }
        for entry in (try? fm.contentsOfDirectory(atPath: legacy)) ?? [] {
            let from = (legacy as NSString).appendingPathComponent(entry)
            let to = (dest as NSString).appendingPathComponent(entry)
            if fm.fileExists(atPath: to) { continue }
            try? fm.moveItem(atPath: from, toPath: to)
        }
        // Only removes an EMPTY directory, so anything the move could not
        // take is left where it is rather than deleted.
        try? fm.removeItem(atPath: (legacy as NSString).appendingPathComponent(".DS_Store"))
        if ((try? fm.contentsOfDirectory(atPath: legacy)) ?? [""]).isEmpty {
            try? fm.removeItem(atPath: legacy)
        }
        return dest
    }

    private static func make(_ name: String) -> String {
        let dir = NSString(string: "~/.mlx-serve/generations/\(name)").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }
}
