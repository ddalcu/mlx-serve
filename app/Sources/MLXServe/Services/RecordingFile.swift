import Foundation

/// A recording written to disk as it arrives: 16 kHz mono, 16-bit PCM with no
/// header, so an hour costs no memory and a quit or a crash loses nothing.
final class RecordingFile: @unchecked Sendable {
    /// Quieter than this (RMS) is silence: room noise sits well below speech.
    static let audible: Float = 0.005

    let url: URL
    private let handle: FileHandle
    private let lock = NSLock()
    private var heard = Date()

    init(url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else { throw CocoaError(.fileWriteUnknown) }
        handle = try FileHandle(forWritingTo: url)
        self.url = url
    }

    /// The last moment a write held something audible (the file's creation before that).
    var lastHeard: Date { lock.withLock { heard } }

    /// Append `samples`, or write them from sample `offset` on: a gap before it reads back as silence.
    func write(_ samples: UnsafeBufferPointer<Float>, at offset: Int? = nil) {
        var energy: Float = 0
        let pcm = samples.map { x -> Int16 in
            energy += x * x
            return Int16(max(-1, min(1, x)) * 32767)
        }
        let audible = !samples.isEmpty && (energy / Float(samples.count)).squareRoot() > Self.audible
        lock.withLock {
            if let offset { try? handle.seek(toOffset: UInt64(offset * 2)) }
            pcm.withUnsafeBytes { try? handle.write(contentsOf: Data($0)) }
            if audible { heard = Date() }
        }
    }

    func close() {
        lock.withLock { try? handle.close() }
    }

    /// The samples of a recording on disk; none when it is missing.
    static func read(_ url: URL) -> [Float] {
        guard let data = try? Data(contentsOf: url) else { return [] }
        return data.withUnsafeBytes { $0.bindMemory(to: Int16.self).map { Float($0) / 32767 } }
    }
}
