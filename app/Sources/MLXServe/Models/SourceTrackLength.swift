import Foundation

/// How long a Cover or Vocal-to-BGM track will be: the source clip decides,
/// so the pane can say it before a single step is denoised.
enum SourceTrackLength {

    /// Seconds of a PCM WAV, read from its own header. The source clip is
    /// written by `AudioReference`, so a shape that does not parse is a bug to
    /// stay quiet about rather than a number to guess at.
    static func seconds(header: Data) -> Double? {
        // A slice keeps its parent's indices; the offsets below are absolute.
        let d = header.startIndex == 0 ? header : Data(header)
        guard d.count >= 44, tag(d, 0) == "RIFF", tag(d, 8) == "WAVE" else { return nil }
        var channels: UInt16 = 0, bits: UInt16 = 0
        var rate: UInt32 = 0, dataBytes: UInt32 = 0
        var offset = 12
        while offset + 8 <= d.count {
            let id = tag(d, offset)
            let size = u32(d, offset + 4)
            let body = offset + 8
            if id == "data" { dataBytes = size; break }
            if id == "fmt " {
                guard body + 16 <= d.count else { return nil }
                channels = u16(d, body + 2)
                rate = u32(d, body + 4)
                bits = u16(d, body + 14)
            }
            offset = body + Int(size) + Int(size & 1)
        }
        let bytesPerSecond = Double(rate) * Double(channels) * Double(bits) / 8
        guard bytesPerSecond > 0, dataBytes > 0 else { return nil }
        return Double(dataBytes) / bytesPerSecond
    }

    /// Reads the header and nothing else: a ten-minute clip is around 100 MB.
    static func seconds(ofWavAt url: URL) -> Double? {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let head = try? handle.read(upToCount: 4096) else { return nil }
        return seconds(header: head)
    }

    /// The sentence the hint says. Words rather than a clock: "0:07" beside
    /// "10 seconds to 10 minutes" reads as two different units.
    static func spoken(seconds: Double) -> String {
        let total = Int(seconds.rounded())
        let minutes = total / 60, rest = total % 60
        let m = "\(minutes) minute\(minutes == 1 ? "" : "s")"
        let s = "\(rest) second\(rest == 1 ? "" : "s")"
        if minutes == 0 { return s }
        if rest == 0 { return m }
        return "\(m) \(s)"
    }

    private static func tag(_ d: Data, _ i: Int) -> String {
        String(decoding: d[i..<(i + 4)], as: UTF8.self)
    }
    private static func u16(_ d: Data, _ i: Int) -> UInt16 {
        d.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: i, as: UInt16.self)) }
    }
    private static func u32(_ d: Data, _ i: Int) -> UInt32 {
        d.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: i, as: UInt32.self)) }
    }
}
