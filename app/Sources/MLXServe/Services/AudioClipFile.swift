import Foundation

/// The on-disk shape of an attached audio clip: a canonical 16 kHz mono
/// 16-bit PCM WAV, half the size of the float32 samples the model is sent and
/// playable by anything. The round trip through 16-bit is taken ONCE, on send,
/// so what the file holds is exactly what the model was handed.
enum AudioClipFile {
    static let sampleRate = 16_000

    /// Float32-LE 16 kHz mono samples to WAV bytes.
    static func encode(pcm: Data) -> Data {
        AudioReference.wavData(fromMonoFloat: AudioReference.floatSamples(from: pcm), sampleRate: sampleRate)
    }

    /// WAV bytes back to float32-LE samples. Only our own shape is read (PCM,
    /// mono, 16-bit, 16 kHz); anything else answers nil rather than samples
    /// that would be a guess handed to the model.
    static func decode(wav: Data) -> Data? {
        // Offsets below are relative to the first byte, whatever the slice.
        let wav = wav.startIndex == 0 ? wav : Data(wav)
        guard wav.count >= 44, tag(wav, at: 0) == "RIFF", tag(wav, at: 8) == "WAVE" else { return nil }
        var fmt: (format: UInt16, channels: UInt16, rate: UInt32, bits: UInt16)?
        var pcm16: Data?
        var offset = 12
        while offset + 8 <= wav.count {
            let id = tag(wav, at: offset)
            let size = Int(u32(wav, at: offset + 4))
            let body = offset + 8
            guard body + size <= wav.count else { return nil }
            switch id {
            case "fmt ":
                guard size >= 16 else { return nil }
                fmt = (u16(wav, at: body), u16(wav, at: body + 2), u32(wav, at: body + 4), u16(wav, at: body + 14))
            case "data":
                pcm16 = wav.subdata(in: body..<(body + size))
            default:
                break
            }
            offset = body + size + (size & 1)
        }
        guard let fmt, let pcm16,
              fmt.format == 1, fmt.channels == 1, fmt.bits == 16, fmt.rate == UInt32(sampleRate) else { return nil }
        // One pass and one append: a per-sample `Data.append` is the freeze
        // the writer beside this already replaced.
        let floats: [Float] = pcm16.withUnsafeBytes { raw in
            (0..<(raw.count / 2)).map { i in
                Float(Int16(bitPattern: UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self)))) / 32767.0
            }
        }
        return floats.withUnsafeBufferPointer { Data(buffer: $0) }
    }

    /// The clip as it is stored and, from then on, as it is sent: the WAV is
    /// written under `root`, and the samples are re-read from that encoding so
    /// the first turn and a regenerate after a relaunch hand the model the same
    /// bytes. A failed write keeps the samples and no path, like a picture.
    static func stored(_ clip: ChatAudio, root: String = AttachmentStore.root) -> ChatAudio {
        var result = clip
        let wav = encode(pcm: clip.pcm)
        result.pcm = decode(wav: wav) ?? clip.pcm
        result.path = AttachmentStore.write(wav, named: AttachmentStore.filename(id: clip.id, name: clip.name, ext: "wav"), in: root)
        return result
    }

    private static func tag(_ d: Data, at i: Int) -> String {
        String(decoding: d[i..<(i + 4)], as: UTF8.self)
    }
    private static func u16(_ d: Data, at i: Int) -> UInt16 {
        d.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: i, as: UInt16.self)) }
    }
    private static func u32(_ d: Data, at i: Int) -> UInt32 {
        d.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: i, as: UInt32.self)) }
    }
}
