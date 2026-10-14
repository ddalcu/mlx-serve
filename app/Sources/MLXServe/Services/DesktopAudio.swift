import AVFoundation
import CoreAudio

/// Everything the Mac plays, through a Core Audio process tap, written as
/// 16 kHz mono like `AudioRecorder`. macOS asks for System Audio Recording
/// the first time it starts.
final class DesktopAudio {
    private let queue = DispatchQueue(label: "mlx-serve.desktop-audio")
    private var tap = AudioObjectID(kAudioObjectUnknown)
    private var device = AudioObjectID(kAudioObjectUnknown)
    private var proc: AudioDeviceIOProcID?
    /// Where the next samples land, read and cleared on `queue`.
    private var offset: Int?

    /// Write into `file` from sample `offset` of the recording; silence fills the gap.
    func start(at offset: Int, into file: RecordingFile) throws {
        stop()
        self.offset = offset
        do {
            let description = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
            description.uuid = UUID()
            description.isPrivate = true
            try check(AudioHardwareCreateProcessTap(description, &tap))
            var format = AudioStreamBasicDescription()
            var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
            var address = AudioObjectPropertyAddress(mSelector: kAudioTapPropertyFormat, mScope: kAudioObjectPropertyScopeGlobal,
                                                     mElement: kAudioObjectPropertyElementMain)
            try check(AudioObjectGetPropertyData(tap, &address, 0, nil, &size, &format))
            let composition: [String: Any] = [
                kAudioAggregateDeviceNameKey: "MLX-Serve transcript",
                kAudioAggregateDeviceUIDKey: UUID().uuidString,
                kAudioAggregateDeviceIsPrivateKey: true,
                kAudioAggregateDeviceTapListKey: [[kAudioSubTapUIDKey: description.uuid.uuidString]],
            ]
            try check(AudioHardwareCreateAggregateDevice(composition as CFDictionary, &device))
            guard let input = AVAudioFormat(streamDescription: &format),
                  let converter = AVAudioConverter(from: input, to: AudioRecorder.targetFormat) else { throw CocoaError(.featureUnsupported) }
            // Both channels into the mono track, not the left one alone.
            converter.downmix = true
            try check(AudioDeviceCreateIOProcIDWithBlock(&proc, device, queue) { [weak self] _, data, _, _, _ in
                guard let self, let buffer = AVAudioPCMBuffer(pcmFormat: input, bufferListNoCopy: data, deallocator: nil),
                      let out = AudioRecorder.resample(buffer, with: converter), let ch = out.floatChannelData else { return }
                file.write(UnsafeBufferPointer(start: ch[0], count: Int(out.frameLength)), at: self.offset)
                self.offset = nil
            })
            try check(AudioDeviceStart(device, proc))
        } catch {
            stop()
            throw error
        }
    }

    func stop() {
        if let proc {
            AudioDeviceStop(device, proc)
            AudioDeviceDestroyIOProcID(device, proc)
        }
        if device != kAudioObjectUnknown { AudioHardwareDestroyAggregateDevice(device) }
        if tap != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tap) }
        (proc, device, tap) = (nil, AudioObjectID(kAudioObjectUnknown), AudioObjectID(kAudioObjectUnknown))
    }

    private func check(_ status: OSStatus) throws {
        if status != noErr { throw NSError(domain: NSOSStatusErrorDomain, code: Int(status)) }
    }
}
