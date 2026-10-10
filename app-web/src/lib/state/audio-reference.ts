import { t } from "../i18n/i18n";
/**
 * Decode supported browser audio and produce the WAV shape used by Swift:
 * speech 24k mono; music 48k stereo. No microphone access during upload.
 */
async function audioReference(file: Blob, name: string, kind: 'voice' | 'music', maxSeconds: number = 30) {
  if (file.size > 100 * 1024 * 1024)
    throw Error(t("Audio file must be at most 100 MB."));
  const rate = kind === "voice" ? 24000 : 48000,
    channels = kind === "voice" ? 1 : 2;
  const decoder = new OfflineAudioContext(1, 1, rate);
  let decoded;
  try {
    decoded = await decoder.decodeAudioData(await file.arrayBuffer());
  } catch {
    throw Error(
      t(
        "This browser could not decode the audio file. Try WAV, MP3 or another supported format.",
      ),
    );
  }
  if (!decoded.duration || decoded.duration > maxSeconds)
    throw Error(t("Choose a clip no longer than %@ seconds.", [maxSeconds]));
  const frames = Math.ceil(decoded.duration * rate),
    context = new OfflineAudioContext(channels, frames, rate),
    source = context.createBufferSource();
  source.buffer = decoded;
  source.connect(context.destination);
  source.start();
  const audio = await context.startRendering(),
    bytes = new Uint8Array(44 + frames * channels * 2),
    v = new DataView(bytes.buffer);
  const text = ( offset: number,  s: string) =>
    bytes.set(new TextEncoder().encode(s), offset);
  text(0, "RIFF");
  v.setUint32(4, bytes.length - 8, true);
  text(8, "WAVEfmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true);
  v.setUint16(22, channels, true);
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * channels * 2, true);
  v.setUint16(32, channels * 2, true);
  v.setUint16(34, 16, true);
  text(36, "data");
  v.setUint32(40, bytes.length - 44, true);
  for (let c = 0; c < channels; c++) {
    const data = audio.getChannelData(c);
    for (let i = 0; i < frames; i++) {
      const x = Math.max(-1, Math.min(1, data[i]));
      v.setInt16(
        44 + (i * channels + c) * 2,
        Math.round(x * (x < 0 ? 32768 : 32767)),
        true,
      );
    }
  }
  let binary = "";
  for (let i = 0; i < bytes.length; i += 8192)
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return { name, base64: btoa(binary), duration: frames / rate };
}

export { audioReference };
