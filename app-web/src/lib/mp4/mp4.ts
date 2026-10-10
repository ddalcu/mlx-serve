import { StudioError } from "../core/client";
import { t } from "../i18n/i18n";

export type EncodedSample = { data: Uint8Array<ArrayBuffer>; timestamp: number; duration: number; key: boolean };
export type VideoTrack = { width: number; height: number; fps: number; description: Uint8Array; chunks: EncodedSample[] };
export type AudioTrack = {
  codec: "aac" | "opus";
  sampleRate: number;
  channels: number;
  description?: Uint8Array;
  chunks: EncodedSample[];
  durationSeconds?: number;
};

type Part = Uint8Array;
const VIDEO_SCALE = 90_000;
const MOVIE_SCALE = 1000;
const AAC_RATES = [96000, 88200, 64000, 48000, 44100, 32000, 24000, 22050, 16000, 12000, 11025, 8000, 7350];

function concat(parts: Part[]): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
}
const be = (width: 1 | 2 | 4, ...values: number[]): Uint8Array<ArrayBuffer> => {
  const out = new Uint8Array(values.length * width), view = new DataView(out.buffer);
  values.forEach((v, i) => (width === 4 ? view.setUint32(i * 4, v >>> 0) : width === 2 ? view.setUint16(i * 2, v & 0xffff) : view.setUint8(i, v)));
  return out;
};
const zeros = (n: number): Uint8Array<ArrayBuffer> => new Uint8Array(n);
const fourcc = (s: string): Uint8Array<ArrayBuffer> => Uint8Array.from(s, (c) => c.charCodeAt(0));
const box = (type: string, ...body: Part[]): Uint8Array<ArrayBuffer> => {
  const content = concat(body), out = new Uint8Array(8 + content.length);
  new DataView(out.buffer).setUint32(0, out.length);
  out.set(fourcc(type), 4);
  out.set(content, 8);
  return out;
};
const full = (type: string, version: number, flags: number, ...body: Part[]) => box(type, be(4, (version << 24) | flags), ...body);
const UNITY = be(4, 0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000);

/** MPEG-4 descriptor with the expandable length encoding. */
function descriptor(tag: number, ...body: Part[]): Part {
  const content = concat(body), size: number[] = [content.length & 0x7f];
  for (let n = content.length >> 7; n; n >>= 7) size.unshift((n & 0x7f) | 0x80);
  return concat([be(1, tag), Uint8Array.from(size), content]);
}

function checkAvcC(bytes: Part) {
  const bad = () => new StudioError("protocol", t("Invalid AVC decoder configuration."));
  if (bytes.length < 7 || bytes[0] !== 1 || (bytes[4]! & 3) !== 3) throw bad();
  let at = 5;
  for (let group = 0; group < 2; group++) {
    let count = group ? bytes[at++]! : bytes[at++]! & 31;
    while (count--) {
      if (at + 2 > bytes.length) throw bad();
      const size = bytes[at]! * 256 + bytes[at + 1]!;
      at += 2 + size;
      if (!size || at > bytes.length) throw bad();
    }
    if (group === 0 && at >= bytes.length) throw bad();
  }
}

/** AudioSpecificConfig for AAC-LC when the encoder did not supply one. */
function audioSpecificConfig(rate: number, channels: number): Part {
  const index = AAC_RATES.indexOf(rate);
  return be(1, (2 << 3) | (index >> 1), ((index & 1) << 7) | (channels << 3));
}

function audioEntry(audio: AudioTrack): { entry: Part; preSkip: number; scale: number } {
  const head = (type: string, rate: number, config: Part) =>
    box(type, zeros(6), be(2, 1), zeros(8), be(2, audio.channels, 16, 0, 0), be(4, rate * 65536), config);
  if (audio.codec === "aac") {
    if (!AAC_RATES.includes(audio.sampleRate) || !audio.chunks.length)
      throw new StudioError("unsupported", t("AAC sample rate or output is unsupported."));
    const asc = audio.description?.length ? audio.description : audioSpecificConfig(audio.sampleRate, audio.channels);
    const esds = full(
      "esds",
      0,
      0,
      descriptor(3, be(2, 2), be(1, 0), descriptor(4, be(1, 0x40, 0x15), zeros(11), descriptor(5, asc)), descriptor(6, be(1, 2))),
    );
    return { entry: head("mp4a", audio.sampleRate, esds), preSkip: 0, scale: audio.sampleRate };
  }
  // https://opus-codec.org/docs/opus_in_isobmff.html (mapping family 0, mono/stereo).
  const header = audio.description;
  if (!header || header.length < 19 || String.fromCharCode(...header.subarray(0, 8)) !== "OpusHead" || header[18] !== 0 || audio.channels > 2 || !audio.chunks.length)
    throw new StudioError("unsupported", t("Unsupported Opus channel mapping."));
  const h = new DataView(header.buffer, header.byteOffset, header.byteLength);
  const preSkip = h.getUint16(10, true);
  const dOps = box("dOps", be(1, 0, audio.channels), be(2, preSkip), be(4, h.getUint32(12, true)), be(2, h.getInt16(16, true)), be(1, 0));
  return { entry: head("Opus", 48_000, dOps), preSkip, scale: 48_000 };
}

type Track = {
  id: number;
  handler: "vide" | "soun";
  scale: number;
  entry: Part;
  chunks: EncodedSample[];
  /** Declared presentation duration. */
  seconds?: number;
  preSkip?: number;
  width?: number;
  height?: number;
};

/** Decode-time deltas in track ticks: each sample starts where its timestamp says, the last one lasts its duration. */
function deltas(track: Track): number[] {
  const base = track.chunks[0]!.timestamp, tick = (us: number) => Math.round(((us - base) * track.scale) / 1e6);
  return track.chunks.map((c, i) => {
    const next = track.chunks[i + 1];
    return Math.max(1, (next ? tick(next.timestamp) : tick(c.timestamp + c.duration)) - tick(c.timestamp));
  });
}

/** What plays: the declared duration, else the samples minus the Opus pre-skip. */
const presentationSeconds = (track: Track) =>
  track.seconds ?? (deltas(track).reduce((a, b) => a + b, 0) - (track.preSkip ?? 0)) / track.scale;

function trak(track: Track, offsets: number[]): Uint8Array<ArrayBuffer> {
  const d = deltas(track), media = d.reduce((a, b) => a + b, 0);
  const movie = Math.round(presentationSeconds(track) * MOVIE_SCALE);
  const runs: number[] = [];
  for (let i = 0; i < d.length; ) {
    let n = 1;
    while (i + n < d.length && d[i + n] === d[i]) n++;
    runs.push(n, d[i]!);
    i += n;
  }
  const sync = track.chunks.flatMap((c, i) => (c.key ? [i + 1] : []));
  const audio = track.handler === "soun";
  const stbl = box(
    "stbl",
    full("stsd", 0, 0, be(4, 1), track.entry),
    full("stts", 0, 0, be(4, runs.length / 2, ...runs)),
    ...(sync.length === track.chunks.length || audio ? [] : [full("stss", 0, 0, be(4, sync.length, ...sync))]),
    full("stsc", 0, 0, be(4, 1, 1, 1, 1)),
    full("stsz", 0, 0, be(4, 0, track.chunks.length, ...track.chunks.map((c) => c.data.byteLength))),
    full("stco", 0, 0, be(4, offsets.length, ...offsets)),
    ...(track.preSkip === undefined
      ? []
      : [
          full("sbgp", 0, 0, fourcc("roll"), be(4, 1, track.chunks.length, 1)),
          full("sgpd", 1, 0, fourcc("roll"), be(4, 2, 1), be(2, -4)),
        ]),
  );
  const edits =
    track.preSkip === undefined ? [] : [box("edts", full("elst", 0, 0, be(4, 1, movie, track.preSkip), be(2, 1, 0)))];
  return box(
    "trak",
    full("tkhd", 0, 3, be(4, 0, 0, track.id, 0, movie), zeros(8), be(2, 0, 0, audio ? 0x100 : 0, 0), UNITY, be(4, (track.width ?? 0) * 65536, (track.height ?? 0) * 65536)),
    ...edits,
    box(
      "mdia",
      full("mdhd", 0, 0, be(4, 0, 0, track.scale, media), be(2, 0x55c4, 0)),
      full("hdlr", 0, 0, be(4, 0), fourcc(track.handler), zeros(12), fourcc(audio ? "SoundHandler" : "VideoHandler"), be(1, 0)),
      box(
        "minf",
        audio ? full("smhd", 0, 0, zeros(4)) : full("vmhd", 0, 1, zeros(8)),
        box("dinf", full("dref", 0, 0, be(4, 1), full("url ", 0, 1))),
        stbl,
      ),
    ),
  );
}

/** Progressive MP4 (moov before mdat): H.264 plus optional AAC or Opus, samples interleaved by timestamp. */
function muxMp4(video: VideoTrack, audio?: AudioTrack): Blob {
  if (!video.chunks.length) throw new StudioError("protocol", t("Encoder returned no video frames."));
  checkAvcC(video.description);
  const avc1 = box(
    "avc1",
    zeros(6),
    be(2, 1),
    zeros(16),
    be(2, video.width, video.height),
    be(4, 0x480000, 0x480000, 0),
    be(2, 1),
    zeros(32),
    be(2, 0x18, 0xffff),
    box("avcC", video.description),
  );
  const tracks: Track[] = [{ id: 1, handler: "vide", scale: VIDEO_SCALE, entry: avc1, chunks: video.chunks, width: video.width, height: video.height }];
  if (audio) {
    const { entry, preSkip, scale } = audioEntry(audio);
    tracks.push({ id: 2, handler: "soun", scale, entry, chunks: audio.chunks, seconds: audio.durationSeconds, preSkip: audio.codec === "opus" ? preSkip : undefined });
  }
  const start = (index: number) => tracks[index]!.chunks[0]!.timestamp;
  const order = tracks
    .flatMap((track, index) => track.chunks.map((c) => ({ c, index })))
    .sort((a, b) => a.c.timestamp - start(a.index) - (b.c.timestamp - start(b.index)) || a.index - b.index);
  const total = order.reduce((n, { c }) => n + c.data.byteLength, 0);
  const ftyp = box("ftyp", fourcc("isom"), be(4, 512), fourcc("isom"), fourcc("iso2"), fourcc("avc1"), fourcc("mp41"));
  const moov = (mdat: number) => {
    const offsets = tracks.map(() => [] as number[]);
    let at = mdat;
    for (const { c, index } of order) {
      offsets[index]!.push(at);
      at += c.data.byteLength;
    }
    const seconds = Math.max(...tracks.map(presentationSeconds));
    return box(
      "moov",
      full("mvhd", 0, 0, be(4, 0, 0, MOVIE_SCALE, Math.round(seconds * MOVIE_SCALE), 0x10000), be(2, 0x100, 0), zeros(8), UNITY, zeros(24), be(4, tracks.length + 1)),
      ...tracks.map((track, i) => trak(track, offsets[i]!)),
    );
  };
  const mdat = ftyp.length + moov(0).length + 8;
  if (mdat + total > 0xffffffff) throw new StudioError("protocol", t("Video is too large to package."));
  return new Blob([ftyp, moov(mdat), be(4, total + 8), fourcc("mdat"), ...order.map(({ c }) => c.data)], { type: "video/mp4" });
}

export { muxMp4 };
