import { describe, expect, it } from "vitest";
import { muxMp4, type AudioTrack, type EncodedSample, type VideoTrack } from "../src/lib/mp4/mp4";

type Box = { type: string; start: number; size: number; body: number; end: number };

function boxes(bytes: Uint8Array, start = 0, end = bytes.length): Box[] {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out: Box[] = [];
  for (let at = start; at < end; ) {
    const size = view.getUint32(at);
    expect(size).toBeGreaterThanOrEqual(8);
    expect(at + size).toBeLessThanOrEqual(end);
    out.push({ type: String.fromCharCode(...bytes.subarray(at + 4, at + 8)), start: at, size, body: at + 8, end: at + size });
    at += size;
  }
  return out;
}
const find = (bytes: Uint8Array, within: Box | undefined, ...path: string[]): Box => {
  let found = within;
  for (const type of path) {
    found = (found ? boxes(bytes, found.body, found.end) : boxes(bytes)).find((b) => b.type === type);
    if (!found) throw new Error(`missing ${type} in ${path.join("/")}`);
  }
  return found!;
};
const all = (bytes: Uint8Array, within: Box, type: string) => boxes(bytes, within.body, within.end).filter((b) => b.type === type);
const dv = (bytes: Uint8Array) => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
const ascii = (bytes: Uint8Array, at: number, n = 4) => String.fromCharCode(...bytes.subarray(at, at + n));

async function mux(video: VideoTrack, audio?: AudioTrack) {
  const bytes = new Uint8Array(await muxMp4(video, audio).arrayBuffer());
  return { bytes, view: dv(bytes) };
}

// avcC with one SPS (4 bytes) and one PPS (2 bytes), 4-byte NAL lengths.
const avcC = Uint8Array.of(1, 0x42, 0, 0x1f, 0xff, 0xe1, 0, 4, 0x67, 0x42, 0, 0x1f, 1, 0, 2, 0x68, 0xce);
const frame = (n: number, key: boolean, fill: number): EncodedSample => ({
  data: new Uint8Array(n).fill(fill),
  timestamp: 0,
  duration: 41_667,
  key,
});
function video(count = 6): VideoTrack {
  const chunks = Array.from({ length: count }, (_, i) => ({
    ...frame(10 + i, i % 3 === 0, 0x10 + i),
    timestamp: Math.round((i * 1e6) / 24),
    duration: Math.round(((i + 1) * 1e6) / 24) - Math.round((i * 1e6) / 24),
  }));
  return { width: 64, height: 48, fps: 24, description: avcC, chunks };
}
const aacConfig = Uint8Array.of(0x11, 0x90); // AAC-LC, 48 kHz, stereo
function aac(count = 12): AudioTrack {
  return {
    codec: "aac",
    sampleRate: 48_000,
    channels: 2,
    description: aacConfig,
    chunks: Array.from({ length: count }, (_, i) => ({
      data: new Uint8Array(5 + i).fill(0x80 + i),
      timestamp: Math.round((i * 1024 * 1e6) / 48_000),
      duration: Math.round((1024 * 1e6) / 48_000),
      key: true,
    })),
  };
}
const opusHead = Uint8Array.of(...new TextEncoder().encode("OpusHead"), 1, 2, 0x38, 0x01, 0x80, 0xbb, 0, 0, 0, 0, 0);
function opus(count = 10): AudioTrack {
  return {
    codec: "opus",
    sampleRate: 48_000,
    channels: 2,
    description: opusHead,
    chunks: Array.from({ length: count }, (_, i) => ({ data: new Uint8Array(7).fill(0x40 + i), timestamp: i * 20_000, duration: 20_000, key: true })),
  };
}

/** Sample bytes of a track read back through stsc/stsz/stco: they must equal the muxed chunks. */
function sampleBytes(bytes: Uint8Array, trak: Box): Uint8Array[] {
  const stbl = find(bytes, trak, "mdia", "minf", "stbl");
  const v = dv(bytes);
  const stsz = find(bytes, stbl, "stsz"), stco = find(bytes, stbl, "stco"), stsc = find(bytes, stbl, "stsc");
  expect(v.getUint32(stsc.body + 4)).toBe(1); // one entry: one sample per chunk
  expect(v.getUint32(stsc.body + 8)).toBe(1);
  expect(v.getUint32(stsc.body + 12)).toBe(1);
  const count = v.getUint32(stsz.body + 8);
  expect(v.getUint32(stco.body + 4)).toBe(count);
  return Array.from({ length: count }, (_, i) => {
    const size = v.getUint32(stsz.body + 12 + 4 * i), at = v.getUint32(stco.body + 8 + 4 * i);
    return bytes.slice(at, at + size);
  });
}
function stts(bytes: Uint8Array, trak: Box): number[] {
  const v = dv(bytes), box = find(bytes, trak, "mdia", "minf", "stbl", "stts");
  const n = v.getUint32(box.body + 4), out: number[] = [];
  for (let i = 0; i < n; i++) for (let k = 0; k < v.getUint32(box.body + 8 + 8 * i); k++) out.push(v.getUint32(box.body + 12 + 8 * i));
  return out;
}

describe("muxMp4", () => {
  it("writes ftyp, moov, mdat in order with a progressive (non-fragmented) layout", async () => {
    const { bytes } = await mux(video());
    expect(boxes(bytes).map((b) => b.type)).toEqual(["ftyp", "moov", "mdat"]);
    expect(ascii(bytes, 8)).toBe("isom");
    const moov = find(bytes, undefined, "moov");
    expect(boxes(bytes, moov.body, moov.end).map((b) => b.type)).toEqual(["mvhd", "trak"]);
  });

  it("describes the video track: size, timescale, avcC verbatim, stts, stss and exact sample bytes", async () => {
    const v = video(), { bytes, view } = await mux(v);
    const moov = find(bytes, undefined, "moov"), trak = find(bytes, moov, "trak");
    const tkhd = find(bytes, trak, "tkhd");
    expect(view.getUint32(tkhd.body + 12)).toBe(1); // track_ID
    expect(view.getUint32(tkhd.body + 76)).toBe(64 << 16);
    expect(view.getUint32(tkhd.body + 80)).toBe(48 << 16);
    const mdhd = find(bytes, trak, "mdia", "mdhd");
    expect(view.getUint32(mdhd.body + 12)).toBe(90_000);
    expect(ascii(bytes, find(bytes, trak, "mdia", "hdlr").body + 8)).toBe("vide");
    const stsd = find(bytes, trak, "mdia", "minf", "stbl", "stsd");
    const avc1 = boxes(bytes, stsd.body + 8, stsd.end)[0]!;
    expect(avc1.type).toBe("avc1");
    expect(view.getUint16(avc1.body + 24)).toBe(64);
    expect(view.getUint16(avc1.body + 26)).toBe(48);
    const config = boxes(bytes, avc1.body + 78, avc1.end)[0]!;
    expect(config.type).toBe("avcC");
    expect(bytes.slice(config.body, config.end)).toEqual(avcC);
    const deltas = stts(bytes, trak);
    expect(deltas.length).toBe(6);
    expect(deltas.reduce((a, b) => a + b, 0)).toBe(Math.round((6 * 90_000) / 24)); // cumulative rounding: no drift
    const stss = find(bytes, trak, "mdia", "minf", "stbl", "stss");
    expect(view.getUint32(stss.body + 4)).toBe(2);
    expect([view.getUint32(stss.body + 8), view.getUint32(stss.body + 12)]).toEqual([1, 4]);
    expect(sampleBytes(bytes, trak)).toEqual(v.chunks.map((c) => c.data));
    expect(view.getUint32(find(bytes, moov, "mvhd").body + 12)).toBe(1000);
    expect(view.getUint32(find(bytes, moov, "mvhd").body + 16)).toBe(250);
  });

  it("omits stss when every frame is a sync sample", async () => {
    const v = video(3);
    for (const c of v.chunks) c.key = true;
    const { bytes } = await mux(v);
    const stbl = find(bytes, find(bytes, find(bytes, undefined, "moov"), "trak"), "mdia", "minf", "stbl");
    expect(boxes(bytes, stbl.body, stbl.end).map((b) => b.type)).not.toContain("stss");
  });

  it("adds an AAC track whose esds carries the encoder's AudioSpecificConfig", async () => {
    const a = aac(), { bytes, view } = await mux(video(), a);
    const moov = find(bytes, undefined, "moov"), traks = all(bytes, moov, "trak");
    expect(traks.length).toBe(2);
    const trak = traks[1]!;
    expect(view.getUint32(find(bytes, trak, "tkhd").body + 12)).toBe(2);
    expect(ascii(bytes, find(bytes, trak, "mdia", "hdlr").body + 8)).toBe("soun");
    expect(view.getUint32(find(bytes, trak, "mdia", "mdhd").body + 12)).toBe(48_000);
    const stsd = find(bytes, trak, "mdia", "minf", "stbl", "stsd");
    const mp4a = boxes(bytes, stsd.body + 8, stsd.end)[0]!;
    expect(mp4a.type).toBe("mp4a");
    expect(view.getUint16(mp4a.body + 16)).toBe(2); // channelcount
    expect(view.getUint32(mp4a.body + 24)).toBe(48_000 * 65536);
    const esds = boxes(bytes, mp4a.body + 28, mp4a.end)[0]!;
    expect(esds.type).toBe("esds");
    const payload = bytes.slice(esds.body + 4, esds.end);
    expect(payload[0]).toBe(0x03);
    const at = payload.findIndex((b, i) => b === 0x05 && payload[i + 1] === 2);
    expect([...payload.subarray(at + 2, at + 4)]).toEqual([...aacConfig]);
    expect(sampleBytes(bytes, trak)).toEqual(a.chunks.map((c) => c.data));
    expect(stts(bytes, trak).every((d) => d === 1024)).toBe(true);
  });

  it("derives the AudioSpecificConfig when the encoder gave none, and refuses unsupported output", async () => {
    const a = aac(2);
    delete a.description;
    const { bytes } = await mux(video(), a);
    const traks = all(bytes, find(bytes, undefined, "moov"), "trak");
    const stsd = find(bytes, traks[1]!, "mdia", "minf", "stbl", "stsd");
    const entry = boxes(bytes, stsd.body + 8, stsd.end)[0]!;
    const esds = boxes(bytes, entry.body + 28, entry.end)[0]!;
    const payload = bytes.slice(esds.body + 4, esds.end);
    const at = payload.findIndex((b, i) => b === 0x05 && payload[i + 1] === 2);
    expect([...payload.subarray(at + 2, at + 4)]).toEqual([...aacConfig]); // AAC-LC, 48 kHz, 2 channels
    expect(() => muxMp4(video(), { ...aac(2), sampleRate: 12_345 })).toThrow(/unsupported/i);
    expect(() => muxMp4(video(), { ...aac(0) })).toThrow(/unsupported/i);
  });

  it("interleaves video and audio samples by timestamp in mdat", async () => {
    const { bytes } = await mux(video(), aac());
    const moov = find(bytes, undefined, "moov"), mdat = find(bytes, undefined, "mdat");
    const offsets: [number, string][] = [];
    for (const [i, trak] of all(bytes, moov, "trak").entries()) {
      const stco = find(bytes, trak, "mdia", "minf", "stbl", "stco"), n = dv(bytes).getUint32(stco.body + 4);
      for (let k = 0; k < n; k++) offsets.push([dv(bytes).getUint32(stco.body + 8 + 4 * k), i ? "a" : "v"]);
    }
    offsets.sort((x, y) => x[0] - y[0]);
    expect(offsets[0]![0]).toBe(mdat.body);
    expect(offsets.map((o) => o[1]).join("")).not.toMatch(/^v+a+$/); // not grouped per track
  });

  it("writes Opus with dOps, a pre-skip edit list and a roll sample group", async () => {
    const o = opus(), { bytes, view } = await mux(video(), o);
    const trak = all(bytes, find(bytes, undefined, "moov"), "trak")[1]!;
    const stbl = find(bytes, trak, "mdia", "minf", "stbl");
    const stsd = find(bytes, stbl, "stsd");
    const entry = boxes(bytes, stsd.body + 8, stsd.end)[0]!;
    expect(entry.type).toBe("Opus");
    const dops = boxes(bytes, entry.body + 28, entry.end)[0]!;
    expect(dops.type).toBe("dOps");
    expect(bytes[dops.body]).toBe(0); // version
    expect(bytes[dops.body + 1]).toBe(2); // output channels
    expect(view.getUint16(dops.body + 2)).toBe(0x0138); // pre-skip (OpusHead is little-endian)
    expect(view.getUint32(dops.body + 4)).toBe(48_000);
    expect(bytes[dops.body + 10]).toBe(0); // channel mapping family
    const elst = find(bytes, trak, "edts", "elst");
    expect(view.getUint32(elst.body + 4)).toBe(1);
    expect(view.getUint32(elst.body + 8)).toBe(Math.round(200 - 0x138 / 48)); // 10 packets minus the pre-skip, in ms
    expect(view.getInt32(elst.body + 12)).toBe(0x0138); // media_time = pre-skip
    expect(boxes(bytes, stbl.body, stbl.end).map((b) => b.type)).toEqual(expect.arrayContaining(["sgpd", "sbgp"]));
    expect(sampleBytes(bytes, trak)).toEqual(o.chunks.map((c) => c.data));
  });

  it("refuses an Opus track it cannot describe", () => {
    expect(() => muxMp4(video(), { ...opus(), description: undefined })).toThrow(/Opus/);
    expect(() => muxMp4(video(), { ...opus(), channels: 6 })).toThrow(/Opus/);
    const family1 = Uint8Array.from(opusHead);
    family1[18] = 1;
    expect(() => muxMp4(video(), { ...opus(), description: family1 })).toThrow(/Opus/);
  });

  it("refuses empty video and a malformed avcC", () => {
    expect(() => muxMp4({ ...video(), chunks: [] })).toThrow(/no video frames/i);
    expect(() => muxMp4({ ...video(), description: Uint8Array.of(1, 0x42, 0, 0x1f, 0xfc, 0xe1) })).toThrow(/AVC/);
    expect(() => muxMp4({ ...video(), description: Uint8Array.of(2, 0x42, 0, 0x1f, 0xff, 0xe1, 0, 0) })).toThrow(/AVC/);
  });
});
