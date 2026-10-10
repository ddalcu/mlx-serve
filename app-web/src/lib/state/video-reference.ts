import { t } from "../i18n/i18n";
import { audioReference } from "./audio-reference";
/**
 * Extract H3 24fps JPEG conditioning frames, capped before allocating the sequence.
 * Audio is explicitly included or declared absent by the user; a decode failure never silently drops it.
 */
async function videoReference(
  file: File,
  maxFrames: number,
  includeAudio: boolean = true,
  signal?: AbortSignal,
) {
  if (file.size > 100 * 1024 * 1024)
    throw Error(t("Reference clips must be at most 100 MB."));
  const video = document.createElement("video"),
    url = URL.createObjectURL(file);
  video.muted = true;
  video.preload = "auto";
  video.src = url;
  const wait = (event: string, action: () => void) =>
    new Promise((resolve, reject) => {
      const clean = () => {
        clearTimeout(timer);
        video.removeEventListener(event, done);
        video.removeEventListener("error", fail);
        signal?.removeEventListener("abort", fail);
      };
      const done = () => {
          clean();
          resolve(undefined);
        },
        fail = () => {
          clean();
          reject(
            Error(
              signal?.aborted
                ? t("Reference conversion cancelled.")
                : t(
                    "This browser could not decode the reference clip. Try MP4/H.264.",
                  ),
            ),
          );
        };
      const timer = setTimeout(fail, 10000);
      video.addEventListener(event, done, { once: true });
      video.addEventListener("error", fail, { once: true });
      signal?.addEventListener("abort", fail, { once: true });
      if (signal?.aborted) fail();
      else action();
    });
  try {
    await wait("loadeddata", () => video.load());
    if (!Number.isFinite(video.duration) || video.duration <= 0)
      throw Error(t("Invalid reference duration."));
    let count = Math.min(Math.floor(video.duration * 24), maxFrames);
    count -= (((count - 5) % 17) + 17) % 17;
    if (count < 5)
      throw Error(t("Reference clips need at least 5 frames at 24 fps."));
    const canvas = document.createElement("canvas"),
      scale = Math.min(1, 1024 / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) throw Error(t("Video conversion unavailable."));
    const frames = [];
    let bytes = 0;
    for (let i = 0; i < count; i++) {
      if (signal?.aborted) throw Error(t("Reference conversion cancelled."));
      if (i)
        await wait("seeked", () => {
          video.currentTime = i / 24;
        });
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const b64 = canvas.toDataURL("image/jpeg", 0.9).split(",")[1];
      bytes += b64.length;
      if (bytes > 40 * 1024 * 1024)
        throw Error(
          t(
            "Reference frames exceed 40 MiB. Choose a smaller or shorter clip.",
          ),
        );
      frames.push(b64);
    }
    let audio;
    if (includeAudio) {
      try {
        audio = (await audioReference(file, file.name, "music", 30)).base64;
      } catch {
        throw Error(
          t(
            "Could not decode the clip soundtrack. If it is silent, select “Clip has no soundtrack”, otherwise use a supported MP4 or upload the audio separately.",
          ),
        );
      }
    }
    return { name: file.name, frames, ...(audio ? { audio } : {}) };
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}

export { videoReference };
