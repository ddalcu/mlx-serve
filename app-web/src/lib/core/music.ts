import { t } from "../i18n/i18n";
import { Client, StudioError } from "./client";
import { audioRequest } from "./media";
import type { MediaResult } from "./media";
import type { MediaOptions } from "./media";
export type MusicRequest = Record<string, unknown> & { model?: string; prompt: string; lyrics?: string; duration_seconds?: number; };

function buildMusicRequest(request: MusicRequest) {
  if (
    /minimax.*music.?3|music3/i.test(request.model ?? "") &&
    !request.instrumental &&
    !request.lyrics?.trim()
  )
    throw new StudioError("protocol", t("MiniMax Music 3 requires lyrics."));
  const body = { ...request, response_format: "wav" };
  if (request.instrumental) delete body.lyrics;
  return body;
}
const music = (client: Client, request: MusicRequest, options: MediaOptions = {}): Promise<MediaResult> =>
  audioRequest(
    client,
    "/v1/audio/music-generations",
    buildMusicRequest(request),
    options,
  );

export { buildMusicRequest, music };
