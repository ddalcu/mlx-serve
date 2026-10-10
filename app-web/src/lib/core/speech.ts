import { Client } from "./client";
import { audioRequest } from "./media";
import type { MediaResult } from "./media";
import type { MediaOptions } from "./media";
export type SpeechRequest = Record<string, unknown> & { model?: string; input: string; voice?: string; speed?: number; ref_audio?: string; };

const speech = (client: Client, request: SpeechRequest, options: MediaOptions = {}): Promise<MediaResult> =>
  audioRequest(
    client,
    "/v1/audio/speech",
    { ...request, response_format: "wav" },
    options,
  );

export { speech };
