// The Enhance dialog's model side, ported from the app's PromptRewriter.swift and
// RewriteProgress: the instructions a chat model gets to rewrite a video prompt or plan
// a storyboard, and what the dialog says while it works.
import { t } from "../i18n/i18n";
import { videoPrompts } from "../state/video-presets";
import type { ChatEvent } from "./chat";

export type PromptFormat = keyof typeof videoPrompts;
export type RewriteRequest = {
  system: string;
  user: string;
  maxTokens: number;
  /** The clip's first frame (base64), shown to a chat model that can see it. */
  firstFrame?: string;
  /** The user turn, with the picture's rule when the picture goes along. */
  userWith(seeingImage: boolean): string;
};

const SECTIONS: Record<PromptFormat, string[]> = {
  ltx: [],
  h3Base: ["integrated_multimodal_description:", "overall_soundscape:", "non_diegetic_music:"],
  h3Reference: ["subject_definitions:", "summary:", "retention_analysis:", "detailed_description:", "overall_soundscape:", "non_diegetic_music:"],
};

/** The video's frame 0 is the picture and the rest follows the text, so a prompt describing another look makes H3 and LTX cut away from it. */
const FIRST_FRAME_RULE =
  "The video starts EXACTLY on the attached picture: it is the first frame. Describe what it shows as it is (the same people, clothes, setting, lighting and camera framing) and continue the action from there. Never contradict it: a prompt that describes a different look makes the video cut away from the picture.";

const examples = (format: PromptFormat, count: number) =>
  videoPrompts[format]
    .slice(0, count)
    .map((x) => x.body)
    .join("\n\n---\n\n");

function request(system: string, user: string, maxTokens: number, firstFrame: string | undefined, rule: string): RewriteRequest {
  return { system, user, maxTokens, firstFrame, userWith: (seeing) => (seeing && firstFrame ? `${user}\n\n${rule}` : user) };
}

function videoRewrite(text: string, format: PromptFormat, seconds: number, firstFrame?: string): RewriteRequest {
  const labels = SECTIONS[format],
    shape = labels.length
      ? `Write the prompt in the exact labelled format of the examples, with these labels in order, each on its own line: ${labels.join(", ")}. Keep any <Picture N>, <Video N> or <Audio N> references verbatim.`
      : "Write ONE paragraph of 4-8 sentences like the examples: subject, action, camera movement, lighting, setting, sound. Keep spoken dialogue in double quotes.";
  return request(
    `You rewrite video prompts for a generative model. ${shape} Keep the user's intent; make it more specific and evocative. Reply with ONLY the rewritten video prompt, no preamble, no quotes, no markdown.\n\nExamples of the expected format:\n\n${examples(format, 3)}`,
    `Rewrite this video prompt:\n\n${text}\n\nThe clip is ${seconds} seconds long. Describe only what happens in that time: no more action than fits.`,
    2048,
    firstFrame,
    FIRST_FRAME_RULE,
  );
}

/** A storyboard plan: shots headed `=== SHOT n | Ns ===`. Each shot is generated alone from the last frame before it, which is why every shot restates the cast. */
function storyboardRewrite(idea: string, format: PromptFormat, totalSeconds: number, [lo, hi]: [number, number], firstFrame?: string): RewriteRequest {
  const labels = SECTIONS[format],
    shape = labels.length
      ? `the exact labelled format of the examples, with these labels in order, each on its own line: ${labels.join(", ")}.`
      : "one paragraph of 4-8 sentences: subject, action, camera movement, lighting, setting, sound. Keep spoken dialogue in double quotes.",
    shots = Math.max(1, Math.ceil(totalSeconds / hi));
  return request(
    [
      "You plan a long video as a storyboard of shots for a generative video model. Each shot is generated on its own: it starts from the last frame of the shot before it and the model sees ONLY that shot's prompt. So:",
      "- Describe every recurring character, outfit, place and visual style again in EVERY shot, in the same words.",
      "- Shots flow continuously: a shot begins exactly where the previous one ended, so move the camera or the action rather than cutting to a new scene.",
      "- Keep the sound and music descriptions the same from shot to shot unless the story changes them.",
      "- Give each shot one beat of action, no more than fits in its length.",
      `- Each shot lasts between ${lo} and ${hi} seconds. Prefer long shots: every join is a seam.`,
      `Write each shot's prompt as ${shape}`,
      `Reply with ONLY the shots. Head each one with a line exactly like \`=== SHOT 1 | ${hi}s ===\` (its number and its length in seconds), then its prompt. No preamble, no markdown.`,
      "",
      "Examples of one shot's prompt format:",
      "",
      examples(format, 2),
    ].join("\n"),
    `Story:\n\n${idea}\n\nTotal length: ${totalSeconds} seconds, about ${shots} shots.`,
    // A labelled shot runs to a few hundred tokens, and a long story is dozens of shots.
    1024 + 512 * shots,
    firstFrame,
    FIRST_FRAME_RULE + " Shot 1 opens on it, and every later shot keeps that look.",
  );
}

/** Model replies sometimes wear a fence or quotes; the editor gets the bare text. */
function cleanReply(reply: string) {
  let out = reply.trim();
  if (out.startsWith("```")) {
    const newline = out.indexOf("\n");
    out = newline < 0 ? "" : out.slice(newline + 1);
    const fence = out.lastIndexOf("```");
    if (fence >= 0) out = out.slice(0, fence);
    out = out.trim();
  }
  return out.replace(/^["“”]+|["“”]+$/g, "").trim();
}

/** What the dialog says while the chat model works: a slow load or a long think is normal, and silence reads as a hang. */
export type RewriteProgress = { stage: "loading" | "waiting" | "thinking" | "writing"; loading: string; text: string; thought: string; truncated: boolean };

/** `loading` names a model that is not resident yet: the server loads it before it reads the prompt. */
function newRewriteProgress(loading = ""): RewriteProgress {
  return { stage: loading ? "loading" : "waiting", loading, text: "", thought: "", truncated: false };
}

function applyRewriteEvent(p: RewriteProgress, e: ChatEvent): RewriteProgress {
  if (e.type === "reasoning") return { ...p, stage: "thinking", thought: p.thought + e.text };
  if (e.type === "content") {
    const text = p.text + e.text;
    // A blank lead-in is not the answer starting.
    return { ...p, text, stage: text.trim() ? "writing" : p.stage === "loading" ? "waiting" : p.stage };
  }
  if (e.type === "finish" && e.reason === "length") return { ...p, truncated: true };
  return p;
}

function rewriteLabel(p: RewriteProgress) {
  return {
    loading: () => t("Loading %@…", [p.loading]),
    waiting: () => t("Waiting for the model…"),
    thinking: () => t("Thinking…"),
    writing: () => t("Writing…"),
  }[p.stage]();
}

/** Why a finished reply left nothing to apply, or "". */
function emptyReplyReason(p: RewriteProgress) {
  if (cleanReply(p.text)) return "";
  return p.truncated && p.thought
    ? t("The model used its whole token budget thinking and wrote nothing. Try again.")
    : t("The model finished without writing anything. Try again.");
}

export { videoRewrite, storyboardRewrite, cleanReply, newRewriteProgress, applyRewriteEvent, rewriteLabel, emptyReplyReason };
