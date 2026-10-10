import { describe, expect, it } from "vitest";
import { applyRewriteEvent, cleanReply, emptyReplyReason, newRewriteProgress, rewriteLabel, storyboardRewrite, videoRewrite } from "../src/lib/core/rewrite";

describe("video rewrite prompts", () => {
  it("names the H3 labels in order and caps the action at the clip length", () => {
    const r = videoRewrite("a fox", "h3Base", 10);
    expect(r.system).toContain("integrated_multimodal_description:, overall_soundscape:, non_diegetic_music:");
    expect(r.user).toContain("The clip is 10 seconds long.");
    expect(r.user).toContain("a fox");
  });

  it("LTX is prose with no labels", () => {
    expect(videoRewrite("a fox", "ltx", 4).system).toContain("ONE paragraph of 4-8 sentences");
  });

  it("the planner is told the bounds it must plan inside", () => {
    const r = storyboardRewrite("a fox's day", "h3Base", 300, [5, 15]);
    expect(r.system).toContain("between 5 and 15 seconds");
    expect(r.system).toContain("integrated_multimodal_description:");
    expect(r.user).toContain("300 seconds, about 20 shots");
    expect(r.maxTokens).toBe(1024 + 512 * 20);
  });

  it("a first frame rides the request with its rule, said only when the picture is sent", () => {
    const clip = videoRewrite("they dance", "h3Base", 10, "PNG");
    const plan = storyboardRewrite("they dance", "h3Base", 30, [5, 10], "PNG");
    for (const r of [clip, plan]) {
      expect(r.firstFrame).toBe("PNG");
      expect(r.userWith(true)).toContain("attached picture");
      expect(r.userWith(false)).toBe(r.user);
    }
    expect(plan.userWith(true)).toContain("Shot 1 opens on it");
    expect(clip.userWith(true)).not.toContain("Shot 1");
    expect(videoRewrite("x", "ltx", 4).userWith(true)).toBe(videoRewrite("x", "ltx", 4).user);
  });

  it("strips fences and quotes off a reply", () => {
    expect(cleanReply("```\n“A red fox.”\n```")).toBe("A red fox.");
  });
});

describe("rewrite progress", () => {
  it("names each wait: loading, waiting, thinking, writing", () => {
    let p = newRewriteProgress("Qwen3.8");
    expect(rewriteLabel(p)).toBe("Loading Qwen3.8…");
    p = newRewriteProgress();
    expect(rewriteLabel(p)).toBe("Waiting for the model…");
    p = applyRewriteEvent(p, { type: "reasoning", text: "The user wants " });
    p = applyRewriteEvent(p, { type: "reasoning", text: "a fox." });
    expect([rewriteLabel(p), p.thought]).toEqual(["Thinking…", "The user wants a fox."]);
    p = applyRewriteEvent(p, { type: "content", text: "\n\n" });
    expect(p.stage).toBe("thinking");
    p = applyRewriteEvent(p, { type: "content", text: "A red fox." });
    expect([rewriteLabel(p), p.text]).toEqual(["Writing…", "\n\nA red fox."]);
  });

  it("an empty reply says why, and a think cut at the budget says so", () => {
    expect(emptyReplyReason(applyRewriteEvent(newRewriteProgress(), { type: "content", text: "A fox." }))).toBe("");
    expect(emptyReplyReason(applyRewriteEvent(newRewriteProgress(), { type: "content", text: "  " }))).toBe("The model finished without writing anything. Try again.");
    let p = applyRewriteEvent(newRewriteProgress(), { type: "reasoning", text: "hmm" });
    p = applyRewriteEvent(p, { type: "finish", reason: "length" });
    expect(emptyReplyReason(p)).toBe("The model used its whole token budget thinking and wrote nothing. Try again.");
  });
});
