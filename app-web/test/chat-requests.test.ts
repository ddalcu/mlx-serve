import { describe, expect, it } from "vitest";
import { buildChatRequest } from "../src/lib/core/chat";
import { buildEditRequest } from "../src/lib/core/images";

describe("request builders", () => {
  it("chat keeps parameters and always streams with usage", () => {
    expect(buildChatRequest({ model: "chat", messages: [], temperature: 0, stream: false })).toEqual({
      model: "chat",
      messages: [],
      temperature: 0,
      stream: true,
      stream_options: { include_usage: true },
    });
  });

  it("edit keeps repeated image order and drops undefined fields", async () => {
    const form = buildEditRequest({ model: "flux", prompt: "blue", seed: 0, unused: undefined }, [
      new Blob(["first"], { type: "image/png" }),
      new Blob(["second"], { type: "image/jpeg" }),
    ]);
    expect(form.get("stream")).toBe("false");
    expect(form.get("seed")).toBe("0");
    expect(form.get("response_format")).toBe("b64_json");
    expect(form.has("unused")).toBe(false);
    expect(await Promise.all(form.getAll("image[]").map((b) => (b as Blob).text()))).toEqual(["first", "second"]);
    expect(() => buildEditRequest({ prompt: "x" }, [])).toThrow(/reference image/);
  });
});
