import { describe, expect, it } from "vitest";
import type { Client } from "../src/lib/core/client";
import { toolLoop, type ToolRound } from "../src/lib/core/tool-loop";

describe("toolLoop", () => {
  it("allows one generation attempt per turn, including later rounds and failures; library still works", async () => {
    for (const fail of [false, true]) {
      let round = 0;
      const executed: string[] = [];
      const results: ToolRound[] = [];
      const tools = ["generate_image", "generate_speech", "search_library"].map((name) => ({ type: "function", function: { name } }));
      const stream = async function* () {
        const names = round++ === 0 ? ["generate_image", "generate_speech", "search_library"] : round === 2 ? ["generate_image"] : [];
        if (names.length)
          yield {
            type: "tools",
            delta: names.map((name, index) => ({ index, id: `r${round}-${index}`, function: { name, arguments: "{}" } })),
          };
        yield { type: "finish", reason: names.length ? "tool_calls" : "stop" };
      };
      const loop = toolLoop({ redact: (s: string) => s } as unknown as Client, { messages: [] } as never, {
        tools,
        stream,
        execute: async (call) => {
          executed.push(call.name);
          if (fail && call.name === "generate_image") throw Error("failed");
          return { text: "ok" };
        },
      });
      for await (const e of loop) if (e.type === "tool-result") results.push(structuredClone((e as { round: ToolRound }).round));
      expect(executed).toEqual(["generate_image", "search_library"]);
      expect(results.some((r) => r.calls.some((c) => c.status === "error" && /one media generation/i.test(c.result ?? "")))).toBe(true);
    }
  });
});
