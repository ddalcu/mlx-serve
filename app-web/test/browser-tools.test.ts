import { describe, expect, it } from "vitest";
import type { Client } from "../src/lib/core/client";
import type { Library } from "../src/lib/core/library";
import { parseModels, type Model } from "../src/lib/core/models";
import { toolLoop } from "../src/lib/core/tool-loop";
import type { Session } from "../src/lib/state/chat-state.svelte";
import { browserTools } from "../src/lib/state/chat-tools";
import { imageProfile } from "../src/lib/state/image-state.svelte";
import { pageServer } from "../src/lib/core/console";

const model = (id: string, capabilities: string[], architecture: string) =>
  ({ id, capabilities, architecture, meta: { architecture } }) as Model;
const models = [
  model("flux", ["image"], "flux2"),
  model("mage-base", ["image"], "mage_flow"),
  model("kokoro", ["audio"], "kokoro"),
  model("ace", ["audio", "music"], "acestep"),
  model("qwen-tts", ["audio"], "qwen3_tts"),
];
const noLibrary = {} as Library;
const client = (parts: Record<string, unknown>) => ({ baseUrl: "https://host", ...parts }) as unknown as Client;
const abort = () => new AbortController().signal;
const names = (tools: { function: { name: string } }[]) => tools.map((t) => t.function.name);
const enumOf = (tools: { function: { name: string; parameters: any } }[], name: string): string[] =>
  tools.find((t) => t.function.name === name)!.function.parameters.properties.model.enum;
const asTools = (v: unknown) => v as { function: { name: string; parameters: any } }[];

describe("browser tool resolution", () => {
  it("capabilities separate speech, music and edit-capable image models", () => {
    const parsed = parseModels({ data: models });
    expect(parsed.filter((m) => m.capabilities.includes("speech")).map((m) => m.id)).toEqual(["kokoro", "qwen-tts"]);
    const { tools } = browserTools(client({}), models, noLibrary, { server: "https://host", messages: [] } as unknown as Session);
    expect(enumOf(asTools(tools), "generate_speech")).toEqual(["kokoro", "qwen-tts"]);
    expect(enumOf(asTools(tools), "generate_music")).toEqual(["ace"]);
    const edit = enumOf(asTools(tools), "edit_image");
    expect(edit).toEqual(models.filter((m) => imageProfile(m)?.edit || m.capabilities.includes("image-edit")).map((m) => m.id));
    expect(edit).not.toContain("mage-base");
  });

  it("edit execution accepts exactly the advertised edit-model enum", async () => {
    const opened: [string, RequestInit][] = [];
    const c = client({
      open: async (path: string, init: RequestInit) => {
        opened.push([path, init]);
        throw Error("fixture request");
      },
    });
    const session = { server: "https://host", messages: [{ role: "user", images: ["data:image/png;base64,YQ=="] }] } as unknown as Session;
    const { tools, execute } = browserTools(c, models, noLibrary, session);
    const allowed = enumOf(asTools(tools), "edit_image");
    for (const m of models) {
      const count = opened.length;
      await expect(execute({ name: "edit_image", args: { model: m.id, prompt: "blue" } } as never, abort())).rejects.toThrow(
        allowed.includes(m.id) ? /fixture request/ : /advertised model/,
      );
      expect(opened.length - count).toBe(allowed.includes(m.id) ? 1 : 0);
    }
    expect(opened[0]![0]).toBe("/v1/images/edits");
    expect((opened[0]![1].body as FormData).get("model")).toBe("flux");
  });

  it("honors a named model and rejects a hallucinated model before HTTP", async () => {
    const sent: any[] = [];
    const c = client({
      open: async (_path: string, init: RequestInit) => {
        sent.push(JSON.parse(init.body as string));
        throw Error("fixture request");
      },
    });
    const { execute } = browserTools(c, models, noLibrary);
    const run = (model: string) => execute({ name: "generate_image", args: { model, prompt: "fox" } } as never, abort());
    await expect(run("flux")).rejects.toThrow(/fixture request/);
    expect(sent[0].model).toBe("flux");
    await expect(run("dall-e-3")).rejects.toThrow(/advertised model/);
    expect(sent.length).toBe(1);
  });

  it("refuses a modality absent from the server before HTTP", async () => {
    let requests = 0;
    const { tools, execute } = browserTools(client({ open: () => void requests++ }), [], noLibrary);
    expect(names(asTools(tools))).toEqual(["search_library"]);
    await expect(execute({ name: "generate_music", args: { prompt: "lofi" } } as never, abort())).rejects.toThrow(/music/);
    expect(requests).toBe(0);
  });

  it("prefers resident, then healthy complete models without reordering discovery", async () => {
    const fleet = [
      { ...models[2]!, id: "broken", state: "error", bytes_on_disk: 100 },
      { ...models[2]!, id: "incomplete", bytes_on_disk: null },
      { ...models[2]!, id: "cold", bytes_on_disk: 100 },
      { ...models[2]!, id: "resident", loaded: true, bytes_on_disk: 100 },
    ];
    for (const [available, expected] of [
      [fleet, "resident"],
      [fleet.slice(0, 3), "cold"],
      [fleet.slice(1, 2), "incomplete"],
    ] as const) {
      let body: any;
      const c = client({
        bytes: async (_path: string, init: RequestInit) => {
          body = JSON.parse(init.body as string);
          throw Error("fixture request");
        },
      });
      const { tools, execute } = browserTools(c, parseModels({ data: available }), noLibrary);
      const speech = asTools(tools).find((t) => t.function.name === "generate_speech")!;
      expect(speech.function.parameters.required).not.toContain("model");
      await expect(execute({ name: "generate_speech", args: { prompt: "hello" } } as never, abort())).rejects.toThrow(/fixture request/);
      expect(body.model).toBe(expected);
    }
    expect(fleet.map((m) => m.id)).toEqual(["broken", "incomplete", "cold", "resident"]);
  });

  it("the system prompt carries the mounted base, inventory, API fields and question rules onto the wire", async () => {
    const baseUrl = pageServer({ origin: "https://inference.example:8443", pathname: "/proxy/mlx/index.html" });
    const options = browserTools(client({ baseUrl }), models, noLibrary);
    const request = {
      messages: [
        { role: "system", content: "User custom instruction." },
        { role: "user", content: "Show me curl for edits." },
      ],
    };
    let sent: { content: string }[] = [];
    const stream = async function* (_client: unknown, req: { messages: { content: string }[] }) {
      sent = req.messages;
      yield { type: "finish", reason: "stop" };
    };
    for await (const _ of toolLoop({} as Client, request as never, { ...options, stream })) void _;
    const prompt = sent[0]!.content;
    expect(prompt).toMatch(/https:\/\/inference.example:8443\/proxy\/mlx/);
    for (const m of models) expect(prompt).toContain(m.id);
    expect(prompt).toMatch(/image.*flux2|flux2.*image/);
    expect(prompt).toMatch(/POST \/v1\/images\/edits/);
    expect(prompt).toMatch(/multipart\/form-data/);
    expect(prompt).toMatch(/image\[\]/);
    expect(prompt).toMatch(/mask/);
    expect(prompt).toMatch(/stream:true/);
    expect(prompt).toMatch(/questions.*text.*no tool/is);
    expect(prompt).toMatch(/one media generation/i);
    expect(sent[1]!.content).toBe("User custom instruction.");
    expect(request.messages.length).toBe(2);
  });
});
