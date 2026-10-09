import { flushSync } from "svelte";
import { afterEach, describe, expect, it, vi } from "vitest";
import { mockApi } from "./support/mock-api";
import { mountApp } from "./support/mount";

let ui: Awaited<ReturnType<typeof mountApp>> | undefined;
afterEach(() => ui?.unmount());

const text = (el: Element | null | undefined) => el?.textContent?.replace(/\s+/g, " ").trim() ?? "";

async function ask(question: string) {
  await ui!.type("#chat-input", question);
  await ui!.click("#chat-send");
}
const done = () => vi.waitFor(() => expect(ui!.app.chat.c.run).toBeUndefined());

describe("Chat pane", () => {
  it("starts empty with a model chosen from the server and send disabled until there is text", async () => {
    ui = await mountApp();
    expect(text(ui.q("#chat-model-name"))).toBe("chat");
    expect(ui.q("#chat-welcome")!.hidden).toBe(false);
    expect(ui.q<HTMLButtonElement>("#chat-send")!.disabled).toBe(true);
    await ui.type("#chat-input", "hello");
    expect(ui.q<HTMLButtonElement>("#chat-send")!.disabled).toBe(false);
  });

  it("sends, streams and renders the reply as markdown; the request carries the model and messages", async () => {
    const api = mockApi();
    api.replies.push({ reasoning: "pondering", content: "# Title\n\nSome **bold** text\n\n```js\nlet a = 1;\n```" });
    ui = await mountApp({ api });
    await ask("What is up?");
    await done();
    flushSync();
    const request = api.requests.find((r) => r.path === "/v1/chat/completions")!.body;
    expect(request.model).toBe("m/chat");
    expect(request.stream).toBe(true);
    expect(request.messages.at(-1)).toEqual({ role: "user", content: "What is up?" });
    const rows = ui.qa(".message");
    expect(rows.length).toBe(2);
    expect(text(rows[0]!.querySelector(".message-text"))).toBe("What is up?");
    const reply = rows[1]!;
    expect(reply.querySelector("h1")?.textContent).toBe("Title");
    expect(reply.querySelector("strong")?.textContent).toBe("bold");
    expect(reply.querySelector("pre code")?.textContent).toBe("let a = 1;\n");
    expect(text(reply.querySelector(".thinking-text"))).toBe("pondering");
    expect(text(reply.querySelector(".token-rate"))).toBe("20 tok/sec");
    expect(ui.q("#chat-welcome")!.hidden).toBe(true);
    expect(text(ui.q("#pane-title"))).toBe("What is up?");
    expect(text(ui.q(".session-entry .nav-row"))).toBe("What is up?");
  });

  it("shows model text as text: markup in a reply never becomes an element", async () => {
    const api = mockApi();
    api.replies.push({ content: '<img src=x onerror="window.pwned=1"> <script>window.pwned=2</script> [x](javascript:window.pwned=3)' });
    ui = await mountApp({ api });
    await ask("go");
    await done();
    flushSync();
    const body = ui.qa(".message")[1]!;
    expect(body.querySelector("img,script,iframe,a")).toBeNull();
    expect(text(body)).toContain("<script>window.pwned=2</script>");
    expect((window as unknown as { pwned?: number }).pwned).toBeUndefined();
  });

  it("shows a server error under the reply and keeps the conversation", async () => {
    const api = mockApi();
    api.replies.push({ status: 500, error: "<b>boom</b>" });
    ui = await mountApp({ api });
    await ask("fail please");
    await done();
    flushSync();
    expect(text(ui.q(".message.assistant .error-message"))).toContain("<b>boom</b>");
    expect(ui.q(".message.assistant .error-message b")).toBeNull();
    expect(ui.qa(".message").length).toBe(2);
  });

  it("Stop ends a running reply and marks it stopped", async () => {
    const api = mockApi();
    let release!: () => void;
    api.replies.push({ content: "partial reply", hold: new Promise<void>((r) => (release = r)) });
    ui = await mountApp({ api });
    await ask("long one");
    await vi.waitFor(() => expect(text(ui!.qa(".message.assistant .message-text").at(-1))).toContain("partial"));
    expect(ui.q("#chat-send")!.getAttribute("aria-label")).toBe("Stop generation");
    await ui.click("#chat-send");
    release();
    await done();
    flushSync();
    expect(text(ui.q(".message.assistant footer"))).toContain("Stopped");
    expect(ui.q("#chat-send")!.getAttribute("aria-label")).toBe("Send message");
  });

  it("edits a sent message and regenerates from it; deleting a turn removes both messages", async () => {
    const api = mockApi();
    api.replies.push({ content: "first answer" }, { content: "second answer" });
    ui = await mountApp({ api });
    await ask("original");
    await done();
    flushSync();
    await ui.click(ui.q(".message.user button[aria-label='Edit & Resend']")!);
    const editor = ui.q<HTMLTextAreaElement>(".message-editor")!;
    editor.value = "changed";
    editor.dispatchEvent(new Event("input", { bubbles: true }));
    flushSync();
    await ui.click(ui.qa<HTMLButtonElement>(".edit-actions button").find((b) => b.textContent === "Save")!);
    await done();
    flushSync();
    expect(ui.qa(".message").map((m) => text(m.querySelector(".message-text")))).toEqual(["changed", "second answer"]);
    await ui.click(ui.q(".message.assistant button[aria-label='Delete Turn']")!);
    expect(ui.qa(".message").length).toBe(0);
    expect(ui.q("#chat-welcome")!.hidden).toBe(false);
  });

  it("a new chat adds a session; picking another session shows its messages", async () => {
    const api = mockApi();
    api.replies.push({ content: "answer one" });
    ui = await mountApp({ api });
    await ask("question one");
    await done();
    flushSync();
    await ui.click("button[aria-label='New Chat']");
    expect(ui.qa(".session-entry").length).toBe(2);
    expect(ui.qa(".message").length).toBe(0);
    await ui.click(ui.qa(".session-entry .nav-row")[1]!);
    expect(text(ui.q(".message.user .message-text"))).toBe("question one");
  });

  it("the settings dialog changes this chat's sampling and the next request carries it", async () => {
    const api = mockApi();
    ui = await mountApp({ api });
    await ui.click("#chat-settings");
    const form = ui.q<HTMLFormElement>("#chat-settings-form")!;
    const temperature = form.querySelector<HTMLInputElement>("input[name=temperature]")!;
    temperature.value = "0.4";
    temperature.dispatchEvent(new Event("input", { bubbles: true }));
    flushSync();
    form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    flushSync();
    expect(ui.app.chat.c.active.settings.temperature).toBe(0.4);
    await ask("hi");
    await done();
    expect(api.requests.find((r) => r.path === "/v1/chat/completions")!.body.temperature).toBe(0.4);
  });

  it("switching the model through the picker changes what is sent", async () => {
    const api = mockApi();
    ui = await mountApp({ api });
    const options = ui.qa("#chat-model-options button");
    await ui.click(options.find((b) => text(b).includes("m/other"))!);
    expect(text(ui.q("#chat-model-name"))).toBe("other");
    await ask("hi");
    await done();
    expect(api.requests.find((r) => r.path === "/v1/chat/completions")!.body.model).toBe("m/other");
  });

  it("restores saved chats from the library on the next start", async () => {
    const api = mockApi();
    api.replies.push({ content: "remembered" });
    ui = await mountApp({ api });
    await ask("remember me");
    await done();
    await ui.app.chat.c.flush();
    const library = ui.library;
    ui.unmount();
    ui = await mountApp({ api: mockApi(), library });
    expect(ui.qa(".message").map((m) => text(m.querySelector(".message-text")))).toEqual(["remember me", "remembered"]);
  });
});
