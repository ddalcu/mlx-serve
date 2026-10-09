import { t, N } from "../i18n/i18n";
import { newId } from "../core/id";
import { toolLoop, roundMessages, cleanToolRounds } from "../core/tool-loop";
import { chat, visionMessage } from "../core/chat";
import { Library } from "../core/library";
import type { ToolRound } from "../core/tool-loop";
import type { ChatRequest } from "../core/chat";
import type { LoopOptions } from "../core/tool-loop";
import type { Client } from "../core/client";
import type { ChatMessage } from "../core/chat";
import type { Model } from "../core/models";
export type ChatSettings = { system: string; temperature: number | null; maxTokens: number | null; thinking: boolean; mtp: boolean | null; toolsEnabled?: boolean; };
export type Message = { id: string; role: 'user' | 'assistant'; text: string; images?: string[]; imagePurpose?: 'edit'; thinking?: string; thinkingSeconds?: number; createdAt: number; status?: string; tokensPerSecond?: number | null; usage?: Record<string, unknown>; error?: string; toolRounds?: ToolRound[]; };
export type Session = { version: 1; id: string; title: string; server: string; model: string; createdAt: number; updatedAt: number; settings: ChatSettings; messages: Message[]; draft: string; };
function newSession(server: string, model: string = ""): Session {
  return {
    version: 1,
    id: newId(),
    title: "New Chat",
    server,
    model,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    settings: {
      system: "",
      temperature: null,
      maxTokens: null,
      thinking: false,
      mtp: null,
    },
    messages: [],
    draft: "",
  };
}
/**
 * Handles in-band tags split across SSE chunks without displaying tag fragments.
 */
function splitThinking(raw: string) {
  // In-band reasoning is a leading envelope, not literal tags in answer prose.
  const start = raw.trimStart();
  if (!start.startsWith("<think>") && !"<think>".startsWith(start))
    return { text: raw, thinking: "" };
  let text = "",
    thinking = "",
    inside = false,
    offset = 0;
  while (offset < raw.length) {
    const tag = inside ? "</think>" : "<think>",
      at = raw.indexOf(tag, offset);
    if (at >= 0) {
      if (inside) thinking += raw.slice(offset, at);
      else text += raw.slice(offset, at);
      inside = !inside;
      offset = at + tag.length;
      if (!inside) {
        text += raw.slice(offset);
        break;
      }
      continue;
    }
    let tail = raw.slice(offset);
    for (let n = Math.min(tag.length - 1, tail.length); n > 0; n--)
      if (tail.endsWith(tag.slice(0, n))) {
        tail = tail.slice(0, -n);
        break;
      }
    if (inside) thinking += tail;
    else text += tail;
    break;
  }
  return { text, thinking };
}
/** Swift Models/ChatModels.swift ThinkingDuration.label. */
function thinkingLabel(seconds: number = 0) {
  if (seconds < 1) return t("Thinking");
  const total = Math.round(seconds),
    minutes = Math.floor(total / 60),
    rest = total % 60;
  const units = [];
  if (minutes)
    units.push(t(minutes === 1 ? N("%@ minute") : N("%@ minutes"), [minutes]));
  if (rest)
    units.push(t(rest === 1 ? N("%@ second") : N("%@ seconds"), [rest]));
  return t("Thinking took ") + units.join(" ");
}
const canThink = (model: Model) =>
  model.capabilities.includes("reasoning") ||
  model.meta.supports_thinking === true;
const canMTP = (model: Model) =>
  model.meta.mtp_available === true || model.meta.supports_mtp === true;
function requestFor(session: Session, model: Model): ChatRequest {
  const s = session.settings;
  if (
    s.temperature !== null &&
    (!Number.isFinite(s.temperature) || s.temperature < 0 || s.temperature > 2)
  )
    throw new Error(t("Temperature must be between 0 and 2."));
  if (
    s.maxTokens !== null &&
    (!Number.isSafeInteger(s.maxTokens) || s.maxTokens < 1)
  )
    throw new Error(t("Max tokens must be a positive whole number."));
   const messages: ChatMessage[] = [];
  if (s.system.trim()) messages.push({ role: "system", content: s.system });
  for (const m of session.messages) {
    for (const round of m.toolRounds ?? [])
      messages.push(...roundMessages(round));
    if (m.role === "assistant" && !m.text) continue;
    if (m.images?.length) {
      if (
        !model.capabilities.includes("vision") &&
        (s.toolsEnabled || m.imagePurpose === "edit")
      ) {
        messages.push({
          role: m.role,
          content:
            m.text +
            "\n[Images attached for editing; this chat model cannot see them.]",
        });
        continue;
      }
      if (!model.capabilities.includes("vision"))
        throw new Error(
          t("Choose a vision model for this conversation’s images."),
        );
      const content =
        ([
          { type: "text", text: m.text },
        ] as Exclude<ChatMessage['content'],string>);
      for (const url of m.images) {
        const image = visionMessage("", url);
        content.push(
          (image.content as Exclude<typeof image.content,string>)[1],
        );
      }
      messages.push({ role: m.role, content });
    } else messages.push({ role: m.role, content: m.text });
  }
  return {
    model: session.model,
    messages,
    ...(s.temperature !== null ? { temperature: s.temperature } : {}),
    ...(s.maxTokens !== null ? { max_tokens: s.maxTokens } : {}),
    ...(canThink(model) ? { enable_thinking: s.thinking } : {}),
    ...(canMTP(model) && s.mtp !== null ? { enable_mtp: s.mtp } : {}),
  };
}
class SessionStore {
  library: Library;
  constructor(library = new Library()) {
    this.library = library;
  }
  save(session: Session) {
    // Explicit schema excludes credentials, client instances and other runtime state.
    const {
      version,
      id,
      title,
      server,
      model,
      createdAt,
      updatedAt,
      settings,
      messages,
      draft,
    } = session;
    return this.library.put(id, {
      type: "chat",
      server,
      model,
      createdAt: updatedAt,
      prompt: title,
      blob: new Blob(
        [
          JSON.stringify({
            version,
            id,
            title,
            server,
            model,
            createdAt,
            updatedAt,
            settings,
            messages,
            draft,
          }),
        ],
        { type: "application/json" },
      ),
    });
  }
  async load() {
     const sessions: Session[] = [];
    for (const row of await this.library.list({ type: "chat" })) {
      const item = await this.library.get(row.id);
      if (!item) continue;
      try {
        const s = JSON.parse(await item.blob.text());
        // Immutable chat exports use a different schema from editable sessions.
        if (
          s.version !== 1 ||
          s.id !== row.id ||
          !Array.isArray(s.messages) ||
          typeof s.title !== "string" ||
          typeof s.model !== "string" ||
          typeof s.server !== "string" ||
          !s.settings
        )
          continue;
        if (
          !s.messages.every(
            ( m: Message) =>
              typeof m.id === "string" &&
              typeof m.text === "string" &&
              ["user", "assistant"].includes(m.role),
          )
        )
          continue;
        for (const m of s.messages) {
          if (m.toolRounds) m.toolRounds = cleanToolRounds(m.toolRounds);
          if (m.status === "streaming") m.status = "stopped";
          for (const round of m.toolRounds ?? [])
            for (const call of round.calls)
              if (["pending", "running"].includes(call.status))
                call.status = "stopped";
        }
        sessions.push(s);
      } catch {
        /* Preserve unknown or corrupt blobs; do not overwrite them. */
      }
    }
    return sessions.sort((a, b) => b.updatedAt - a.updatedAt);
  }
   delete(id: string) {
    return this.library.delete(id);
  }
}
type Run = { session: Session; message: Message; abort: AbortController };

/**
 * Sessions, the active one and the single running generation. Everything the UI shows is reactive state;
 * a pushed message is re-read through its proxy before it is mutated, or the screen would not follow it.
 */
class ChatController {
  sessions = $state<Session[]>([]);
  active = $state<Session>(newSession("http://localhost"));
  run = $state.raw<Run>();
  tools = $state.raw<LoopOptions>();
  persistenceError = $state("");
  pendingWrites = $state(0);
  queue = Promise.resolve();
  stream: typeof chat;
  store: SessionStore;
  constructor(store: SessionStore = new SessionStore(), stream: typeof chat = chat) {
    this.store = store;
    this.stream = stream;
  }
  async load() {
    this.sessions = await this.store.load();
  }
  create(server: string, model: string = "") {
    this.stop();
    if (!this.active.messages.length && !this.active.draft.trim() && this.sessions.includes(this.active)) {
      this.setServer(server);
      this.active.model = model;
      return this.active;
    }
    this.active = newSession(server, model);
    this.sessions.unshift(this.active);
    return this.active;
  }
  /** Retarget only an unsent conversation. Existing history keeps its server. */
  setServer(server: string) {
    this.stop();
    if (!this.active.messages.length && this.active.server !== server) {
      this.active.server = server;
      this.active.model = "";
    }
  }
  select(id: string) {
    const s = this.sessions.find((s) => s.id === id);
    if (s) {
      this.stop();
      this.active = s;
    }
  }
  save(s: Session = this.active) {
    s.updatedAt = Date.now();
    const snapshot = $state.snapshot(s) as Session;
    this.pendingWrites++;
    this.queue = this.queue
      .then(() => this.store.save(snapshot))
      .then(() => {
        this.persistenceError = "";
      })
      .catch(() => {
        this.persistenceError = t("Could not save chat in this browser. Keep this tab open and retry saving.");
      })
      .then(() => {
        this.pendingWrites--;
      });
    return this.queue;
  }
  flush() {
    return this.queue;
  }
  stop() {
    const run = this.run;
    if (!run) return;
    this.run = undefined;
    run.abort.abort();
    run.message.status = "stopped";
    for (const round of run.message.toolRounds ?? [])
      for (const call of round.calls) if (["pending", "running"].includes(call.status)) call.status = "stopped";
    void this.save(run.session);
  }
  async send(text: string, images: string[], model: Model, client: Client) {
    if (this.run || (!text.trim() && !images.length)) return;
    const s = this.active;
    if (client.baseUrl !== s.server) throw new Error(t("Select this chat’s server before sending."));
    const user = {
      id: newId(),
      role: "user",
      text,
      images: [...images],
      ...(images.length && !model.capabilities.includes("vision") && s.settings.toolsEnabled ? { imagePurpose: "edit" } : {}),
      createdAt: Date.now(),
    } as Message;
    // Validate before mutating the transcript or consuming the draft.
    requestFor({ ...s, messages: [...s.messages, user] }, model);
    s.messages.push(user);
    s.draft = "";
    if (s.title === "New Chat") s.title = (text.trim() || t("Image chat")).slice(0, 60);
    await this.generate(s, model, client);
  }
  async generate(s: Session, model: Model, client: Client) {
    if (client.baseUrl !== s.server) throw new Error(t("Select this chat’s server before sending."));
    const request = requestFor(s, model);
    s.messages.push({ id: newId(), role: "assistant", text: "", thinking: "", createdAt: Date.now(), status: "streaming" } as Message);
    const message = s.messages[s.messages.length - 1]!;
    const run: Run = { session: s, message, abort: new AbortController() };
    this.run = run;
    let raw = "",
      reasoning = "",
      started = 0,
      ended = 0,
      lastSaved = performance.now();
    void this.save(s);
    try {
      const events =
        this.tools && s.settings.toolsEnabled
          ? toolLoop(client, request, { ...this.tools, stream: this.stream, signal: run.abort.signal })
          : this.stream(client, request, { signal: run.abort.signal });
      for await (const event of events) {
        if (this.run !== run) break;
        if (event.type === "tool-round") {
          message.toolRounds ??= [];
          message.toolRounds.push({ ...structuredClone(event.round), text: message.text });
          message.text = "";
          raw = "";
        } else if (event.type === "tool-result") {
          const round = message.toolRounds?.at(-1);
          if (round) round.calls = structuredClone(event.round.calls);
        } else if (event.type === "content" || event.type === "reasoning") {
          if (event.type === "reasoning") reasoning += event.text;
          else raw += event.text;
          const parsed = splitThinking(raw);
          message.text = parsed.text;
          message.thinking = reasoning + parsed.thinking;
          if (message.thinking && !started) started = performance.now();
          if (started && message.text && !ended) ended = performance.now();
          if (started) message.thinkingSeconds = ((ended || performance.now()) - started) / 1000;
        } else if (event.type === "done") {
          message.tokensPerSecond = event.metrics.tokensPerSecond;
          message.usage = event.usage;
        }
        if (performance.now() - lastSaved > 1000) {
          lastSaved = performance.now();
          void this.save(s);
        }
      }
      if (this.run === run) message.status = "complete";
    } catch (error) {
      if (this.run === run) {
        message.status = "error";
        message.error = error instanceof Error ? error.message : t("Chat request failed.");
      }
    } finally {
      if (this.run === run) {
        this.run = undefined;
        await this.save(s);
      }
    }
  }
  async regenerate(model: Model, client: Client) {
    if (this.run) return;
    const s = this.active;
    const last = s.messages.at(-1);
    if (last?.role === "assistant") {
      requestFor({ ...s, messages: s.messages.slice(0, -1) }, model);
      s.messages.pop();
    }
    if (s.messages.at(-1)?.role === "user") await this.generate(s, model, client);
  }
  async edit(id: string, text: string, model: Model, client: Client) {
    if (this.run) return;
    const s = this.active,
      i = s.messages.findIndex((m) => m.id === id);
    if (i < 0 || !text.trim()) return;
    const m = s.messages[i]!;
    if (m.role === "assistant") {
      m.text = text;
      await this.save();
      return;
    }
    requestFor({ ...s, messages: s.messages.slice(0, i + 1) }, model);
    m.text = text;
    s.messages = s.messages.slice(0, i + 1);
    await this.generate(s, model, client);
  }
  deleteMessage(id: string) {
    if (this.run) return;
    const s = this.active,
      i = s.messages.findIndex((m) => m.id === id);
    if (i < 0) return;
    if (s.messages[i]!.role === "assistant" && s.messages[i - 1]?.role === "user") s.messages.splice(i - 1, 2);
    else s.messages.splice(i, 1);
    void this.save();
  }
  async deleteSession(id: string) {
    if (this.run?.session.id === id) this.stop();
    await this.flush();
    await this.store.delete(id);
    this.sessions = this.sessions.filter((s) => s.id !== id);
  }
}

export { newSession, splitThinking, thinkingLabel, canThink, canMTP, requestFor, SessionStore, ChatController };
