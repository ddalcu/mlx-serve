import { audioRequest } from "../core/media";
import type { Library } from "../core/library";
import type { StorageAdapter } from "../core/servers";
import { t } from "../i18n/i18n";
import { audioDefaults, audioProfile, buildAudioRequest } from "./audio-state.svelte";
import { ChatController, SessionStore } from "./chat-state.svelte";
import { browserTools } from "./chat-tools";
import { playSpeech, VoiceLoop, voiceAvailable, type Recognition } from "./chat-voice.svelte";
import { defaultModel, loadNotice, type Connection } from "./connection.svelte";
import { imageProfile } from "./image-state.svelte";

type RecognitionConstructor = new () => Recognition;
export type ChatDeps = {
  connection: Connection;
  storage: StorageAdapter;
  library: Library;
  /** Show the chat pane. */
  openChat: () => void;
  recognition?: () => RecognitionConstructor | undefined;
};

const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];

const readDataUrl = (file: File) =>
  new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error(t("Could not read image.")));
    reader.readAsDataURL(file);
  });

const checkImage = (url: string) =>
  new Promise<void>((resolve, reject) => {
    const img = new Image();
    img.onload = () => (img.naturalWidth * img.naturalHeight > 40_000_000 ? reject(new Error(t("Image is too large (40 megapixels maximum)."))) : resolve());
    img.onerror = () => reject(new Error(t("Could not decode image.")));
    img.src = url;
  });

function browserRecognition(): RecognitionConstructor | undefined {
  const w = window as Window & { SpeechRecognition?: RecognitionConstructor; webkitSpeechRecognition?: RecognitionConstructor };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition;
}

/** Everything the chat does that is not drawing: sessions, the composer's pending state, attachments and voice. */
export class ChatWorkspace {
  readonly deps: ChatDeps;
  readonly c: ChatController;
  readonly library: Library;
  images = $state<string[]>([]);
  error = $state("");
  attaching = $state(false);
  voice = $state.raw<VoiceLoop | null>(null);
  voiceModel = $state("");
  voiceName = $state("af_heart");
  #draftTimer: ReturnType<typeof setTimeout> | undefined;
  #lastServer = "";

  constructor(deps: ChatDeps) {
    this.deps = deps;
    this.library = deps.library;
    this.c = new ChatController(new SessionStore(deps.library));
    this.#lastServer = deps.connection.active.url;
    document.addEventListener("visibilitychange", () => document.hidden && this.voice?.stop());
    window.addEventListener("pagehide", () => this.voice?.stop());
  }


  get models() {
    return this.connection.models.filter((m) => m.capabilities.includes("chat") || !m.capabilities.length);
  }

  /** The active chat's model, when that chat belongs to the selected server. */
  get model() {
    return this.onServer() ? this.models.find((m) => m.id === this.c.active.model) : undefined;
  }

  get voicing() {
    return !!this.voice?.run;
  }

  get canAttach() {
    const model = this.model;
    return (
      !!model?.capabilities.includes("vision") ||
      (!!model && !!this.c.active.settings.toolsEnabled && this.connection.models.some((m) => imageProfile(m)?.edit || m.capabilities.includes("image-edit")))
    );
  }

  /** The notice under the composer when the chat model is not resident yet. */
  get loadNotice() {
    return this.c.run ? "" : loadNotice(this.model, this.connection.serverName());
  }

  private onServer() {
    return this.c.active.server === this.deps.connection.active.url;
  }

  private get connection() {
    return this.deps.connection;
  }

  client() {
    return this.connection.client();
  }

  /** Voice needs a secure context and the browser's own recognizer. */
  get voiceSupported() {
    return voiceAvailable(window.isSecureContext, (this.deps.recognition ?? browserRecognition)());
  }

  async init() {
    try {
      await this.c.load();
    } catch {
      this.c.persistenceError = t("Could not read saved chats. Keep this tab open and retry saving.");
    }
    const url = this.connection.active.url;
    const activeId = this.deps.storage.getItem("studio.activeChat");
    const saved = this.c.sessions.find((s) => s.id === activeId) ?? this.c.sessions.find((s) => s.server === url) ?? this.c.sessions[0];
    if (saved) this.c.select(saved.id);
    else this.c.create(url);
    this.#lastServer = this.c.active.server;
    this.connectionChanged();
  }

  /** Follow the selected server: an unsent chat retargets, an empty model is filled. */
  connectionChanged() {
    const url = this.connection.active.url;
    if (this.#lastServer !== url) {
      this.voice?.stop();
      this.#lastServer = url;
      clearTimeout(this.#draftTimer);
      this.c.stop();
      this.images = [];
      this.c.setServer(url);
      if (!this.c.active.messages.length) void this.c.save();
    }
    if (this.c.active.server === url && !this.c.active.model && this.models.length) {
      this.c.active.model = this.connection.selectedModel || defaultModel(this.models)?.id || "";
      void this.c.save();
    }
  }

  newChat() {
    this.voice?.stop();
    clearTimeout(this.#draftTimer);
    void this.c.save();
    this.images = [];
    this.error = "";
    this.c.create(this.connection.active.url, defaultModel(this.models)?.id ?? "");
    void this.c.save();
    this.deps.openChat();
  }

  selectSession(id: string) {
    this.voice?.stop();
    const session = this.c.sessions.find((s) => s.id === id);
    if (!session) return;
    const server = this.connection.servers.find((s) => s.url === session.server);
    if (!server) this.error = t("This chat’s server was removed. Add it again in Settings to send messages.");
    else if (this.connection.activeId !== server.id) {
      this.connection.select(server.id);
      void this.connection.refresh();
    }
    clearTimeout(this.#draftTimer);
    void this.c.save();
    this.c.select(id);
    this.images = [];
    this.deps.openChat();
  }

  async deleteSession(id: string) {
    await this.c.deleteSession(id);
    if (this.c.active.id === id) {
      const next = this.c.sessions.find((s) => s.server === this.connection.active.url);
      if (next) this.c.select(next.id);
      else this.c.create(this.connection.active.url);
    }
    this.deps.openChat();
  }

  /** The textarea changed: keep the draft and save it shortly after typing stops. */
  draftChanged() {
    clearTimeout(this.#draftTimer);
    this.#draftTimer = setTimeout(() => void this.c.save(), 400);
  }

  pick(id: string) {
    this.voice?.stop();
    if (this.c.run) return;
    this.c.active.model = id;
    void this.c.save();
  }

  /** Tools are rebuilt from the live catalogue before each run. */
  private prepare() {
    this.c.tools = browserTools(this.client(), this.connection.models, this.library, this.c.active);
  }

  async send() {
    if (this.voice?.run) return;
    const model = this.model;
    if (!model) return;
    this.error = "";
    clearTimeout(this.#draftTimer);
    const text = this.c.active.draft,
      images = [...this.images];
    try {
      this.prepare();
      const pending = this.c.send(text, images, model, this.client());
      if (this.c.run) this.images = [];
      await pending;
    } catch (e) {
      this.error = e instanceof Error ? e.message : t("Could not send message.");
    }
  }

  async regenerate() {
    const model = this.model;
    if (!model) return;
    try {
      this.prepare();
      await this.c.regenerate(model, this.client());
    } catch (e) {
      this.error = String(e);
    }
  }

  async edit(id: string, text: string) {
    const model = this.model;
    if (!model) return;
    try {
      this.prepare();
      await this.c.edit(id, text, model, this.client());
    } catch (e) {
      this.error = String(e);
    }
  }

  async attach(files: FileList | File[] | null) {
    const sessionId = this.c.active.id;
    if (this.attaching || !files?.length || !this.canAttach) return;
    this.attaching = true;
    try {
      if (this.images.length + files.length > 4) throw new Error(t("Attach up to four images per message."));
      const pending: string[] = [];
      for (const file of files) {
        if (!IMAGE_TYPES.includes(file.type) || file.size > 10 * 1024 * 1024) throw new Error(t("Use PNG, JPEG or WebP images up to 10 MB each."));
        const url = await readDataUrl(file);
        await checkImage(url);
        pending.push(url);
      }
      if (this.c.active.id !== sessionId) return;
      this.images.push(...pending);
      this.error = "";
    } catch (e) {
      this.error = e instanceof Error ? e.message : t("Could not attach image.");
    } finally {
      this.attaching = false;
    }
  }

  /** Speech models this server can synthesize with. */
  get speechModels() {
    return this.connection.models.filter((m) => audioProfile(m)?.tab === "voice");
  }

  startVoice(speechModelId: string, voiceName: string) {
    const Recognition = (this.deps.recognition ?? browserRecognition)();
    const model = this.speechModels.find((m) => m.id === speechModelId);
    if (!Recognition || !model) return;
    this.voiceModel = model.id;
    this.voiceName = voiceName;
    const client = this.client(),
      session = this.c.active;
    this.voice = new VoiceLoop({
      recognition: () => new Recognition(),
      send: async (text, signal) => {
        if (signal.aborted || this.c.active !== session) throw Error(t("Voice stopped."));
        const chatModel = this.model;
        if (!chatModel) throw Error(t("Choose a chat model first."));
        this.prepare();
        // Spoken turns are text-only; pending image attachments stay in the composer.
        await this.c.send(text, [], chatModel, client);
        const reply = session.messages.at(-1);
        if (reply?.status !== "complete") throw Error(reply?.error || t("Chat stopped."));
        return reply.text;
      },
      synthesize: async (text, signal) => {
        const request = buildAudioRequest(model, { ...audioDefaults(model), prompt: text, voice: this.voiceName });
        return (await audioRequest(client, request.path, request.body, { signal, timeoutMs: 600000 })).blob;
      },
      play: playSpeech,
      stopChat: () => this.c.stop(),
      changed: () => {},
    });
    void this.voice.start();
  }
}
