import { getContext } from "svelte";
import { N } from "./i18n/i18n";
import { pageApiKey, pageServer } from "./core/console";
import { Library } from "./core/library";
import { AudioWorkspace } from "./state/audio-workspace.svelte";
import { ChatWorkspace } from "./state/chat-workspace.svelte";
import { Connection } from "./state/connection.svelte";
import { ImageWorkspace } from "./state/image-workspace.svelte";
import { MonitorWorkspace } from "./state/monitor-workspace.svelte";
import { InterfacePreferences } from "./state/preferences.svelte";
import { Router, type View } from "./state/router.svelte";
import { SafeStorage } from "./state/storage.svelte";
import { VideoWorkspace } from "./state/video-workspace.svelte";

export const titles: Record<View, string> = {
  chat: N("New Chat"),
  api: N("API Reference"),
  models: N("Models"),
  monitoring: N("Monitoring"),
  settings: N("Settings"),
  image: N("Image Generation"),
  video: N("Video Generation"),
  audio: N("Audio & Music"),
  library: N("Library"),
};

export type SettingsCategory = "all" | "interface" | "servers" | "about";

/** Everything the panes share: storage, preferences, the selected server and where we are. */
export class App {
  readonly storage = new SafeStorage();
  readonly prefs = new InterfacePreferences(this.storage);
  readonly connection = new Connection(pageServer(location), this.storage, {}, pageApiKey(location.search));
  readonly router = new Router();
  readonly library: Library;
  readonly chat: ChatWorkspace;
  readonly image: ImageWorkspace;
  readonly audio: AudioWorkspace;
  readonly video: VideoWorkspace;
  readonly monitor: MonitorWorkspace;
  phone = $state(false);
  sidebarOpen = $state(true);
  settingsCategory = $state<SettingsCategory>("all");

  /** `drafts` names a separate database per pane, so source pictures and drafts stay out of the Library's gallery. */
  constructor(library = new Library(), drafts: (name: string) => Pick<Library, "get" | "put"> = (name) => new Library(name)) {
    this.library = library;
    this.chat = new ChatWorkspace({ connection: this.connection, storage: this.storage, library, openChat: () => this.go("chat") });
    this.image = new ImageWorkspace(this.connection, library, drafts("mlx-serve-studio-image-drafts"));
    this.audio = new AudioWorkspace(this.connection, library, drafts("mlx-serve-studio-audio-drafts"));
    this.video = new VideoWorkspace(this.connection, library, drafts("mlx-serve-studio-video-drafts"));
    this.monitor = new MonitorWorkspace(this.connection);
    const query = matchMedia("(max-width: 700px)");
    const apply = () => {
      this.phone = query.matches;
      this.sidebarOpen = !query.matches;
    };
    apply();
    query.addEventListener("change", apply);
  }

  /** Switch views; on a phone the sidebar closes over the new pane. */
  go(view: View) {
    this.router.go(view);
    if (this.phone) this.sidebarOpen = false;
  }
}

export const appKey = Symbol("app");
/** The running App, for components too deep to be handed it. */
export const getApp = () => getContext<App>(appKey);
