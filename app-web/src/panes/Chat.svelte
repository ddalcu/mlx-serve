<script lang="ts">
  import { untrack } from "svelte";
  import Icon from "../components/Icon.svelte";
  import MessageRow from "../components/MessageRow.svelte";
  import type { App } from "../lib/app.svelte";
  import { displayName, t } from "../lib/i18n/i18n";
  import { canThink } from "../lib/state/chat-state.svelte";
  import ContextDialog from "./chat/ContextDialog.svelte";
  import ModelPalette from "./chat/ModelPalette.svelte";
  import SettingsDialog from "./chat/SettingsDialog.svelte";
  import VoiceDialog from "./chat/VoiceDialog.svelte";

  let { app }: { app: App } = $props();
  const chat = $derived(app.chat);
  const connection = $derived(app.connection);
  const session = $derived(chat.c.active);
  const busy = $derived(!!chat.c.run);
  const populated = $derived(session.messages.length > 0);
  const model = $derived(chat.model);
  const toolsOn = $derived(!!session.settings.toolsEnabled);
  const used = $derived([...session.messages].reverse().find((m) => m.usage)?.usage?.total_tokens);
  const loaded = $derived(model?.loaded === true && connection.status === "online");
  const wrongServer = $derived(connection.active.url !== session.server);
  const enabled = $derived(busy || chat.voicing || !(chat.attaching || !model || wrongServer || (!session.draft.trim() && !chat.images.length)));
  const status = $derived(
    chat.voice?.error ||
      (chat.voicing ? t("Voice: %@. Stop ends voice chat.", [displayName(chat.voice?.phase ?? "off")]) : "") ||
      chat.c.persistenceError ||
      chat.error ||
      (chat.c.pendingWrites ? t("Saving…") : "") ||
      (wrongServer
        ? t("Select this chat’s server in Settings to continue.")
        : !model
          ? connection.status === "checking"
            ? t("Checking connection…")
            : t("You need a model to chat. Choose a server in Settings or add a model in the Mac app.")
          : ""),
  );

  let dialog = $state<"" | "settings" | "palette" | "context" | "voice">("");
  let modelMenu = $state(false);
  let transcript = $state<HTMLElement>();
  let input = $state<HTMLTextAreaElement>();
  let files = $state<HTMLInputElement>();
  let follow = $state(true);

  function toBottom() {
    if (transcript) transcript.scrollTop = transcript.scrollHeight;
    follow = true;
  }

  // The transcript sticks to the end while a reply grows, until the reader scrolls away.
  $effect(() => {
    const el = transcript;
    if (!el) return;
    let frame = 0;
    const stick = () => {
      if (follow && !frame)
        frame = requestAnimationFrame(() => {
          frame = 0;
          el.scrollTop = el.scrollHeight;
        });
    };
    const observer = new MutationObserver(stick);
    observer.observe(el, { childList: true, subtree: true, characterData: true });
    el.addEventListener("load", stick, true);
    return () => {
      observer.disconnect();
      el.removeEventListener("load", stick, true);
      cancelAnimationFrame(frame);
    };
  });
  $effect(() => {
    session.id;
    if (populated) untrack(toBottom);
  });

  // The composer grows with its text, up to a point.
  $effect(() => {
    session.draft;
    if (!input) return;
    input.style.height = "auto";
    input.style.height = Math.min(180, Math.max(36, input.scrollHeight)) + "px";
  });

  // Leaving the chat stops what is running and drops what was only pending.
  $effect(() => () => {
    chat.voice?.stop();
    chat.c.stop();
    void chat.c.save();
    chat.images = [];
  });

  function sendOrStop() {
    if (chat.voice?.run) chat.voice.stop();
    else if (chat.c.run) chat.c.stop();
    else {
      follow = true;
      void chat.send();
    }
  }
  function keydown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!chat.c.run) {
        follow = true;
        void chat.send();
      }
    }
  }
  function paste(event: ClipboardEvent) {
    const pasted = event.clipboardData?.files;
    if (pasted?.length) {
      event.preventDefault();
      void chat.attach(pasted);
    }
  }
  function scrolled() {
    if (transcript) follow = transcript.scrollHeight - transcript.clientHeight - transcript.scrollTop < 45;
  }
</script>

<section class="chat-screen live-chat" class:has-messages={populated} aria-label={t("Chat")}>
  <div class="wordmark" role="img" aria-label="MLX-Serve" hidden={populated}></div>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
  <div id="transcript" class="transcript" tabindex="0" aria-label={t("Conversation")} hidden={!populated} bind:this={transcript} onscroll={scrolled}>
    {#each session.messages as m (m.id)}
      <article id={m.id} class="message {m.role}"><MessageRow message={m} last={m === session.messages.at(-1)} /></article>
    {/each}
  </div>
  <div class="chat-center">
    <div id="chat-welcome" hidden={populated}>
      <h1>{t("How can I help you today?")}</h1>
      <div class="discovery">
        <details>
          <summary class="chip"><Icon name="sparkles" />{t("Create Media")} <Icon name="chevron-down" /></summary>
          <div class="discovery-menu">
            <button onclick={() => app.go("image")}><Icon name="image" />{t("Image Generation")}</button>
            <button onclick={() => app.go("video")}><Icon name="film" />{t("Video Generation")}</button>
            <button onclick={() => app.go("audio")}><Icon name="audio-lines" />{t("Audio & Music")}</button>
          </div>
        </details>
        <button class="chip" onclick={() => app.go("models")}><Icon name="search" />{t("Browse Models")}</button>
      </div>
    </div>
    <div class="jump-latest-row"><button id="jump-latest" hidden={follow} onclick={toBottom}>{t("Jump to the latest message ↓")}</button></div>
    <div class="composer">
      <div id="pending-images" class="pending-images">
        {#each chat.images as url, i}
          <span><img src={url} alt={t("Pending image %@", [i + 1])} /><button aria-label={t("Remove image %@", [i + 1])} onclick={() => chat.images.splice(i, 1)}>×</button></span>
        {/each}
      </div>
      <textarea id="chat-input" rows="1" aria-label={t("Message")} placeholder={t("Ask me anything…")} bind:this={input} bind:value={session.draft} oninput={() => chat.draftChanged()} onkeydown={keydown} onpaste={paste} disabled={chat.voicing}></textarea>
      <div class="composer-tools">
        <button class="icon-button" id="attach-image" aria-label={t("Attach image")} title={t("Attach image")} disabled={busy || chat.voicing || chat.attaching || !chat.canAttach} onclick={() => files?.click()}><Icon name="paperclip" /></button>
        <input
          type="file"
          id="image-files"
          accept="image/png,image/jpeg,image/webp"
          multiple
          hidden
          bind:this={files}
          onchange={async () => {
            await chat.attach(files?.files ?? null);
            if (files) files.value = "";
          }}
        />
        <button
          class="icon-button"
          id="chat-thinking"
          aria-label={t("Thinking")}
          title={t("Thinking")}
          aria-pressed={session.settings.thinking}
          disabled={busy || chat.voicing || !model || !canThink(model)}
          onclick={() => {
            session.settings.thinking = !session.settings.thinking;
            void chat.c.save();
          }}><Icon name="lightbulb" /></button
        >
        <button
          class="icon-button"
          id="chat-tools"
          aria-label={t("Tools")}
          aria-pressed={toolsOn}
          disabled={busy || chat.voicing}
          title={t("Tools · %@: Image, speech, music, sound, video, Library search — run a tool-calling loop", [toolsOn ? "ON" : "OFF"])}
          onclick={() => {
            session.settings.toolsEnabled = !toolsOn;
            void chat.c.save();
          }}><Icon name="wrench" /></button
        >
        <details class="chat-model-menu" bind:open={modelMenu}>
          <summary class="composer-model" aria-label={t("Chat model")}>
            <Icon name="cpu" /><span id="chat-model-name">{session.model.split("/").at(-1) || t("Select a model")}</span>
            <span class="model-chevrons" aria-hidden="true"><Icon name="chevron-down" /><Icon name="chevron-down" /></span>
            <span id="chat-model-status" class="status-dot" data-status={loaded ? "online" : "idle"} title={loaded ? t("Model loaded · Server online") : t("Model not confirmed loaded")}></span>
          </summary>
          <div id="chat-model-options" class="model-menu-options">
            {#each chat.models as m (m.id)}
              <button
                disabled={busy}
                onclick={() => {
                  chat.pick(m.id);
                  modelMenu = false;
                }}>{m.id === session.model ? "✓ " : ""}{m.id}</button
              >
            {:else}
              <p>{t("No chat models on this server")}</p>
            {/each}
            <button
              id="switch-model"
              disabled={busy}
              onclick={() => {
                modelMenu = false;
                dialog = "palette";
              }}>{t("Switch Model…")}</button
            >
            <button onclick={() => app.go("models")}>{t("Manage Models…")}</button>
          </div>
        </details>
        <button class="context-button" id="chat-context" aria-label={t("Context window")} title={t("Context window")} hidden={!model?.contextLength} onclick={() => (dialog = "context")}
          >{typeof used === "number" && model?.contextLength ? ((used / model.contextLength) * 100).toFixed(1) + "%" : ""}</button
        >
        <button
          class="icon-button"
          id="chat-voice"
          hidden={!chat.voiceSupported}
          aria-pressed={chat.voicing}
          aria-label={chat.voicing ? t("Stop voice chat") : t("Voice chat")}
          title={t("Voice chat")}
          onclick={() => {
            if (chat.voice?.run) chat.voice.stop();
            else if (!chat.c.run) {
              if (session.draft.trim() || chat.images.length) chat.error = t("Send or clear the composer before starting voice chat.");
              else dialog = "voice";
            }
          }}><Icon name="audio-lines" /></button
        >
        <button class="icon-button" id="chat-settings" aria-label={t("Chat settings")} title={t("Chat settings")} disabled={busy || chat.voicing} onclick={() => (dialog = "settings")}><Icon name="settings" /></button>
        <button class="send-button" id="chat-send" aria-label={chat.voicing ? t("Stop voice chat") : busy ? t("Stop generation") : t("Send message")} disabled={!enabled} onclick={sendOrStop}>
          {#if busy || chat.voicing}<Icon name="stop" filled />{:else}<Icon name="arrow-up" />{/if}
        </button>
      </div>
    </div>
    <p id="chat-load-notice" class="chat-status">{chat.loadNotice}</p>
    <p id="chat-status" class="chat-status" role="status">{status}</p>
    <button id="retry-save" hidden={!chat.c.persistenceError} onclick={() => void chat.c.save()}>{t("Retry saving")}</button>
  </div>
</section>

{#if dialog === "settings"}<SettingsDialog onclose={() => (dialog = "")} />{/if}
{#if dialog === "palette"}<ModelPalette onclose={() => (dialog = "")} />{/if}
{#if dialog === "context"}<ContextDialog onclose={() => (dialog = "")} />{/if}
{#if dialog === "voice"}<VoiceDialog onclose={() => (dialog = "")} />{/if}
