<script lang="ts">
  import { getApp } from "../lib/app.svelte";
  import { t } from "../lib/i18n/i18n";
  import SessionDialog from "../panes/chat/SessionDialog.svelte";
  import Icon from "./Icon.svelte";

  const chat = getApp().chat;
  let actions = $state("");
  const label = (title: string) => (title === "New Chat" ? t("New Chat") : title);
</script>

{#each chat.c.sessions as s (s.id)}
  <div class="session-entry">
    <button class="nav-row" class:active={s.id === chat.c.active.id} title={label(s.title)} onclick={() => chat.selectSession(s.id)}>
      <Icon name="message-square" /><span>{label(s.title)}</span>
    </button>
    <button class="session-actions" aria-label={t("Actions for %@", [s.title])} onclick={() => (actions = s.id)}>⋯</button>
  </div>
{/each}
{#if actions}<SessionDialog id={actions} onclose={() => (actions = "")} />{/if}
