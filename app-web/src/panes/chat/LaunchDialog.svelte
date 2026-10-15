<script lang="ts">
  import CodeBlock from "../../components/CodeBlock.svelte";
  import Dialog from "../../components/Dialog.svelte";
  import { getApp } from "../../lib/app.svelte";
  import { launchCommand } from "../../lib/core/console";
  import { t } from "../../lib/i18n/i18n";

  let { agent, label, onclose }: { agent: string; label: string; onclose: () => void } = $props();
  const app = getApp();
  const command = $derived(launchCommand(app.connection.active.url, agent, app.chat.c.active.model || undefined));
</script>

<Dialog title={t("Launch %@", [label])} class="launch-dialog" {onclose}>
  {#snippet children()}
    <p>{t("Run this in a terminal on the computer where %@ is installed. It writes the agent's settings under ~/.mlx-serve and starts it against this server.", [label])}</p>
    <CodeBlock lang="sh" text={command} />
  {/snippet}
</Dialog>
