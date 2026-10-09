<script lang="ts">
  import Dialog from "../../components/Dialog.svelte";
  import { getApp } from "../../lib/app.svelte";
  import { t } from "../../lib/i18n/i18n";

  let { onclose }: { onclose: () => void } = $props();
  const chat = getApp().chat;
  const model = chat.model;
  const message = [...chat.c.active.messages].reverse().find((m) => m.role === "assistant" && m.usage);
  const usage = message?.usage;
  const number = (value: unknown) => (typeof value === "number" && Number.isFinite(value) ? value.toLocaleString() : t("Not reported"));
  const total = typeof usage?.total_tokens === "number" ? usage.total_tokens : null;
</script>

<Dialog title={t("Context window")} {onclose}>
  {#snippet children()}
    <p>{model?.id}</p>
    <dl class="context-stats">
      <dt>{t("Prompt")}</dt>
      <dd>{number(usage?.prompt_tokens)}</dd>
      <dt>{t("Used")}</dt>
      <dd>{number(total)}</dd>
      <dt>{t("Context size")}</dt>
      <dd>{number(model?.contextLength)}</dd>
      <dt>{t("Remaining")}</dt>
      <dd>{number(total !== null && model?.contextLength ? Math.max(0, model.contextLength - total) : null)}</dd>
      <dt>{t("Decode speed")}</dt>
      <dd>{message?.tokensPerSecond ? number(message.tokensPerSecond) + " tok/sec" : t("Not reported")}</dd>
    </dl>
    <p>{t("Reported by the last completed request; includes no estimate for your draft.")}</p>
  {/snippet}
</Dialog>
