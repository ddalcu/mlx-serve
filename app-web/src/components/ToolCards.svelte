<script lang="ts">
  import { displayName, t } from "../lib/i18n/i18n";
  import type { ToolRound } from "../lib/core/tool-loop";
  import Icon from "./Icon.svelte";
  import ToolMedia from "./ToolMedia.svelte";

  let { rounds }: { rounds: ToolRound[] } = $props();

  function resultText(text: string) {
    const lines = text.split("\n");
    return lines.slice(0, 50).join("\n") + (lines.length > 50 ? t("\n... (%@ more lines)", [lines.length - 50]) : "");
  }
</script>

{#each rounds as round, i}
  {@const names = round.calls.map((c) => c.name)}
  {@const shown = names.length > 3 ? names.slice(0, 2) : names}
  {@const running = round.calls.some((c) => c.status === "running")}
  {@const headline = names.length === 1 ? String(round.calls[0]!.args.prompt ?? round.calls[0]!.args.query ?? "") : ""}
  {#if round.text}<div class="tool-intro">{round.text}</div>{/if}
  <details class="tool-card" class:running>
    <summary>
      <Icon name="wrench" />
      {#each shown as name, k}{#if k}<span>·</span>{/if}<code>{name}</code>{/each}
      {#if names.length > 3}<span>{t("· +%@ other tools", [names.length - 2])}</span>{/if}
      {#if headline}<span class="tool-headline">· {headline}</span>{/if}
      <span class="tool-chevron">›</span>
      <span class="sr-only">{running ? t("Running") : t("Tool calls")}</span>
    </summary>
    <div class="tool-call-body">
      {#each round.calls as call}
        <section>
          {#if names.length > 1}<code class="tool-name">{call.name}</code>{/if}
          <dl>
            {#each Object.entries(call.args) as [key, value]}<dt>{key}</dt><dd>{typeof value === "string" ? value : JSON.stringify(value)}</dd>{/each}
          </dl>
          <details class="tool-result {call.status}">
            <summary>
              <Icon name={call.status === "complete" ? "check" : call.status === "error" || call.status === "stopped" ? "x" : "wrench"} />
              <span>{t("Result · %@", [displayName(call.status)])}</span>
              <small>{call.durationMs ? t("%@ms", [call.durationMs]) : ""}</small>
              <span>›</span>
            </summary>
            <pre>{resultText(call.result ?? (call.status === "running" ? t("Running…") : t("Waiting…")))}</pre>
          </details>
        </section>
      {/each}
    </div>
  </details>
  {#each round.calls.filter((c) => c.mediaId) as call (call.mediaId)}<ToolMedia id={call.mediaId!} />{/each}
{/each}
