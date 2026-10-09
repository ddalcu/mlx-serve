<script lang="ts">
  import { copyText } from "../lib/clipboard";
  import { t } from "../lib/i18n/i18n";
  import Icon from "./Icon.svelte";

  let { lang, text }: { lang: string; text: string } = $props();
  let state = $state<"idle" | "copied" | "failed">("idle");

  async function copy() {
    state = (await copyText(text)) ? "copied" : "failed";
    setTimeout(() => (state = "idle"), 2000);
  }
</script>

<section class="code-block">
  <header>
    <span>{lang}</span>
    <button type="button" onclick={copy} title={state === "failed" ? t("Clipboard unavailable. Select the text and copy it manually.") : undefined}>
      <Icon name={state === "copied" ? "check" : "copy"} />
      <span>{state === "copied" ? t("Copied") : t("Copy")}</span>
    </button>
  </header>
  <pre><code>{text}</code></pre>
</section>
