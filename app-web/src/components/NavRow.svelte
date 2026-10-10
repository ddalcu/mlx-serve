<script lang="ts">
  import { t } from "../lib/i18n/i18n";
  import type { IconName } from "../lib/icons";
  import Icon from "./Icon.svelte";

  let { icon, label, current = false, onclick }: { icon: IconName; label: string; current?: boolean; onclick: () => void } = $props();
</script>

<button class="nav-row" aria-current={current ? "page" : undefined} {onclick}><Icon name={icon} /><span>{t(label)}</span></button>
