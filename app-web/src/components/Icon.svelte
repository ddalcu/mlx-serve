<script lang="ts">
  import { icons, type IconName } from "../lib/icons";

  let { name, filled = false }: { name: IconName; filled?: boolean } = $props();
</script>

<svg class="icon i-{name}" viewBox="0 0 24 24" aria-hidden="true" fill={filled ? "currentColor" : "none"} stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d={icons[name]} /></svg>
