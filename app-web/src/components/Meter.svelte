<script lang="ts">
  import { fmt, level } from "../lib/format";
  import { t } from "../lib/i18n/i18n";

  let { value }: { value: number | null | undefined } = $props();
</script>

<meter min="0" max="100" value={Math.min(100, Math.max(0, value ?? 0))} data-level={level(value)} aria-label={t("%@ percent", [fmt(value)])} hidden={value == null}></meter>
