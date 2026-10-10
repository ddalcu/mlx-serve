<script lang="ts">
  import { fmt } from "../lib/format";

  // A 60-point sparkline that can be read with the pointer or the arrow keys.
  let { id, label, values, from, to, description }: { id: string; label: string; values: (number | null)[]; from: number; to: number; description: string } = $props();
  let index = $state(59);
  const max = $derived(Math.max(1, ...values.map((v) => v ?? 0)));
  const y = (v: number) => 41 - (v / max) * 38;
  const path = $derived(
    values
      .map((v, i) => (v === null ? "" : `${values[i - 1] === null || i === 0 ? "M" : "L"}${(i * 300) / 59},${y(v)} `))
      .join(""),
  );
  const reading = $derived(`${new Date(from + ((index + 1) * (to - from)) / 60).toLocaleTimeString()} · ${fmt(values[index])}`);
  const at = (event: PointerEvent) => {
    const box = event.currentTarget as HTMLElement;
    index = Math.max(0, Math.min(59, Math.round(((event.clientX - box.getBoundingClientRect().left) / box.clientWidth) * 59)));
  };
</script>

<section class="metric-card monitor-chart">
  <h2 id="chart-label-{id}">{label}</h2>
  <p id="chart-value-{id}" class="metric-detail">{reading}</p>
  <!-- svelte-ignore a11y_no_noninteractive_tabindex, a11y_no_noninteractive_element_interactions -->
  <div
    id="chart-{id}"
    class="metric-spark"
    tabindex="0"
    role="img"
    title={description}
    aria-label="{label} {reading}"
    onpointermove={at}
    onpointerleave={() => (index = 59)}
    onkeydown={(e) => {
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        e.preventDefault();
        index = Math.max(0, Math.min(59, index + (e.key === "ArrowLeft" ? -1 : 1)));
      }
    }}
  >
    <svg viewBox="0 0 300 44" preserveAspectRatio="none" aria-hidden="true">
      <path class="spark-baseline" d="M0 43H300" />
      <path d={path} />
      <circle r="3" cx={(index * 300) / 59} cy={y(values[index] ?? 0)} style:display={values[index] == null ? "none" : "block"}></circle>
    </svg>
  </div>
</section>
