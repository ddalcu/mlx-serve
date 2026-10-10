<script lang="ts" module>
  export type Cell = string | { text: string; meter?: number };
</script>

<script lang="ts">
  import Meter from "./Meter.svelte";

  // A table (or its empty note) in a keyboard-focusable scroll region, so a wide table can be panned without a pointer.
  let { id, label, heads, rows, empty }: { id: string; label: string; heads: string[]; rows: Cell[][]; empty: string } = $props();
</script>

<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div {id} class="monitor-table" tabindex="0" role="group" aria-label={label}>
  {#if rows.length}
    <table>
      <thead><tr>{#each heads as head}<th scope="col">{head}</th>{/each}</tr></thead>
      <tbody>
        {#each rows as row}
          <tr>
            {#each row as cell}
              <td>{typeof cell === "string" ? cell : cell.text}{#if typeof cell !== "string" && cell.meter != null}<Meter value={cell.meter} />{/if}</td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  {:else}
    <p class="section-description">{empty}</p>
  {/if}
</div>
