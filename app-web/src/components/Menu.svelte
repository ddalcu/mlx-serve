<script lang="ts">
  import type { Snippet } from "svelte";
  import Icon from "./Icon.svelte";

  // A button that opens a native popover of choices. Items are buttons with a data-value; arrows, Home/End and typeahead move.
  let { id, label, onchoose, disabled = false, children }: { id: string; label: string; onchoose: (value: string) => void; disabled?: boolean; children: Snippet } = $props();
  let trigger: HTMLButtonElement;
  let popup: HTMLElement;
  let expanded = $state(false);
  let search = "";
  let searchedAt = 0;

  const items = () => [...popup.querySelectorAll<HTMLButtonElement>("button:not(:disabled)")];

  function close(focus = false) {
    popup.hidePopover();
    expanded = false;
    if (focus) trigger.focus();
  }

  function open(last = false) {
    if (trigger.matches(":disabled")) return;
    popup.showPopover();
    expanded = true;
    const rect = trigger.getBoundingClientRect();
    popup.style.maxHeight = `${Math.max(100, innerHeight - 16)}px`;
    popup.style.left = `${Math.max(8, Math.min(rect.left, innerWidth - popup.offsetWidth - 8))}px`;
    popup.style.top = `${Math.max(8, Math.min(rect.bottom + 4, innerHeight - popup.offsetHeight - 8))}px`;
    const choices = items();
    (last ? choices.at(-1) : choices.find((b) => b.getAttribute("aria-checked") === "true") || choices[0])?.focus();
  }

  function pick(event: MouseEvent) {
    const button = event.target instanceof Element ? event.target.closest("button") : null;
    if (!button || button.disabled || trigger.matches(":disabled")) return;
    const value = button.dataset.value;
    if (value === undefined) return;
    close();
    onchoose(value);
    trigger.focus();
  }

  function keys(event: KeyboardEvent) {
    if (event.key === "Escape" || event.key === "Tab") {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
      }
      close(true);
      return;
    }
    const choices = items(),
      index = choices.indexOf(document.activeElement as HTMLButtonElement);
    let next: number | undefined;
    if (event.key === "ArrowDown") next = (index + 1) % choices.length;
    else if (event.key === "ArrowUp") next = (index - 1 + choices.length) % choices.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = choices.length - 1;
    else if (event.key.length === 1 && event.key !== " " && !event.ctrlKey && !event.metaKey && !event.altKey) {
      search = (Date.now() - searchedAt > 700 ? "" : search) + event.key.toLowerCase();
      searchedAt = Date.now();
      next = choices.findIndex((b) => b.textContent?.trim().toLowerCase().startsWith(search));
    }
    if (next !== undefined) {
      event.preventDefault();
      choices[next]?.focus();
    }
  }
</script>

<span class="image-menu">
  <button
    type="button"
    {id}
    class="chip"
    aria-haspopup="menu"
    aria-expanded={expanded}
    aria-controls="{id}-menu"
    {disabled}
    bind:this={trigger}
    onclick={() => (popup.matches(":popover-open") ? close(true) : open())}
    onkeydown={(e) => {
      if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        e.preventDefault();
        open(e.key === "ArrowUp");
      }
    }}
  >
    <span>{label}</span><Icon name="chevron-down" />
  </button>
  <span id="{id}-menu" class="image-menu-popup" popover="auto" role="menu" tabindex="-1" aria-labelledby={id} bind:this={popup} onbeforetoggle={(e) => e.newState === "closed" && (expanded = false)} onclick={pick} onkeydown={keys}>
    {@render children()}
  </span>
</span>
