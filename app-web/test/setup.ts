// happy-dom does not match `:checked` on a selected <option>, which Svelte's `bind:value` on <select> reads.
const original = HTMLSelectElement.prototype.querySelector;
HTMLSelectElement.prototype.querySelector = function (this: HTMLSelectElement, selector: string) {
  return selector === ":checked" ? (this.options[this.selectedIndex] ?? null) : original.call(this, selector);
} as typeof original;
