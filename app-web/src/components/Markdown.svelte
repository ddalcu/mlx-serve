<script lang="ts">
  import { parseMarkdown, type Block, type Inline } from "../lib/markdown";
  import CodeBlock from "./CodeBlock.svelte";

  let { source }: { source: string } = $props();
  const blocks = $derived(parseMarkdown(source));
</script>

{#snippet inline(nodes: Inline[])}
  {#each nodes as node}
    {#if node.type === "text"}{node.text}
    {:else if node.type === "code"}<code>{node.text}</code>
    {:else if node.type === "br"}<br />
    {:else if node.type === "em"}<em>{@render inline(node.children)}</em>
    {:else if node.type === "strong"}<strong>{@render inline(node.children)}</strong>
    {:else if node.type === "del"}<s>{@render inline(node.children)}</s>
    {:else if node.type === "link"}<a href={node.href} title={node.title} rel="noopener noreferrer" target="_blank">{@render inline(node.children)}</a>
    {/if}
  {/each}
{/snippet}

{#snippet render(nodes: Block[], tight: boolean)}
  {#each nodes as block}
    {#if block.type === "paragraph"}
      {#if tight}{@render inline(block.children)}{:else}<p>{@render inline(block.children)}</p>{/if}
    {:else if block.type === "heading"}
      <svelte:element this={`h${block.level}`}>{@render inline(block.children)}</svelte:element>
    {:else if block.type === "code"}
      <CodeBlock lang={block.lang} text={block.text} />
    {:else if block.type === "quote"}
      <blockquote>{@render render(block.children, false)}</blockquote>
    {:else if block.type === "hr"}
      <hr />
    {:else if block.type === "list"}
      <svelte:element this={block.ordered ? "ol" : "ul"} start={block.ordered && block.start !== 1 ? block.start : undefined}>
        {#each block.items as item}<li>{@render render(item, block.tight)}</li>{/each}
      </svelte:element>
    {:else}
      <div class="markdown-table">
        <table>
          <thead>
            <tr>
              {#each block.head as cell, i}<th class={block.align[i] ? `align-${block.align[i]}` : undefined}>{@render inline(cell)}</th>{/each}
            </tr>
          </thead>
          {#if block.rows.length}
            <tbody>
              {#each block.rows as row}
                <tr>
                  {#each row as cell, i}<td class={block.align[i] ? `align-${block.align[i]}` : undefined}>{@render inline(cell)}</td>{/each}
                </tr>
              {/each}
            </tbody>
          {/if}
        </table>
      </div>
    {/if}
  {/each}
{/snippet}

{@render render(blocks, false)}
