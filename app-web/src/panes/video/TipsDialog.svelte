<script lang="ts">
  import Dialog from "../../components/Dialog.svelte";
  import { t } from "../../lib/i18n/i18n";

  let { h3, references, onclose }: { h3: boolean; references: boolean; onclose: () => void } = $props();
</script>

<Dialog title={t("Prompt tips")} {onclose}>
  {#snippet children(close)}
    <p>
      {#if h3 && references}
        {t("Use the six sections: subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape and non_diegetic_music. Refer to files as <Picture 1>, <Video 1> and <Audio 1>.")}
      {:else if h3}
        {t("Use integrated_multimodal_description, overall_soundscape and non_diegetic_music. Describe the shot, physical sound and score in those sections.")}
      {:else}
        {t("Write 4–8 concrete sentences. Describe the subject, motion, camera, lighting and sound. Put spoken words in quotes.")}
      {/if}
    </p>
    <button onclick={close}>{t("Close")}</button>
  {/snippet}
</Dialog>
