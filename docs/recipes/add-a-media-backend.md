# Add a media backend

1. Copy `src/plugins/_template_media/` (contract: `src/abi/media.zig`). The modality decides the endpoint and the
   engine slot (`host/media_jobs/`); a new modality also needs `detectModality` / `modalityFromType`.
2. Bill the peak honestly (per stage when staged): admission reads the backend's estimate before the job starts.
3. Run on the inference thread only (post to the job queue); CPU-only stages go through the worker offload.
4. Support a bf16 pack and the quantized ones; parity against the reference with an env-gated cos + rms oracle.
5. An integration script in `tests/` for its HTTP surface, small canvas and few steps.
