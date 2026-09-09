# Append-only Qwen media history

Qwen vision requests retain supported historical image/video blocks in their original messages. Owned placeholders are inserted before chat templating; ordered pixel digests and token spans limit prefix reuse at the first changed media block. Full-history M-RoPE is computed from that same layout.

Each model owns separate bounded CPU preprocessing and projected-image caches (256 MiB / 64 entries each). GPU handles are evaluated, shared and freed on the inference thread. Appending an image can reuse state after earlier images; edits, removal and reordering invalidate the divergent suffix.

## Limits

- Post-image prefix state is RAM-only. When a disk tier is enabled, lookup may restore text strictly before the first media span; media-bearing state is not committed to SSD. Restart, eviction and early edits can replay history, and warm hybrid requests can replay a short checkpoint tail.
- Video embeddings are not cached. Images and video in separate turns work; mixing both in one message is rejected until cross-modality order is represented.
- Incomplete media, unsupported Qwen audio and literal image/video protocol-control tokens are rejected rather than producing a blind answer. Ordinary role-marker text is allowed.
- Stored Responses histories omit pixels. Image-bearing `previous_response_id` continuations (HTTP, WebSocket and compact) must resend full history instead.

## Behavioral checks

Run `zig build test` and, against an already-serving Qwen vision model, `python3 tests/test_media_history.py --help`, `python3 tests/test_media_surfaces.py --help` and `python3 tests/test_media_cancellation.py --help` for the live test arguments. These exercise append/edit/reorder/removal, historical recall, malformed input, API error termination, cancellation and request isolation. Live tests submit requests and alter the server's cache contents; use a dedicated test instance.
