# ds4 adapter

- The GGUF owns its vocabulary, trained window and native tool template. Host and engine capabilities must agree before a request is submitted.
- A Qwen n-gram tensor is disk-only even with resident weights. SSD weight streaming disables native MTP; SSD KV persistence is a separate feature.
- Image atoms come from vocabulary markers; RGB crosses the engine ABI with explicit ownership. Equal token ids do not authenticate different image pixels.
- Retained replies require complete visible prose and typed tool-argument equivalence. A matching suffix is never proof of a continuation.
- Native SSD payloads store exact text checkpoints, isolated by model/template/runtime/settings identity. Image-conditioned sessions and interrupted passes are not persisted.
- Cache I/O and graph operations run on the adapter worker. Save before replacing a valid session, prefer live continuation, then restore an exact disk prefix.
