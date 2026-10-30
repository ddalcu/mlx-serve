# Add an engine

An engine owns a runtime (ds4, llama.cpp, the ANE); the host talks to it only through `src/abi/engine.zig`.

1. Add `src/engines/<name>/` with an adapter over the runtime; a runtime with its own thread reuses
   `src/engines/own_thread/`.
2. Implement the table: open, the request lifecycle, `Caps` of the opened model, every fallible call a `Verdict`.
3. Pass the contract: `src/abi/contract_test.zig` and `src/abi/exactness_test.zig` run every engine;
   `src/testing/fake_engine.zig` is the smallest example.
4. Pick it in the dispatch (`--engine`, the GGUF header rules in the root `AGENTS.md`).
