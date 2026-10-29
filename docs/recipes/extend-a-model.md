# Extend a model

A new size, variant or block of a served family belongs in that family's plugin.

- Geometry comes from the checkpoint, never a constant: once a second size exists, read it per pack.
- A new `model_type` spelling: add it to the plugin's `config.zig` claims and a fixture config; `*_text` siblings
  collapse to the base type.
- A new block: if a second plugin can use it, put it in `src/sdk/blocks/` with its parity test; otherwise in the
  plugin's folder.
- A new quant layout of the same weights: a quant plugin (`_template_quant`) or `src/sdk/quant/`, never a branch in
  the model's forward.
- Re-run the plugin's proof (`zig build test -Dtest-filter=<name>`) and its greedy bytes on its gate model.
