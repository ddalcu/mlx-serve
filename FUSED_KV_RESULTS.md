# Fused KV-attention measurement — 2026-05-16 15:57

| Architecture | Config | prefill tok/s | decode tok/s |
|---|---|---|---|
| Gemma 4 31B (4-bit weights) | dense+off | 19.2 | 24.3 |
| Gemma 4 31B (4-bit weights) | dense+8 | 18.4 | 23.2 |
| Gemma 4 31B (4-bit weights) | fused+8 | 18.5 | 23.4 |
| Qwen 3.6 27B dense (4-bit weights) | dense+off | 15.7 | 25.8 |
| Qwen 3.6 27B dense (4-bit weights) | dense+8 | 15.4 | 25.9 |
| Qwen 3.6 27B dense (4-bit weights) | fused+8 | 15.3 | 25.7 |
