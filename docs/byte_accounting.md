# Byte Accounting (`count_bytes`)

**What it is for:** the x-axis of the headline plot is *total model size in bytes*, packed
honestly. `count_bytes(model, precision)` (`src/tinychat/bytes.py`) is the single tested
function that defines it. The whole result depends on it being correct, so it is pinned to
spec §6 verbatim and covered by `tests/test_bytes.py`.

## The formula (spec §6, literal — three terms)

```
total_bytes = block_bytes + embed_head_bytes + scale_bytes
```

| Term | fp16 arm | ternary arm |
|------|----------|-------------|
| `block_bytes` (in-block linear weights: q,k,v,o,gate,up,down) | `n_w · 2` | `n_w · 1.58/8` (packed ternary) |
| `embed_head_bytes` (embeddings + tied LM head, counted once) | `2 · vocab · n_embd` | `2 · vocab · n_embd` |
| `scale_bytes` (per-layer absmean scale, fp16) | `0` | `num_quantized_layers · 2` |

"In-block linears" are exactly the layers inside `model.blocks`; the tied LM head is **not**
double-counted (it shares storage with the embedding, already in `embed_head_bytes`).

## Decision: RMSNorm gains are NOT counted

§6's formula has these three terms and does not mention RMSNorm gains. Per the user decision
(design_notes.md item 2) we count `count_bytes` **literally** — norm gains are excluded. They
are negligible (~`n_embd` per norm), fp16 in both arms, and identical across precision, so they
cannot affect the fp16-vs-ternary comparison; excluding them keeps the headline number matched
to the frozen formula. The returned dict has exactly the keys
`{block_bytes, embed_head_bytes, scale_bytes, total}` — no `norm` key (asserted by a test).

## Why the small vocab matters (sanity numbers at vocab=4096, ctx=512)

| Tier | fp16 total | ternary total | embed+head | ternary/fp16 |
|------|-----------:|--------------:|-----------:|-------------:|
| tiny   |  2.75 MB |  1.22 MB | 1.05 MB | 0.442 |
| small  | 11.73 MB |  3.05 MB | 2.10 MB | 0.260 |
| medium | 31.46 MB |  5.94 MB | 3.15 MB | 0.189 |
| large  | 55.57 MB |  9.27 MB | 4.19 MB | 0.167 |

The embedding/head table stays small (e.g. 4.19 MB of the large tier's 55.57 MB), so the
transformer blocks dominate and the ternary savings are real and grow with size. With a 50k
GPT-2 vocab the embed/head term would dwarf the blocks and the ternary advantage would nearly
vanish — exactly the failure §5 warns about. `test_ternary_smaller_than_fp16_with_4k_vocab`
guards this at every tier.
