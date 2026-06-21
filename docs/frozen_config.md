# Frozen Global Configuration

**Committed before the first sweep run. Never edited after.** These are the spec's
controlled variables (§4): everything here is held fixed across all 16 runs. The only
things that vary are **model size** (tier) and **precision** (fp16 vs ternary).

## Architecture (fixed across the whole sweep)
- Decoder-only transformer, **pre-norm** RMSNorm, **RoPE**, **SwiGLU** MLP.
- Tied input embedding / LM head.
- All linear layers `bias=False`.
- MLP hidden width = `8/3 · n_embd` rounded **up** to a multiple of 64.

## Context length
- `ctx = 512` for **every** tier. This resolves the §4/§5 conflict (§5 lists ctx 256 for
  the two smaller tiers; §4 says hold ctx fixed) in favor of §4. Decision recorded in
  `design_notes.md`.

## Tiers (only size varies)
| Tier   | n_layer | n_embd | n_head |
|--------|---------|--------|--------|
| tiny   | 4       | 128    | 4      |
| small  | 6       | 256    | 8      |
| medium | 8       | 384    | 8      |
| large  | 8       | 512    | 8      |

## Precision (the independent variable)
- **fp16:** all in-block linears are `nn.Linear`; run under bf16/fp16 autocast.
- **ternary:** in-block linears → `BitLinear` (latent fp32 weight, per-tensor absmean
  ternary forward, STE backward).
- In **both** arms: embeddings, final RMSNorm, LM head, and all RMSNorm gains stay fp16.
  Activations stay fp16 (v1).

## Tokenizer
- HF `tokenizers` BPE (ByteLevel), **vocab = 4096**, special tokens `<|bos|>=0`,
  `<|eos|>=1`, `<|pad|>=2`. Trained once on the TinyStories train split; committed under
  `artifacts/tokenizer/tokenizer.json`. Never retrained mid-sweep.

## Token budget
- **500M tokens / run**, identical for every run.

## Optimizer / schedule (fixed across sweep)
- AdamW: β1=0.9, β2=0.95, weight_decay=0.1, eps=1e-8.
- Grad-clip 1.0.
- Cosine decay to 10% of peak after 3% linear warmup.
- **Peak LR = 3e-4 for all runs** (held fixed; a ternary collapse is a reported finding,
  not a reason to retune one arm — see `design_notes.md`).
- **Tokens per optimizer step = 65536** (via grad accumulation; micro-batch tuned only to
  fit memory, never changes the optimization). ≈ 7630 optimizer steps / run.

## Seeds
- `{0, 1}` (≥2 per spec §4). The seed seeds data order, init, and dropout.

## Ternary layer (`BitLinear`)
- `w_q = scale · round(clip(w_latent / scale, -1, 1))`, `scale = mean(|w_latent|)`,
  **per-tensor** (one fp16 scale per layer).
- STE backward: identity through round/clip, gradient masked to the active region
  (`|w_latent| ≤ scale`).
- Only in-block linears quantized; activations fp16 (v1).

## Byte accounting (spec §6 — literal, three terms)
`total_bytes = block_bytes + embed_head_bytes + scale_bytes`
- in-block linear weights: ternary `n_w · 1.58/8`; fp16 `n_w · 2`.
- embeddings + tied head: `2 · vocab · n_embd` (counted once).
- per-layer ternary scales: `num_quantized_layers · 2` (fp16); fp16 arm: 0.
- **RMSNorm gains are NOT counted** (matches the literal §6 formula; decision in
  `design_notes.md`).

## Judge (frozen, committed)
- `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0), loaded 4-bit by default on a 16GB GPU.
  Pluggable backend, but this is the frozen default.
- Rubric, 200 prefixes, and judge prompt committed before any sweep run
  (`eval/rubric.md`, `eval/prefixes.jsonl`, `eval/judge_prompt.md`).

## Capability gate (frozen)
Capable iff **both**:
- **Coherence:** mean judge score **≥ 4.0 / 5** over the fixed 200 prefixes (pooled across
  seeds).
- **Perplexity:** val PPL **≤ 1.5 × best fp16 val PPL** in the sweep.

Tier-separation claims count **only** if the 95% CI on the mean judge score does **not**
straddle 4.0; otherwise report tiers as indistinguishable.
