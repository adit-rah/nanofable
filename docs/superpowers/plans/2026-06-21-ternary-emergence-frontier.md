# Ternary vs FP16 Emergence Frontier — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Save the executed copy to `docs/superpowers/plans/2026-06-21-ternary-emergence-frontier.md`** (this scratch copy lives in `~/.claude/plans/` because of plan-mode write restrictions).

**Goal:** Build the full research harness that draws the coherence-vs-total-bytes frontier for fp16 vs ternary (1.58-bit) tiny LMs on TinyStories, applies a pre-registered capability gate, and reports the smallest capable English model in bytes.

**Architecture:** Decoder-only transformer (RMSNorm + RoPE + SwiGLU, tied embeddings/head, no biases) with a global `precision` flag that swaps every in-block `nn.Linear` for a `BitLinear` (latent fp32 weights, per-tensor absmean ternary forward, straight-through-estimator backward). A custom 4k BPE tokenizer keeps the embedding table small so ternary block savings dominate. A seedable, hard-kill-resumable training loop logs per-run CSVs; a frozen eval harness (200 prefixes, Qwen2.5-7B-Instruct judge, committed rubric) computes the gate; a plotting script and idempotent sweep entrypoint produce the headline deliverable.

**Tech Stack:** Python, PyTorch, HuggingFace `tokenizers` + `datasets`, `transformers` + `bitsandbytes` (judge), `matplotlib`, `pytest`.

## Global Constraints (frozen — copied verbatim into `docs/frozen_config.md` and never edited after first sweep run)

- **Architecture (fixed across the whole sweep):** decoder-only transformer, RMSNorm (pre-norm), RoPE, SwiGLU MLP, tied input embedding / LM head, **all linears `bias=False`**, MLP hidden = `8/3 · n_embd` rounded to a multiple of 64.
- **Context length:** `ctx = 512` for **every** tier (resolves the §4/§5 spec conflict in favor of §4 "hold fixed"; chosen by user).
- **Tiers (only size varies):** tiny `L4 d128 h4`, small `L6 d256 h8`, medium `L8 d384 h8`, large `L8 d512 h8`.
- **Precision (independent variable):** `fp16` (all linears `nn.Linear`, run in bf16/fp16 autocast) vs `ternary` (in-block linears → `BitLinear`). Embeddings, final RMSNorm, LM head, and all RMSNorm gains stay fp16 in **both** arms.
- **Tokenizer:** HF `tokenizers` BPE, **vocab = 4096**, trained once on TinyStories train split, committed under `artifacts/tokenizer/`. Never retrained mid-sweep.
- **Token budget:** **500M tokens/run**, identical for every run.
- **Optimizer/schedule (fixed across sweep):** AdamW (β1=0.9, β2=0.95, weight_decay=0.1, eps=1e-8), grad-clip 1.0, cosine decay to 10% of peak after 3% linear warmup, **peak LR = 3e-4 for all runs**, **tokens-per-optimizer-step = 65536** (via grad accumulation; micro-batch tuned only to fit memory, never changes optimization). ≈ 7630 optimizer steps/run.
- **Seeds:** `{0, 1}` (≥2 per spec). Same seed protocol seeds data order, init, and dropout.
- **Ternary layer:** `w_q = scale · round(clip(w_latent/scale, -1, 1))`, `scale = mean(|w_latent|)` **per-tensor**; STE backward (identity through round/clip, gradient clipped to where `|w_latent/scale|≤1`). Only in-block linears quantized; activations stay fp16 (v1).
- **Judge (frozen, committed):** `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0), loaded 4-bit by default (16GB GPU); pluggable backend, but this is the frozen default. Rubric, 200 prefixes, prompt all committed before any sweep run.
- **Capability gate (frozen):** capable iff **mean judge ≥ 4.0/5** over the fixed 200 prefixes (pooled across seeds) **AND** val PPL ≤ `1.5 × best_fp16_val_ppl`. Tier-separation claims count only if the 95% CI on the mean judge score does **not** straddle 4.0.

## Open items to confirm with the user during execution (do not silently resolve)

1. **§6 byte formula omits RMSNorm gains/biases.** The spec's `count_bytes` formula has three terms (ternary blocks + fp16 embed/head + scales) and does not mention norm weights. We use `bias=False` (no bias bytes) and **include RMSNorm gains at fp16** as a separately-reported line for honesty. This is negligible (~`n_embd` per norm) but is a deviation-by-addition from the literal §6 formula — confirm before freezing the headline number. (Plan defaults to: include norms, report the delta.)
2. **Peak LR fixed vs ternary stability (spec §12).** We hold LR=3e-4 across the sweep for apples-to-apples. If a ternary run collapses, that is a *reported finding*; any LR change must be applied to **all 16 runs** (re-run fp16 too), never per-run. Confirm this policy.
3. **BitLinear scale granularity.** Spec allows per-tensor or per-output-channel; we chose per-tensor (matches cited BitNet practice, 1 fp16 scale/layer). Per-channel is left as a stretch toggle.

---

## File Structure

```
specs/                         # frozen spec (exists)
docs/
  frozen_config.md             # Global Constraints, committed before sweep
  design_notes.md              # rationale: arch, ctx decision, LR policy, byte-accounting note
  byte_accounting.md           # WHAT count_bytes computes + the §6 derivation
  writeup.md                   # final 1–2 page finding (Definition of Done #2)
src/nanofable/
  config.py                    # ModelConfig, TierConfig, TIERS dict
  tokenizer.py                 # train + load BPE(4k)
  rope.py                      # RoPE precompute + apply
  bitlinear.py                 # BitLinear + STE autograd fn
  model.py                     # RMSNorm, SwiGLU, Block, Transformer, build_model(precision)
  bytes.py                     # count_bytes(model, precision)
  data.py                      # tokenize TinyStories -> memmap shards; batch iterator
  train.py                     # seedable loop, checkpoint/resume, CSV logging, val PPL
  flops.py                     # flops(n_params, n_tokens) = 6*N*T
  generate.py                  # completion sampler from a checkpoint
eval/
  rubric.md                    # FROZEN rubric text
  prefixes.jsonl               # FROZEN 200 prefixes
  judge_prompt.md              # FROZEN judge prompt template
  judge.py                     # JudgeBackend protocol; LocalQwenJudge, AnthropicJudge
  calibration.md               # FROZEN calibration + judge-reliability results
  gate.py                      # capability_gate(...) predicate + CI
  run_eval.py                  # generate completions -> judge -> per-config scores CSV
scripts/
  build_tokenizer.py
  build_dataset.py
  make_prefixes.py             # one-shot generator for prefixes.jsonl (then committed)
  run_calibration.py           # one-shot; writes eval/calibration.md inputs
  run_sweep.py                 # idempotent 4x2x2 matrix
  plot_frontier.py             # coherence-vs-bytes headline plot
runs/<tier>_<precision>_<seed>/  # per-run: ckpt_latest.pt, ckpt_step*.pt, metrics.csv, meta.json
artifacts/tokenizer/           # committed tokenizer.json
tests/                         # mirrors src/ + eval/
```

Phases are ordered by dependency. Each phase ends with working, independently testable software. Within a task, code shown is real; "standard" modules give exact signatures + test code rather than full literal source to keep the plan scannable — implementers fill the body to satisfy the shown tests.

---

## Phase 0 — Scaffold, config, docs

### Task 0.1: Repo scaffold + deps + frozen config doc
**Files:** Create `pyproject.toml` (or `requirements.txt`), `src/nanofable/__init__.py`, `docs/frozen_config.md`, `docs/design_notes.md`, `tests/__init__.py`.

- [ ] **Step 1:** Create `requirements.txt`: `torch`, `tokenizers`, `datasets`, `transformers`, `accelerate`, `bitsandbytes`, `matplotlib`, `numpy`, `pytest`. Pin majors.
- [ ] **Step 2:** Write `docs/frozen_config.md` = the **Global Constraints** section above verbatim. Write `docs/design_notes.md` capturing the three Open Items and the ctx-conflict resolution.
- [ ] **Step 3:** `pytest -q` (collects nothing yet) → exit 0. **Commit:** `chore: scaffold repo + frozen config doc`.

### Task 0.2: Config objects
**Files:** Create `src/nanofable/config.py`, `tests/test_config.py`.
**Interfaces — Produces:**
```python
@dataclass(frozen=True)
class ModelConfig:
    n_layer: int; n_embd: int; n_head: int; ctx: int = 512; vocab: int = 4096
    @property
    def mlp_hidden(self) -> int: ...   # round(8/3 * n_embd) up to multiple of 64
TIERS: dict[str, ModelConfig]          # "tiny","small","medium","large" per Global Constraints
```
- [ ] **Step 1 (failing test):**
```python
def test_tiers_fixed_ctx_and_vocab():
    for c in TIERS.values():
        assert c.ctx == 512 and c.vocab == 4096
def test_tiny_shape_and_mlp_hidden():
    t = TIERS["tiny"]
    assert (t.n_layer, t.n_embd, t.n_head) == (4, 128, 4)
    assert t.mlp_hidden % 64 == 0
def test_head_divides_embd():
    for c in TIERS.values(): assert c.n_embd % c.n_head == 0
```
- [ ] **Step 2:** Run → FAIL (ImportError). **Step 3:** Implement. **Step 4:** Run → PASS. **Step 5:** Commit `feat: model/tier config`.

---

## Phase 1 — Tokenizer

### Task 1.1: Train + load 4k BPE
**Files:** Create `src/nanofable/tokenizer.py`, `scripts/build_tokenizer.py`, `tests/test_tokenizer.py`.
**Interfaces — Produces:**
```python
def train_tokenizer(texts: Iterable[str], vocab_size: int = 4096, save_path: str) -> None
def load_tokenizer(path: str) -> Tokenizer        # HF tokenizers Tokenizer
# special tokens: <|bos|>, <|eos|>, <|pad|>  (ids 0,1,2); ByteLevel pre-tokenizer
```
- [ ] **Step 1 (failing test):** train on a tiny in-memory corpus, assert roundtrip `decode(encode(s)) == s` for an ASCII story, `vocab_size <= 4096`, and that `<|bos|>/<|eos|>` ids are stable (0,1).
- [ ] **Step 2:** FAIL. **Step 3:** Implement BPE trainer (ByteLevel, special tokens). **Step 4:** PASS.
- [ ] **Step 5:** `scripts/build_tokenizer.py` streams `datasets.load_dataset("roneneldan/TinyStories", split="train")["text"]` into `train_tokenizer`, saves `artifacts/tokenizer/tokenizer.json`. Run it; commit the artifact.
- [ ] **Step 6:** Commit `feat: 4k BPE tokenizer + committed artifact`.

---

## Phase 2 — Model + BitLinear

### Task 2.1: RoPE
**Files:** Create `src/nanofable/rope.py`, `tests/test_rope.py`.
**Interfaces — Produces:** `build_rope_cache(head_dim, ctx, base=10000) -> (cos, sin)`; `apply_rope(q, k, cos, sin) -> (q,k)` (shapes `[B,H,T,Dh]`).
- [ ] **Step 1 (failing test):** rotation preserves norm (`‖apply_rope(q)‖ ≈ ‖q‖`); relative-position property: dot product of rotated q_t,k_s depends only on `t-s` for a constant vector. **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS. **Step 5:** Commit `feat: RoPE`.

### Task 2.2: BitLinear with STE  *(spec non-negotiable — full code)*
**Files:** Create `src/nanofable/bitlinear.py`, `tests/test_bitlinear.py`.
**Interfaces — Produces:** `class BitLinear(nn.Module)` with `weight` latent param (fp32), `bias=None`, `.quantized_weight()` returning `w_q` and `.scale()` returning the fp16 scalar; forward uses STE.
- [ ] **Step 1 (failing tests):**
```python
def test_forward_is_ternary_times_scale():
    m = BitLinear(8, 8, bias=False)
    wq = m.quantized_weight()
    s = m.scale()
    assert torch.unique((wq / s).round()).abs().max() <= 1  # values in {-1,0,1}
def test_ste_gradient_flows_to_latent():
    m = BitLinear(4, 4, bias=False)
    x = torch.randn(2, 4, requires_grad=False)
    m(x).sum().backward()
    assert m.weight.grad is not None and m.weight.grad.abs().sum() > 0
def test_scale_is_absmean():
    m = BitLinear(4, 4, bias=False)
    assert torch.allclose(m.scale(), m.weight.detach().abs().mean(), atol=1e-6)
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement:
```python
class _TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, scale):
        ctx.save_for_backward(w, scale)
        return scale * torch.round(torch.clamp(w / scale, -1, 1))
    @staticmethod
    def backward(ctx, g):
        w, scale = ctx.saved_tensors
        mask = (w.abs() <= scale).to(g.dtype)   # STE clipped to active region
        return g * mask, None

class BitLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, std=0.02)
        assert not bias
    def scale(self):  # per-tensor absmean
        return self.weight.detach().abs().mean().clamp_min(1e-8)
    def quantized_weight(self):
        return _TernarySTE.apply(self.weight, self.scale())
    def forward(self, x):
        return F.linear(x, self.quantized_weight())
```
- [ ] **Step 4:** PASS. **Step 5:** Commit `feat: BitLinear + STE`.

### Task 2.3: RMSNorm, SwiGLU, Block, Transformer, build_model
**Files:** Create `src/nanofable/model.py`, `tests/test_model.py`.
**Interfaces — Produces:**
```python
def build_model(cfg: ModelConfig, precision: Literal["fp16","ternary"]) -> Transformer
# Linear factory: precision=="ternary" -> BitLinear for q,k,v,o,gate,up,down; else nn.Linear(bias=False)
# Embedding (vocab,n_embd) tied to LM head; final RMSNorm fp16; RMSNorm in each block fp16
class Transformer(nn.Module):
    def forward(self, idx, targets=None) -> (logits, loss|None)  # CE loss, ignore_index=pad
```
- [ ] **Step 1 (failing tests):**
```python
def test_forward_shapes():
    m = build_model(TIERS["tiny"], "fp16")
    idx = torch.randint(0, 4096, (2, 16))
    logits, loss = m(idx, idx)
    assert logits.shape == (2, 16, 4096) and loss.ndim == 0
def test_precision_swaps_linears():
    from nanofable.bitlinear import BitLinear
    mt = build_model(TIERS["tiny"], "ternary")
    assert any(isinstance(x, BitLinear) for x in mt.modules())
    mf = build_model(TIERS["tiny"], "fp16")
    assert not any(isinstance(x, BitLinear) for x in mf.modules())
def test_head_tied_to_embedding():
    m = build_model(TIERS["tiny"], "fp16")
    assert m.lm_head.weight is m.tok_emb.weight
def test_embeddings_never_quantized():
    mt = build_model(TIERS["tiny"], "ternary")
    from nanofable.bitlinear import BitLinear
    assert not isinstance(mt.tok_emb, BitLinear)
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement (pre-norm block: `x += attn(norm(x))`, `x += mlp(norm(x))`; causal mask; RoPE on q,k; SwiGLU `down(silu(gate(x))*up(x))`; tie head). **Step 4:** PASS. **Step 5:** Commit `feat: transformer + precision-switchable build_model`.

---

## Phase 3 — Byte accounting  *(spec non-negotiable — headline depends on it)*

### Task 3.1: count_bytes
**Files:** Create `src/nanofable/bytes.py`, `docs/byte_accounting.md`, `tests/test_bytes.py`.
**Interfaces — Produces:**
```python
def count_bytes(model: Transformer, precision: str) -> dict
# keys: block_bytes, embed_head_bytes, scale_bytes, norm_bytes, total
```
Accounting (per §6, + flagged norm line from Open Item 1):
- in-block linear weights: ternary → `n_w * 1.58/8`; fp16 → `n_w * 2`.
- embeddings+tied head: `2 * vocab * n_embd` (counted once).
- scales: ternary → `num_quantized_layers * 2` (per-tensor fp16); fp16 → 0.
- norms (RMSNorm gains, fp16): `sum(numel) * 2` — separate line.
- `total = block + embed_head + scale + norm`.
- [ ] **Step 1 (failing tests):**
```python
def test_fp16_total_is_two_bytes_per_param():
    m = build_model(TIERS["tiny"], "fp16")
    n = sum(p.numel() for p in m.parameters()) - m.tok_emb.weight.numel()  # head tied, count once
    assert count_bytes(m, "fp16")["total"] == 2 * n
def test_ternary_block_uses_1p58():
    mt = build_model(TIERS["small"], "ternary")
    from nanofable.bitlinear import BitLinear
    nq = sum(l.weight.numel() for l in mt.modules() if isinstance(l, BitLinear))
    assert count_bytes(mt, "ternary")["block_bytes"] == nq * 1.58 / 8
def test_ternary_smaller_than_fp16_with_4k_vocab():
    cfg = TIERS["medium"]
    bt = count_bytes(build_model(cfg, "ternary"), "ternary")["total"]
    bf = count_bytes(build_model(cfg, "fp16"), "fp16")["total"]
    assert bt < bf   # sanity: the whole experiment depends on this being true at 4k vocab
def test_scale_bytes_counts_every_quantized_layer():
    mt = build_model(TIERS["tiny"], "ternary")
    from nanofable.bitlinear import BitLinear
    nlayers = sum(isinstance(l, BitLinear) for l in mt.modules())
    assert count_bytes(mt, "ternary")["scale_bytes"] == nlayers * 2
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement; write `docs/byte_accounting.md` deriving each term and stating the norm-inclusion decision (Open Item 1). **Step 4:** PASS. **Step 5:** Commit `feat: tested count_bytes (§6)`.

---

## Phase 4 — Data pipeline

### Task 4.1: Tokenize TinyStories to memmap shards + batch iterator
**Files:** Create `src/nanofable/data.py`, `scripts/build_dataset.py`, `tests/test_data.py`.
**Interfaces — Produces:**
```python
def build_token_memmap(split: str, tokenizer, out_path: str) -> int  # returns n_tokens (uint16)
def batch_iterator(memmap_path, ctx, tokens_per_step, seed) -> Iterator[(x,y)]  # deterministic
```
- [ ] **Step 1 (failing tests):** build a memmap from a small fake corpus; assert dtype uint16 (vocab 4k fits), every token `< 4096`, and that `batch_iterator` with a fixed seed yields **identical** first batch across two calls (determinism = apples-to-apples). x,y shapes `[micro_b, ctx]`, y = x shifted by 1.
- [ ] **Step 2:** FAIL. **Step 3:** Implement (documents joined with `<|eos|>`; contiguous packing; sampler RNG seeded; grad-accum handled by yielding `tokens_per_step//ctx` rows per step). **Step 4:** PASS.
- [ ] **Step 5:** `scripts/build_dataset.py` builds train + validation memmaps under `artifacts/data/`. Run; do **not** commit large memmaps (gitignore). Commit `feat: data pipeline` + `.gitignore`.

---

## Phase 5 — Training loop (seedable, hard-kill resumable, CSV, PPL)

### Task 5.1: FLOPs + seeding helpers
**Files:** Create `src/nanofable/flops.py`, add `set_seed` to `train.py`, `tests/test_flops.py`.
**Interfaces — Produces:** `flops(n_params: int, n_tokens: int) -> float  # 6*N*T`; `set_seed(seed)` seeds python/numpy/torch+cuda.
- [ ] **Step 1 (failing test):** `flops(10, 100) == 6000`. **Step 2:** FAIL. **Step 3:** Implement. **Step 4:** PASS. **Step 5:** Commit `feat: flops + seeding`.

### Task 5.2: Checkpoint save/load (atomic, survives hard kill)
**Files:** Create checkpoint helpers in `src/nanofable/train.py`, `tests/test_checkpoint.py`.
**Interfaces — Produces:**
```python
def save_checkpoint(run_dir, step, tokens_seen, model, opt, sched, rng_state) -> None  # atomic
def load_latest(run_dir) -> dict | None   # reads ckpt_latest.pt; None if absent
```
- [ ] **Step 1 (failing tests):** save then `load_latest` restores `step`, `tokens_seen`, and model state_dict bytewise-equal; **atomicity:** write to `ckpt_latest.pt.tmp` then `os.replace` (test that a leftover `.tmp` is ignored and the last good `ckpt_latest.pt` still loads).
- [ ] **Step 2:** FAIL. **Step 3:** Implement (`torch.save` to tmp + `os.replace`; also periodic `ckpt_step{N}.pt`; store model+opt+sched+RNG+step+tokens_seen). **Step 4:** PASS. **Step 5:** Commit `feat: atomic checkpoint/resume primitives`.

### Task 5.3: Eval (held-out val PPL)
**Files:** Add `evaluate_ppl` to `train.py`, `tests/test_eval_ppl.py`.
**Interfaces — Produces:** `evaluate_ppl(model, val_memmap, ctx, n_batches) -> float`.
- [ ] **Step 1 (failing test):** on a tiny model+memmap, PPL is finite, `>1`, and `== exp(mean_ce)` for a hand-computed 2-batch case. **Step 2:** FAIL. **Step 3:** Implement (no-grad, autocast, mean CE → exp). **Step 4:** PASS. **Step 5:** Commit `feat: val perplexity`.

### Task 5.4: Training loop + CSV logging + resume integration
**Files:** Add `train_run(cfg, precision, seed, run_dir)` to `train.py`, `tests/test_train_smoke.py`.
**Interfaces — Produces:**
```python
def train_run(cfg, precision, seed, run_dir, total_tokens=500_000_000,
              tokens_per_step=65536, peak_lr=3e-4) -> None
# CSV runs/<...>/metrics.csv columns:
# step,tokens_seen,train_loss,val_loss,val_ppl,flops,wall_clock_s,lr,timestamp
```
- [ ] **Step 1 (failing tests):**
```python
def test_smoke_train_writes_csv_and_ckpt(tmp_path):
    train_run(TIERS["tiny"], "fp16", seed=0, run_dir=tmp_path,
              total_tokens=65536*3, tokens_per_step=65536)  # 3 steps
    assert (tmp_path/"metrics.csv").exists() and (tmp_path/"ckpt_latest.pt").exists()
def test_resume_continues_from_last_step(tmp_path):
    train_run(TIERS["tiny"],"fp16",0,tmp_path,total_tokens=65536*2,tokens_per_step=65536)
    s1 = last_step(tmp_path/"metrics.csv")
    train_run(TIERS["tiny"],"fp16",0,tmp_path,total_tokens=65536*4,tokens_per_step=65536)
    assert last_step(tmp_path/"metrics.csv") == 4 and last_step >= s1  # no restart from 0
def test_completion_marker_written(tmp_path):
    train_run(TIERS["tiny"],"fp16",0,tmp_path,total_tokens=65536*2,tokens_per_step=65536)
    assert (tmp_path/"DONE").exists()
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement: `set_seed`; build model+AdamW+cosine sched; `start = load_latest(run_dir)` (resume tokens_seen/step/RNG or fresh); grad-accum to `tokens_per_step`; clip 1.0; every `eval_every` steps log val PPL + append CSV row + `save_checkpoint`; on reaching `total_tokens` write `DONE` + final `ckpt_latest.pt`; write `meta.json` (cfg, precision, seed, total_bytes via `count_bytes`).
- [ ] **Step 4:** PASS. **Step 5 (hard-kill manual check):** start a tiny run in background, `kill -9` mid-run, restart same command, confirm CSV step increases monotonically (no row at step 0 second time). **Step 6:** Commit `feat: resumable training loop + CSV logging`.

---

## Phase 6 — Frozen eval artifacts + calibration  *(must be committed BEFORE any sweep run)*

### Task 6.1: Freeze rubric, prefixes, judge prompt
**Files:** Create `eval/rubric.md`, `eval/judge_prompt.md`, `scripts/make_prefixes.py`, `eval/prefixes.jsonl`.
- [ ] **Step 1:** Write `eval/rubric.md`: three axes grammar(0–5), consistency(0–5), completes-sensibly(0–5); per-completion score = mean of the three; rubric anchor descriptions for 0/2/4/5.
- [ ] **Step 2:** Write `eval/judge_prompt.md`: a fixed template that takes `{prefix}` + `{completion}`, instructs Qwen to return strict JSON `{"grammar":int,"consistency":int,"completes":int}`.
- [ ] **Step 3:** `scripts/make_prefixes.py` samples **exactly 200** held-out validation stories with a fixed seed, truncates each to its first ~40 tokens as the prefix, writes `eval/prefixes.jsonl` (`{"id","prefix","gold_continuation"}`). Run once.
- [ ] **Step 4:** Commit `eval(freeze): rubric + 200 prefixes + judge prompt` (these files are now immutable per spec §8).

### Task 6.2: Judge backend (pluggable; Qwen2.5-7B-Instruct default)
**Files:** Create `eval/judge.py`, `tests/test_judge.py`.
**Interfaces — Produces:**
```python
class JudgeBackend(Protocol):
    def score(self, prefix: str, completion: str) -> dict  # {"grammar","consistency","completes"}
class LocalQwenJudge(JudgeBackend):  # Qwen/Qwen2.5-7B-Instruct, load_in_4bit=True default
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", four_bit=True): ...
class AnthropicJudge(JudgeBackend):  # paid fallback for borderline configs
def parse_judge_json(text: str) -> dict  # robust JSON extraction, clamps to 0..5
```
- [ ] **Step 1 (failing tests):** `parse_judge_json` extracts scores from messy model text and clamps out-of-range to [0,5]; a `FakeJudge` implementing the protocol returns the right dict shape (no model download in unit tests). **Step 2:** FAIL. **Step 3:** Implement parser + backends (Local Qwen via `transformers`, 4-bit through `bitsandbytes`; chat template + `judge_prompt.md`). **Step 4:** PASS. **Step 5:** Commit `feat: pluggable judge backends`.

### Task 6.3: Calibration + judge-reliability pass (sweep-blind, one-time)
**Files:** Create `scripts/run_calibration.py`, `eval/calibration.md`.
- [ ] **Step 1:** `run_calibration.py` scores three reference sets through the frozen rubric+Qwen judge: (a) real TinyStories gold continuations, (b) TinyStories-33M completions, (c) degenerate text (shuffled/truncated/repetitive), over the 200 prefixes.
- [ ] **Step 2:** Compute: rank-ordering (a>b>c), **intra-judge std** (re-score same set ≥3×), and 95% CI width of the mean on N=200.
- [ ] **Step 3:** Write `eval/calibration.md`: confirm good refs clear 4.0, garbage well below, 4.0 sits in the discriminative band; report intra-judge std vs good-vs-bad gap; if std > gap, STOP and swap/upgrade judge before freezing. Commit `eval(freeze): calibration + judge reliability`.

### Task 6.4: Capability gate predicate
**Files:** Create `eval/gate.py`, `tests/test_gate.py`.
**Interfaces — Produces:**
```python
def mean_and_ci(scores: list[float]) -> tuple[float, float]   # mean, 95% half-width
def capability_gate(judge_scores, val_ppl, best_fp16_ppl) -> dict
# returns {"coherence_pass","ppl_pass","capable","mean","ci_low","ci_high","ci_straddles_4"}
```
- [ ] **Step 1 (failing tests):**
```python
def test_capable_requires_both_gates():
    assert capability_gate([5]*200, val_ppl=10, best_fp16_ppl=10)["capable"] is True
    assert capability_gate([5]*200, val_ppl=100, best_fp16_ppl=10)["capable"] is False  # ppl fail
    assert capability_gate([3]*200, val_ppl=10, best_fp16_ppl=10)["capable"] is False   # coh fail
def test_ci_straddle_flagged():
    g = capability_gate([4.0]*100 + [3.9,4.1]*50, val_ppl=10, best_fp16_ppl=10)
    assert "ci_straddles_4" in g
```
- [ ] **Step 2:** FAIL. **Step 3:** Implement (PPL threshold = `1.5*best_fp16_ppl`; CI via t/normal on N=200). **Step 4:** PASS. **Step 5:** Commit `feat: capability gate + CI`.

### Task 6.5: Eval runner (completions → judge → per-config scores)
**Files:** Create `src/nanofable/generate.py`, `eval/run_eval.py`, `tests/test_generate.py`.
**Interfaces — Produces:**
```python
def generate(model, tokenizer, prefix: str, max_new_tokens=200, seed=0) -> str
# run_eval.py: load run_dir checkpoint -> complete all 200 prefixes -> judge -> write
#   runs/<...>/eval.json {per_prefix scores, mean, ci_low, ci_high}
```
- [ ] **Step 1 (failing test):** `generate` on a tiny model returns a decodable string of ≤max_new_tokens, deterministic under fixed seed. **Step 2:** FAIL. **Step 3:** Implement sampler + `run_eval.py` glue. **Step 4:** PASS. **Step 5:** Commit `feat: completion generation + eval runner`.

---

## Phase 7 — Sweep orchestration (idempotent)

### Task 7.1: run_sweep
**Files:** Create `scripts/run_sweep.py`, `tests/test_sweep.py`.
**Interfaces — Produces:** `sweep_matrix() -> list[(tier,precision,seed)]` (16 entries); `run_sweep(only=None)` trains+evals each, **skipping any run dir with a `DONE` marker**.
- [ ] **Step 1 (failing tests):** `len(sweep_matrix()) == 16` (4×2×2); `run_sweep` skips a run whose dir already has `DONE` (monkeypatch `train_run` to record calls; pre-create one `DONE`, assert it's not called for that combo). **Step 2:** FAIL. **Step 3:** Implement (deterministic run-dir naming `runs/<tier>_<precision>_<seed>`; resume-safe; runs eval after train; tolerant of being killed/re-run). **Step 4:** PASS. **Step 5:** Commit `feat: idempotent run_sweep`.

---

## Phase 8 — Plotting + writeup (Definition of Done)

### Task 8.1: Frontier plot
**Files:** Create `scripts/plot_frontier.py`, `tests/test_plot.py`.
**Interfaces — Produces:** `collect_results(runs_dir) -> list[dict]` (tier, precision, seed, total_bytes, val_ppl, judge_mean, ci); `plot_frontier(results, out_png)` — coherence(y) vs total_bytes(log x), two curves, marker/tier, **annotate smallest capable point per curve**, title names global-min headline.
- [ ] **Step 1 (failing test):** `collect_results` reads two synthetic run dirs (`meta.json`+`eval.json`+`metrics.csv`) and returns rows with `total_bytes` and `judge_mean`; `plot_frontier` writes a non-empty PNG. **Step 2:** FAIL. **Step 3:** Implement (mean±std across seeds; gate applied via `eval/gate.py`; annotate smallest-bytes capable point). **Step 4:** PASS. **Step 5:** Commit `feat: coherence-vs-bytes frontier plot`.

### Task 8.2: Writeup
**Files:** Create `docs/writeup.md`.
- [ ] **Step 1:** After the sweep+plot run, write the 1–2 page finding: where ternary wins on bytes / where it crosses, the smallest-capable headline (or "tiers indistinguishable" if CIs straddle 4.0), the compute-axis caveat (ternary costs *more* to train), and the embedding-dominance lesson. **Step 2:** Commit `docs: experiment writeup`.

---

## Verification (end-to-end)

1. **Unit suite:** `pytest -q` — all phases' tests green (tokenizer roundtrip, BitLinear STE + ternary values, count_bytes fp16/ternary identities, data determinism, resume monotonicity, gate logic).
2. **Byte sanity:** `count_bytes(ternary) < count_bytes(fp16)` at every tier (test) — the whole experiment's premise.
3. **Hard-kill resume:** start a `tiny/fp16/seed0` run, `kill -9`, restart same command, confirm `metrics.csv` step is monotonic and never restarts at 0.
4. **Freeze check:** `eval/rubric.md`, `eval/prefixes.jsonl` (exactly 200 lines), `eval/judge_prompt.md`, `eval/calibration.md` all committed before any `run_sweep` invocation; `git log` shows them predating run commits.
5. **Smoke sweep:** `run_sweep` on a reduced token budget for `tiny` only (both precisions, both seeds) → 4 run dirs with `DONE`, `eval.json`, and a generated frontier PNG; re-running `run_sweep` is a no-op (idempotency).
6. **Full sweep (Kaggle, ≤12h sessions):** 16 runs at 500M tokens; `scripts/plot_frontier.py` emits the headline plot; `docs/writeup.md` states the result.

## Self-review notes
- Spec coverage: §5 arch/tiers→P2/0; §6 bytes→P3; §7 BitLinear→2.2; §8 metrics+gate+calibration→P5.3/P6; §9 resume+CSV→P5; §10 plot+writeup→P8; §13 components all mapped. Embedding-dominance (§5 gotcha) handled by 4k vocab (Task 1.1) + byte test `test_ternary_smaller_than_fp16`.
- Three spec ambiguities surfaced as **Open Items**, not silently resolved (norm bytes, LR policy, scale granularity); ctx conflict resolved by user (512).
- Type consistency: `count_bytes` keys, `capability_gate` return dict, `JudgeBackend.score` shape, and `metrics.csv` columns are referenced identically across phases.
