"""Autoregressive completion sampler for eval.

Deterministic given `seed` (uses a dedicated torch.Generator). Generates up to
`max_new_tokens`, stopping early at the EOS token, and returns the decoded continuation
(the new tokens only).
"""

from __future__ import annotations

import torch

from .data import EOS_ID


@torch.no_grad()
def generate(model, tokenizer, prefix: str, max_new_tokens: int = 200, seed: int = 0,
             temperature: float = 1.0, top_k: int = 40) -> str:
    device = next(model.parameters()).device
    ctx = model.cfg.ctx
    gen = torch.Generator(device="cpu").manual_seed(seed)

    ids = tokenizer.encode(prefix).ids
    out_ids: list[int] = []
    cur = torch.tensor([ids], dtype=torch.long, device=device)

    was_training = model.training
    model.eval()
    for _ in range(max_new_tokens):
        logits, _ = model(cur[:, -ctx:])
        logits = logits[0, -1] / max(temperature, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, dim=-1).cpu()
        nxt = int(torch.multinomial(probs, 1, generator=gen).item())
        if nxt == EOS_ID:
            break
        out_ids.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
    if was_training:
        model.train()
    return tokenizer.decode(out_ids)
