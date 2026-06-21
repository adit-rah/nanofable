"""Data pipeline: tokenize TinyStories to a flat uint16 memmap, then sample batches.

Documents are joined with `<|eos|>` and packed contiguously. The batch iterator is fully
deterministic given its seed — apples-to-apples across the sweep depends on it.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
import torch

EOS_ID = 1


def build_token_memmap(texts: Iterable[str], tokenizer, out_path: str) -> int:
    """Tokenize `texts`, append `<|eos|>` after each, write uint16 to `out_path`.

    Returns the total number of tokens written. Writes incrementally so the full
    corpus need not fit in memory.
    """
    eos = tokenizer.token_to_id("<|eos|>")
    if eos is None:
        eos = EOS_ID
    n = 0
    with open(out_path, "wb") as f:
        for text in texts:
            ids = tokenizer.encode(text).ids
            ids.append(eos)
            arr = np.asarray(ids, dtype=np.uint16)
            f.write(arr.tobytes())
            n += arr.size
    return n


def batch_iterator(
    memmap_path: str, ctx: int, tokens_per_step: int, seed: int
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield (x, y) batches of shape [tokens_per_step // ctx, ctx], y = x shifted by 1.

    Deterministic: a fresh RNG seeded with `seed` means two iterators with the same seed
    yield identical sequences.
    """
    data = np.memmap(memmap_path, dtype=np.uint16, mode="r")
    rows = tokens_per_step // ctx
    max_start = data.shape[0] - ctx - 1
    if max_start <= 0:
        raise ValueError("memmap too small for the requested ctx")
    rng = np.random.default_rng(seed)
    while True:
        starts = rng.integers(0, max_start, size=rows)
        x = np.stack([data[s : s + ctx] for s in starts]).astype(np.int64)
        y = np.stack([data[s + 1 : s + ctx + 1] for s in starts]).astype(np.int64)
        yield torch.from_numpy(x), torch.from_numpy(y)
