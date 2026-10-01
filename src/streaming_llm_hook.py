"""
StreamingLLM eviction baseline hook — teacher-forced, pure PyTorch.

Query i attends only to the sink tokens (j < sink_len) and a sliding recent
window (i - window < j <= i); every other key is evicted. For a fair comparison
with a rank-R compressed cache, the window is sized to the same KV bytes:
per head, (T*R + R*d) extra floats -> T*R/d + R extra exact tokens.
"""

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F


def matched_window(rank: int, ctx_len: int, head_dim: int, window_len: int) -> int:
    """Recent-window length with the same per-head KV bytes as a rank-`rank` compressed cache."""
    return window_len + round(ctx_len * rank / head_dim) + rank


def _make_streaming_llm_fn(windows: List[int], sink_len: int):
    def _attn(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0,
              **kwargs) -> Tuple[torch.Tensor, None]:
        W = windows[min(getattr(module, "layer_idx", 0), len(windows) - 1)]
        T = query.shape[2]
        i = torch.arange(T, device=query.device)[:, None]
        j = torch.arange(T, device=query.device)[None, :]
        keep = (j <= i) & ((j < sink_len) | (j > i - W))
        out = F.scaled_dot_product_attention(query, key, value, attn_mask=keep, scale=scaling,
                                             enable_gqa=query.shape[1] != key.shape[1])
        return out.transpose(1, 2), None   # (batch, T, n_q_heads, d)
    return _attn


def patch_model_streaming_llm(model, ranks: Union[int, List[int]], warmup_len: int = 512,
                              sink_len: int = 64, window_len: int = 256, ctx_len: int = 2048,
                              key: str = "streaming_llm", **_) -> str:
    """
    Registers StreamingLLM attention with per-layer windows memory-matched to
    `ranks`. warmup_len is unused (no basis). Returns the original
    _attn_implementation; restore with model_hooks.unpatch_model.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    cfg = model.config
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    ranks = [ranks] if isinstance(ranks, int) else list(ranks)
    windows = [matched_window(r, ctx_len, d, window_len) for r in ranks]
    ALL_ATTENTION_FUNCTIONS.register(key, _make_streaming_llm_fn(windows, sink_len))
    original = model.config._attn_implementation
    model.config._attn_implementation = key
    return original
