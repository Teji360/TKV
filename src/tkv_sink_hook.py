"""
TKV+Sink attention hook — batched pure PyTorch, teacher-forced.

Simulates the streaming TKV+Sink cache in a single forward pass. For each
layer and KV head:
  - Frozen basis: SVD of K[:T0] and V[:T0] (T0 = warmup_len; causal — only
    the first T0 tokens are used). K_hat = (K B_k^T) B_k, V_hat likewise.
  - Query i attends to key j <= i with EXACT K/V if
        j < sink_len                  (attention sink)
     or j > i - window_len            (per-query sliding recent window)
     or i < T0                        (warmup: basis not frozen yet, cache exact)
    and with the compressed K_hat/V_hat otherwise.

This matches the streaming semantics of sinked_add_token + tkv_add_token
(kv_state.py) with a basis frozen at T0: at step i the recent window holds
[i-W+1, i], the middle holds [S, i-W] projected on the frozen basis.

Residual retention (retain_frac > 0): of the middle tokens [S, i-W] visible to
query i, the fraction retain_frac with the largest key residual
||k - B^T B k|| (pre-RoPE if pre_rope) is kept EXACT as well. Streaming
equivalent: a heap of the top-residual evicted tokens; the residual is free
(it is computed with the projection). Since RoPE is a rotation, a compressed
key's logit error obeys |q.k - q.k_hat| <= ||q|| ||k - k_hat||, so keeping the
largest residuals minimizes the worst-case logit-error bound among compressed
tokens. retain_by="random" keeps a random subset of the same size (control).
Streaming realization: a fixed-capacity min-heap of M = retain_frac * (T - S - W)
evicted tokens keyed by residual always holds the top-M residuals seen so far.
The simulated set (top floor(retain_frac * |middle_i|) for query i) is a subset
of the heap's, and equal to it for the final query, so the simulation is
conservative for perplexity and exact for last-position retrieval.

center=True fits the bases to keys/values minus their warmup mean mu (one
d-vector per head): x_hat = mu + B^T B (x - mu). This removes a shared offset
(e.g. a key-projection bias) that would otherwise occupy a basis direction.

Diagnostics: set _DIAG["span"] = (start, end) (token positions, e.g. a needle)
before a forward pass; each layer/head appends, for the last query, the fraction
of that span kept exact and the mean residual percentile of the span among the
query's middle tokens to _DIAG["stats"].
"""

import sys
from typing import List, Tuple, Union

import torch

# pre_rope: the wrapped apply_rotary_pos_emb stashes the pre-RoPE keys and RoPE
# fn/args here; the attention hook of the same layer (called right after) uses them.
_ROPE_STASH = {}
_DIAG = {"span": None, "stats": []}


def _retained_mask(score: torch.Tensor, sink_len: int, window_len: int, frac: float) -> torch.Tensor:
    """
    (T, T) bool: query i keeps middle token j (sink_len <= j <= i - window_len)
    exactly if fewer than frac * |middle_i| middle tokens have a larger score.
    """
    T = score.shape[0]
    idx = torch.arange(T, device=score.device)
    mid = idx >= sink_len
    # C[p, j] = #{l <= p, l in middle : score_l > score_j}
    C = ((score[:, None] > score[None, :]) & mid[:, None]).to(torch.int32).cumsum(0)
    p = idx - window_len                                          # last middle position for query i
    n_mid = (p - sink_len + 1).clamp(min=0)
    Crow = C[p.clamp(min=0)]                                      # (T_i, T_j)
    j = idx[None, :]
    return (j >= sink_len) & (j <= p[:, None]) & (Crow < frac * n_mid[:, None])


def _make_tkv_sink_fn(ranks: List[int], warmup_len: int, sink_len: int, window_len: int,
                      pre_rope: bool = False, retain_frac: float = 0.0, retain_by: str = "residual",
                      center: bool = False):
    def _attn(
        module,
        query: torch.Tensor,   # (batch, n_q_heads, T, d)
        key: torch.Tensor,     # (batch, n_kv_heads, T, d)
        value: torch.Tensor,
        attention_mask=None,
        scaling=None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        layer_idx = getattr(module, "layer_idx", 0)
        rank = ranks[min(layer_idx, len(ranks) - 1)]
        batch, n_kv_heads, T, d = key.shape
        n_q_heads = query.shape[1]
        groups = n_q_heads // n_kv_heads
        device, dtype = query.device, query.dtype
        scale = scaling if scaling is not None else d ** -0.5
        T0 = min(warmup_len, T)
        r = min(rank, T0, d)

        k = key.float()
        v = value.float()
        # Keys to compress: post-RoPE (default) or pre-RoPE (RoPE re-applied to K_hat).
        k_src = _ROPE_STASH.pop("k").float() if pre_rope else k

        # Frozen bases from the first T0 tokens. SVD on CPU when on MPS (not implemented there).
        svd_dev = "cpu" if device.type == "mps" else device
        mu_k = k_src[:, :, :T0].mean(2, keepdim=True) if center else torch.zeros_like(k_src[:, :, :1])
        mu_v = v[:, :, :T0].mean(2, keepdim=True) if center else torch.zeros_like(v[:, :, :1])
        _, _, Vt_k = torch.linalg.svd((k_src[:, :, :T0] - mu_k).to(svd_dev), full_matrices=False)
        _, _, Vt_v = torch.linalg.svd((v[:, :, :T0] - mu_v).to(svd_dev), full_matrices=False)
        basis_k = Vt_k[..., :r, :].to(device)   # (batch, n_kv, r, d)
        basis_v = Vt_v[..., :r, :].to(device)
        k_hat = mu_k + ((k_src - mu_k) @ basis_k.transpose(-1, -2)) @ basis_k   # (batch, n_kv, T, d)
        if retain_frac > 0:
            if retain_by == "residual":
                keep_score = (k_src - k_hat).norm(dim=-1)            # (batch, n_kv, T), rotation-invariant
            else:
                g = torch.Generator().manual_seed(layer_idx)
                keep_score = torch.rand(k_src.shape[:3], generator=g).to(device)
        if pre_rope:
            rope, cos, sin = _ROPE_STASH.pop("rope"), _ROPE_STASH.pop("cos"), _ROPE_STASH.pop("sin")
            k_hat = rope(k_hat, k_hat, cos.float(), sin.float())[1]
        v_hat = mu_v + ((v - mu_v) @ basis_v.transpose(-1, -2)) @ basis_v
        v_res = v - v_hat

        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        causal = j <= i
        exact = (j < sink_len) | (j > i - window_len) | (i < T0)

        q = query.float().reshape(batch, n_kv_heads, groups, T, d)
        out = torch.empty_like(q)
        # One KV head (and its GQA group of query heads) at a time bounds memory
        # to a few (groups, T, T) score matrices.
        for h in range(n_kv_heads):
            exact_h = exact
            if retain_frac > 0:   # (batch, 1, T, T), broadcast over the GQA group
                kept = torch.stack([_retained_mask(keep_score[b, h], sink_len, window_len,
                                                   retain_frac) for b in range(batch)])[:, None]
                exact_h = exact | kept
                if _DIAG["span"] is not None:
                    a, e = _DIAG["span"]
                    last = T - 1 - window_len                           # last middle token of the final query
                    mid = keep_score[0, h, sink_len:last + 1]
                    pct = (mid[None, :] < keep_score[0, h, a:e, None]).float().mean(1)   # percentile per span token
                    _DIAG["stats"].append((kept[0, 0, T - 1, a:e].float().mean().item(), pct.mean().item()))
            q_h = q[:, h]                                          # (batch, g, T, d)
            s_exact = q_h @ k[:, h, None].transpose(-1, -2)        # (batch, g, T, T)
            s_comp = q_h @ k_hat[:, h, None].transpose(-1, -2)
            scores = torch.where(exact_h, s_exact, s_comp) * scale
            del s_exact, s_comp
            scores = scores.masked_fill(~causal, float("-inf"))
            w = torch.softmax(scores, dim=-1)
            del scores
            # sum_j w_ij * (exact ? v_j : v_hat_j) = w @ v_hat + (w * exact) @ (v - v_hat)
            out[:, h] = w @ v_hat[:, h, None] + w.masked_fill(~exact_h, 0.0) @ v_res[:, h, None]

        out = out.reshape(batch, n_q_heads, T, d).to(dtype)
        return out.transpose(1, 2), None   # (batch, T, n_q_heads, d)

    return _attn


def install_rope_stash(model) -> None:
    """Wraps the modeling module's apply_rotary_pos_emb so each call stashes the
    pre-RoPE keys and RoPE fn/args in _ROPE_STASH (left installed; idempotent)."""
    mod = sys.modules[type(model).__module__]
    rope = getattr(mod.apply_rotary_pos_emb, "_orig", mod.apply_rotary_pos_emb)

    def _stash_rope(q, k, cos, sin, *args, **kwargs):
        _ROPE_STASH.update(k=k, cos=cos, sin=sin,
                           rope=lambda q_, k_, c, s: rope(q_, k_, c, s, *args, **kwargs))
        return rope(q, k, cos, sin, *args, **kwargs)
    _stash_rope._orig = rope
    mod.apply_rotary_pos_emb = _stash_rope


def patch_model_tkv_sink(
    model,
    ranks: Union[int, List[int]],
    warmup_len: int = 512,
    sink_len: int = 64,
    window_len: int = 256,
    pre_rope: bool = False,
    retain_frac: float = 0.0,
    retain_by: str = "residual",
    center: bool = False,
    key: str = "tkv_sink",
) -> str:
    """
    Registers TKV+Sink attention and activates it for model. ranks is an int
    (uniform) or a per-layer list. Returns the original _attn_implementation;
    restore with model_hooks.unpatch_model(model, original).

    pre_rope=True compresses keys before RoPE (RoPE models only, e.g. LLaMA/Mistral):
    wraps the modeling module's apply_rotary_pos_emb (left installed; it only
    stashes its inputs for the hook).

    retain_frac > 0 additionally keeps that fraction of the middle tokens exact,
    chosen by largest key residual (retain_by="residual") or at random ("random").
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if isinstance(ranks, int):
        ranks = [ranks]
    if pre_rope:
        install_rope_stash(model)
    ALL_ATTENTION_FUNCTIONS.register(
        key, _make_tkv_sink_fn(list(ranks), warmup_len, sink_len, window_len, pre_rope,
                               retain_frac, retain_by, center)
    )
    original = model.config._attn_implementation
    model.config._attn_implementation = key
    return original
