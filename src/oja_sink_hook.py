"""
Oja+Sink attention hook: a fair, causal OjaKV baseline for teacher-forced perplexity.

Per layer and KV head, for one forward pass over T tokens:
  - Warmup: basis = top-R right singular vectors of K[:T0] (and V[:T0]),
    T0 = min(warmup_len, T) — the same warmup window TKV uses.  Warmup tokens
    are stored as their projections onto this basis.
  - Streaming (t >= T0): tokens are processed in blocks of update_every.
    A block's tokens are projected onto the basis available at the block
    start (built only from earlier tokens), and every query in the block
    attends through that same basis.  After the block, the basis takes a
    normalized Oja step (eta_t = 1/((t+1)*||x_t||^2), summed over the block)
    followed by QR, and all stored coefficients are rotated into the new
    basis (C <- C @ (B_old @ B_new.T)), exactly as in oja_kv._oja_step.
    update_every=1 is fully per-token (token t is projected with the basis
    built from tokens < t); larger values trade basis freshness for speed.
  - Sink tokens (j < sink_len) and a per-query sliding window
    (i - window_len < j <= i) use exact K/V; every other causal key uses
    the compressed K_hat/V_hat as of query i.
  - pre_rope: bases, coefficients and Oja updates use pre-RoPE keys; RoPE is
    re-applied to K_hat (as in tkv_sink_hook). Exact keys stay post-RoPE.

Pure PyTorch, batched over batch and KV heads; GQA handled by grouping the
query heads of each KV head (as in model_hooks' generic hooks).
"""

from typing import Callable, Optional, Tuple

import torch


def _svd_basis(x: torch.Tensor, r: int) -> torch.Tensor:
    """Top-r right singular vectors of x (..., n, d) -> (..., r, d); r <= d even if n < r."""
    _, _, Vh = torch.linalg.svd(x, full_matrices=x.shape[-2] < r)
    return Vh[..., :r, :]


def _oja_block_update(basis: torch.Tensor, x: torch.Tensor, t0: int) -> torch.Tensor:
    """
    Normalized Oja step on tokens x (..., n, d) at positions t0..t0+n-1, then QR.
    basis: (..., r, d) orthonormal rows.  Returns the new orthonormal basis.
    """
    t = torch.arange(t0, t0 + x.shape[-2], device=x.device, dtype=x.dtype)
    eta = 1.0 / ((t + 1) * ((x * x).sum(-1) + 1e-9))        # (..., n)
    h = x @ basis.mT                                        # (..., n, r)
    new = basis + (h * eta[..., None]).mT @ x               # (..., r, d)
    Q, R = torch.linalg.qr(new.mT)                          # (..., d, r), (..., r, r)
    signs = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
    return (Q * signs[..., None, :]).mT


def oja_sink_attention(
    query: torch.Tensor,   # (batch, n_q_heads, T, d)
    key: torch.Tensor,     # (batch, n_kv_heads, T, d)
    value: torch.Tensor,
    rank: int,
    warmup_len: int = 512,
    sink_len: int = 64,
    window_len: int = 256,
    update_every: int = 16,
    scaling: Optional[float] = None,
    attention_mask: Optional[torch.Tensor] = None,
    k_pre: Optional[torch.Tensor] = None,
    rope: Optional[Callable] = None,
) -> torch.Tensor:
    """Returns attention output (batch, n_q_heads, T, d) in float32.
    k_pre/rope (pre-RoPE mode): pre-RoPE keys and rope(k_hat, e) re-applying
    RoPE for positions [0, e)."""
    B, n_kv, T, d = key.shape
    n_q = query.shape[1]
    assert query.shape[2] == T, "teacher-forced full pass only (no KV cache)"
    groups = n_q // n_kv
    scale = scaling if scaling is not None else d ** -0.5
    r = min(rank, d)
    T0 = min(warmup_len, T)

    q = query.float().reshape(B, n_kv, groups, T, d)
    k, v = key.float(), value.float()
    kc = k_pre.float() if k_pre is not None else k   # keys that get compressed

    # --- Masks (T, T): causal, exact (sink + sliding window), compressed ---
    i = torch.arange(T, device=k.device)[:, None]
    j = torch.arange(T, device=k.device)[None, :]
    causal = j <= i
    exact = causal & ((j < sink_len) | (j > i - window_len))
    comp = causal & ~exact

    # --- Warmup: SVD basis on K[:T0], V[:T0] ---
    basis_k, basis_v = _svd_basis(kc[:, :, :T0], r), _svd_basis(v[:, :, :T0], r)
    C_k = kc[:, :, :T0] @ basis_k.mT       # (B, n_kv, T0, r)
    C_v = v[:, :, :T0] @ basis_v.mT

    # --- Blocks of queries, each with its own (basis, coefficients) snapshot ---
    comp_scores = torch.zeros(B, n_kv, groups, T, T, device=k.device)
    blocks = []   # (s, e, V_hat) with V_hat: (B, n_kv, e, d)
    bounds = [(0, T0)] + [(s, min(s + update_every, T)) for s in range(T0, T, update_every)]
    for s, e in bounds:
        if s >= T0:
            C_k = torch.cat([C_k, kc[:, :, s:e] @ basis_k.mT], dim=2)
            C_v = torch.cat([C_v, v[:, :, s:e] @ basis_v.mT], dim=2)
        K_hat = C_k @ basis_k                                      # (B, n_kv, e, d)
        if rope is not None:
            K_hat = rope(K_hat, e)
        comp_scores[..., s:e, :e] = torch.einsum("bhgtd,bhsd->bhgts", q[..., s:e, :], K_hat)
        blocks.append((s, e, C_v @ basis_v))
        if s >= T0 and e < T:
            new_k = _oja_block_update(basis_k, kc[:, :, s:e], s)
            new_v = _oja_block_update(basis_v, v[:, :, s:e], s)
            C_k = C_k @ (basis_k @ new_k.mT)
            C_v = C_v @ (basis_v @ new_v.mT)
            basis_k, basis_v = new_k, new_v

    # --- Single softmax over exact + compressed keys ---
    exact_scores = torch.einsum("bhgtd,bhsd->bhgts", q, k)
    scores = torch.where(exact, exact_scores, comp_scores) * scale
    scores = scores.masked_fill(~causal, float("-inf"))
    if attention_mask is not None:
        m = attention_mask[..., :T, :T].unsqueeze(2)               # (B, 1, 1, T, T)
        scores = scores.masked_fill(~m, float("-inf")) if m.dtype == torch.bool else scores + m.float()
    w = torch.softmax(scores, dim=-1)

    out = torch.einsum("bhgts,bhsd->bhgtd", w * exact, v)
    w_comp = w * comp
    for s, e, V_hat in blocks:
        out[..., s:e, :] += torch.einsum("bhgts,bhsd->bhgtd", w_comp[..., s:e, :e], V_hat)
    return out.reshape(B, n_q, T, d)


def make_oja_sink_attention(
    ranks,
    warmup_len: int = 512,
    sink_len: int = 64,
    window_len: int = 256,
    update_every: int = 16,
    pre_rope: bool = False,
):
    """
    Returns an ALL_ATTENTION_FUNCTIONS-compatible fn. ranks: int (every
    layer) or per-layer list indexed by module.layer_idx.
    """
    def _attn(
        module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask=None,
        scaling=None,
        dropout: float = 0.0,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        if isinstance(ranks, int):
            rank = ranks
        else:
            layer_idx = getattr(module, "layer_idx", 0)
            rank = ranks[min(layer_idx, len(ranks) - 1)]
        k_pre = rope = None
        if pre_rope:
            from tkv_sink_hook import _ROPE_STASH
            k_pre, fn = _ROPE_STASH.pop("k"), _ROPE_STASH.pop("rope")
            cos, sin = _ROPE_STASH.pop("cos").float(), _ROPE_STASH.pop("sin").float()
            rope = lambda x, e: fn(x, x, cos[..., :e, :], sin[..., :e, :])[1]
        out = oja_sink_attention(
            query, key, value, rank,
            warmup_len=warmup_len, sink_len=sink_len, window_len=window_len,
            update_every=update_every, scaling=scaling, attention_mask=attention_mask,
            k_pre=k_pre, rope=rope,
        )
        return out.to(query.dtype).transpose(1, 2), None   # (batch, T, n_q_heads, d)

    return _attn


def patch_model_oja_sink(
    model,
    ranks,
    warmup_len: int = 512,
    sink_len: int = 64,
    window_len: int = 256,
    update_every: int = 16,
    pre_rope: bool = False,
    key: str = "oja_sink",
) -> str:
    """
    Registers the Oja+Sink hook and activates it for model.
    Returns the original _attn_implementation; restore with
    model_hooks.unpatch_model(model, original).
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if pre_rope:
        from tkv_sink_hook import install_rope_stash
        install_rope_stash(model)
    ALL_ATTENTION_FUNCTIONS.register(
        key, make_oja_sink_attention(ranks, warmup_len, sink_len, window_len, update_every, pre_rope)
    )
    original = model.config._attn_implementation
    model.config._attn_implementation = key
    return original
