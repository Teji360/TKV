"""
Query-agnostic context compression with token-selection criteria — teacher-forced.

The context (positions < Tc) is prefilled with exact attention and then
compressed once, before the question (positions >= Tc) arrives. Queries at
i >= Tc see, for each context key j < Tc:
  - EXACT   if j < sink_len, or j >= Tc - window_len, or j is among the M selected tokens;
  - otherwise the rank-R compressed key/value (lowrank=True), or nothing (lowrank=False: evicted).
Question keys (j >= Tc) are always exact. Set _SEL["compress_at"] = Tc before each forward.

Selection criteria (per layer and KV head, over the middle context tokens):
  residual  ||k - B^T B k|| on the (pre-RoPE) key, B = rank-R SVD basis of K[:warmup_len]   (TKV-R)
  residualkv  the Proposition-1 bound itself: (exp(2*alpha*|q|*rho_k) - 1)*|v| + rho_v,
            with |q|, |v| estimated on the warmup window -- ranks tokens by their predicted
            contribution to the attention-output error, not by the key residual alone
  residualx residual with cross-fitting: warmup tokens in each half of the warmup are scored
            against a basis fit on the other half (their own basis underestimates them)
  snapkv    attention mass from the last obs_len context queries, max-pooled (kernel 7)  (SnapKV)
  h2o       total attention mass received from all context queries                       (H2O)
  random    uniform random (control)
  snapkv_q  SnapKV using the question queries (positions >= Tc): query-AWARE reference
Attention-based scores are summed over the GQA group's query heads.
"""

from typing import Tuple

import torch
import torch.nn.functional as F

from tkv_sink_hook import _ROPE_STASH, install_rope_stash

_SEL = {"compress_at": None}


def _select_scores(criterion, p, k_res, Tc, obs_len, layer_idx, h, device):
    """p: (g, T, T) exact causal attention probs of this KV head's group; returns (T,) scores."""
    T = p.shape[-1]
    if criterion.startswith("residual"):
        return k_res
    if criterion == "random":
        g = torch.Generator().manual_seed(1000 * layer_idx + h)
        return torch.rand(T, generator=g).to(device)
    if criterion == "h2o":
        return p[:, :Tc].sum((0, 1))
    rows = slice(Tc - obs_len, Tc) if criterion == "snapkv" else slice(Tc, T)
    s = p[:, rows].sum((0, 1))
    return F.max_pool1d(s[None, None], kernel_size=7, stride=1, padding=3)[0, 0]


def _make_select_fn(criterion: str, lowrank: bool, rank: int, n_select: int, warmup_len: int,
                    sink_len: int, window_len: int, pre_rope: bool, obs_len: int = 32):
    def _attn(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0,
              **kwargs) -> Tuple[torch.Tensor, None]:
        layer_idx = getattr(module, "layer_idx", 0)
        B_, n_kv, T, d = key.shape
        assert B_ == 1, "batch size 1"
        g = query.shape[1] // n_kv
        device, dtype = query.device, query.dtype
        scale = scaling if scaling is not None else d ** -0.5
        Tc = _SEL["compress_at"]
        assert Tc is not None and 0 < Tc <= T

        k, v = key.float()[0], value.float()[0]                      # (n_kv, T, d)
        k_src = _ROPE_STASH.pop("k").float()[0] if pre_rope else k
        need_basis = lowrank or criterion.startswith("residual")
        k_hat, v_hat, k_res = k, v, None
        if need_basis:
            T0 = min(warmup_len, Tc)
            r = min(rank, T0, d)
            svd_dev = "cpu" if device.type == "mps" else device
            Bk = torch.linalg.svd(k_src[:, :T0].to(svd_dev), full_matrices=False)[2][:, :r].to(device)
            Bv = torch.linalg.svd(v[:, :T0].to(svd_dev), full_matrices=False)[2][:, :r].to(device)
            k_hat_src = (k_src @ Bk.mT) @ Bk
            k_res = (k_src - k_hat_src).norm(dim=-1)                 # (n_kv, T)
            if criterion == "residualkv":
                v_res = (v - (v @ Bv.mT) @ Bv).norm(dim=-1)          # (n_kv, T)
                q_typ = query.float()[0, :, :T0].norm(dim=-1).mean()  # typical |q| on the warmup
                v_typ = v[:, :T0].norm(dim=-1).mean(-1, keepdim=True)
                k_res = (torch.expm1(2 * scale * q_typ * k_res) * v_typ + v_res)
            if criterion == "residualx":
                half = T0 // 2
                for a, b, fit in ((0, half, slice(half, T0)), (half, T0, slice(0, half))):
                    Bh = torch.linalg.svd(k_src[:, fit].to(svd_dev), full_matrices=False)[2][:, :r].to(device)
                    x = k_src[:, a:b]
                    k_res[:, a:b] = (x - (x @ Bh.mT) @ Bh).norm(dim=-1)
            if lowrank:
                k_hat = k_hat_src
                if pre_rope:
                    rope, cos, sin = _ROPE_STASH["rope"], _ROPE_STASH["cos"], _ROPE_STASH["sin"]
                    k_hat = rope(k_hat[None], k_hat[None], cos.float(), sin.float())[1][0]
                v_hat = (v @ Bv.mT) @ Bv
        _ROPE_STASH.clear()

        i = torch.arange(T, device=device)[:, None]
        j = torch.arange(T, device=device)[None, :]
        causal = j <= i
        base = (i < Tc) | (j >= Tc) | (j < sink_len) | (j >= Tc - window_len)   # exact regardless of selection
        mid = (torch.arange(T, device=device) >= sink_len) & (torch.arange(T, device=device) < Tc - window_len)

        q = query.float()[0].reshape(n_kv, g, T, d)
        out = torch.empty_like(q)
        for h in range(n_kv):
            s_exact = (q[h] @ k[h].mT) * scale                      # (g, T, T)
            p = torch.softmax(s_exact.masked_fill(~causal, float("-inf")), -1)
            score = _select_scores(criterion, p, None if k_res is None else k_res[h], Tc, obs_len,
                                   layer_idx, h, device)
            score = score.masked_fill(~mid, float("-inf"))
            m = min(n_select, int(mid.sum()))
            sel = torch.zeros(T, dtype=torch.bool, device=device)
            if m > 0:
                sel[score.topk(m).indices] = True
            exact = base | sel[None, :]
            if lowrank:
                s = torch.where(exact, s_exact, (q[h] @ k_hat[h].mT) * scale)
            else:
                s = s_exact.masked_fill(~exact, float("-inf"))
            w = torch.softmax(s.masked_fill(~causal, float("-inf")), -1)
            out[h] = w @ v_hat[h] + w.masked_fill(~exact, 0.0) @ (v[h] - v_hat[h])
        return out.reshape(1, n_kv * g, T, d).to(dtype).transpose(1, 2), None

    return _attn


def select_budget(budget: float, T: int, d: int, rank: int, lowrank: bool, sink_len: int,
                  window_len: int) -> int:
    """
    Number of exact middle tokens M so that per-head KV bytes = budget * full cache.
    A token kept exactly does not also need coefficients, so in lowrank mode only the
    T - M compressed tokens are charged rank-R storage:
        (T - M) R/d + R + S + W + M = budget * T.
    """
    if not lowrank:
        return max(0, round(budget * T - sink_len - window_len))
    m = (budget * T - T * rank / d - rank - sink_len - window_len) / (1 - rank / d)
    return max(0, round(m))


def patch_model_select(model, criterion: str, lowrank: bool, rank: int, n_select: int,
                       warmup_len: int = 512, sink_len: int = 64, window_len: int = 256,
                       pre_rope: bool = True, key: str = "select") -> str:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if pre_rope:
        install_rope_stash(model)
    ALL_ATTENTION_FUNCTIONS.register(key, _make_select_fn(criterion, lowrank, rank, n_select, warmup_len,
                                                          sink_len, window_len, pre_rope))
    original = model.config._attn_implementation
    model.config._attn_implementation = key
    return original
