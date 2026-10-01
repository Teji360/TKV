"""
TKV-R decode step in JAX, XLA-compilable with static shapes.

State per (layer, KV head), all shapes fixed at compile time:
  basis/basis_v (r, d)   frozen bases for keys (pre-RoPE) and values
  coeff/coeff_v (T_max, r)  rank-r coefficients of every compressed token
  sink_k/v (S, d)        exact attention sinks
  win_k/v  (W, d)        exact recent window, a ring buffer
  keep_k/v (M, d)        exact copies of the M largest-residual evicted tokens
  keep_rho (M,)          their residuals: a fixed-capacity min-heap kept as a flat
                         array, since argmin over M is one reduction on an accelerator
  keep_pos (M,)          their positions (for the causal mask / RoPE)
  comp_live (T_max,)     which coefficient rows are still the *only* copy of a token
                         (a token promoted into keep_k is scored exactly, not twice)

One decode token costs O(r d) to project, O(M) for the heap compare, and
O(r(T + d) + (S + W + M) d) to read. Nothing depends on T beyond the coefficient
matmul, and no attention probabilities are ever materialized.

add_token/attend are deliberately left un-jitted: they are meant to be called
inside a caller's jit (see profile_jax.step), where donating the state lets XLA
update the coefficient buffers in place. A nested jit would block that donation.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class TKVState(NamedTuple):
    basis: jnp.ndarray      # (r, d) keys, pre-RoPE
    basis_v: jnp.ndarray    # (r, d) values
    coeff: jnp.ndarray      # (T_max, r) key coefficients
    coeff_v: jnp.ndarray    # (T_max, r) value coefficients
    sink_k: jnp.ndarray     # (S, d)
    sink_v: jnp.ndarray
    win_k: jnp.ndarray      # (W, d) ring buffer
    win_v: jnp.ndarray
    keep_k: jnp.ndarray     # (M, d)
    keep_v: jnp.ndarray
    keep_rho: jnp.ndarray   # (M,)
    keep_pos: jnp.ndarray   # (M,) int32
    comp_live: jnp.ndarray  # (T_max,) bool
    t: jnp.ndarray          # scalar int32, tokens seen


def init_state(k_warm, v_warm, rank, t_max, sink_len, window_len, keep_len,
               k_warm_post=None):
    """
    Prefill: fit the frozen basis on the warmup keys (pre-RoPE), then lay out the
    cache exactly as streaming would leave it -- sinks and the last window_len
    tokens exact, everything between them compressed. k_warm_post is the
    post-RoPE warmup keys for the window (defaults to k_warm, i.e. no RoPE).
    """
    T0, d = k_warm.shape
    k_post = k_warm if k_warm_post is None else k_warm_post
    # the prefill SVD runs in float32 (cuSolver has no bf16 path) and the basis is
    # cast back to the cache dtype
    _, _, vh = jnp.linalg.svd(k_warm.astype(jnp.float32), full_matrices=False)
    basis = vh[:rank].astype(k_warm.dtype)                   # (r, d)
    _, _, vhv = jnp.linalg.svd(v_warm.astype(jnp.float32), full_matrices=False)
    basis_v = vhv[:rank].astype(k_warm.dtype)

    coeff = jnp.zeros((t_max, rank), k_warm.dtype)
    coeff_v = jnp.zeros((t_max, rank), k_warm.dtype)
    mid = slice(sink_len, max(T0 - window_len, sink_len))
    coeff = coeff.at[mid].set(k_warm[mid] @ basis.T)
    coeff_v = coeff_v.at[mid].set(v_warm[mid] @ basis_v.T)
    live = jnp.zeros((t_max,), bool).at[mid].set(True)

    return TKVState(
        basis=basis, basis_v=basis_v,
        coeff=coeff, coeff_v=coeff_v,
        sink_k=k_post[:sink_len], sink_v=v_warm[:sink_len],
        win_k=k_post[T0 - window_len:T0], win_v=v_warm[T0 - window_len:T0],
        keep_k=jnp.zeros((keep_len, d), k_warm.dtype),
        keep_v=jnp.zeros((keep_len, d), k_warm.dtype),
        keep_rho=jnp.full((keep_len,), -jnp.inf, k_warm.dtype),
        keep_pos=jnp.full((keep_len,), -1, jnp.int32),
        comp_live=live,
        t=jnp.asarray(T0, jnp.int32),
    )


def add_token(state: TKVState, k_pre, v, k_post):
    """
    Append one token. k_pre/k_post are the key before/after RoPE; v the value.
    The token entering the cache evicts the oldest window slot, which is then
    either promoted into the exact keep-set (if its residual beats the heap
    minimum) or left as coefficients only.
    """
    W = state.win_k.shape[0]
    slot = state.t % W

    # evicted = the window slot we are about to overwrite
    ev_k, ev_v = state.win_k[slot], state.win_v[slot]   # note: post-RoPE key (see add_token docstring)
    ev_pos = state.t - W
    ev_c = state.basis @ ev_k
    ev_cv = state.basis_v @ ev_v
    ev_rho = jnp.sqrt(jnp.maximum(ev_k @ ev_k - ev_c @ ev_c, 0.0))

    # fixed-capacity heap: replace the current minimum if this residual is larger
    lo = jnp.argmin(state.keep_rho)
    take = (ev_rho > state.keep_rho[lo]) & (ev_pos >= 0)
    keep_rho = state.keep_rho.at[lo].set(jnp.where(take, ev_rho, state.keep_rho[lo]))
    keep_pos = state.keep_pos.at[lo].set(jnp.where(take, ev_pos, state.keep_pos[lo]))
    keep_k = state.keep_k.at[lo].set(jnp.where(take, ev_k, state.keep_k[lo]))
    keep_v = state.keep_v.at[lo].set(jnp.where(take, ev_v, state.keep_v[lo]))

    # liveness: the evicted token is compressed unless promoted; a token displaced
    # from the keep-set falls back to the coefficients it already has.
    displaced = state.keep_pos[lo]
    comp_live = state.comp_live
    comp_live = comp_live.at[jnp.maximum(displaced, 0)].set(
        jnp.where(take & (displaced >= 0), True, comp_live[jnp.maximum(displaced, 0)]))
    comp_live = comp_live.at[jnp.maximum(ev_pos, 0)].set(jnp.where(ev_pos >= 0, ~take, False))

    # every evicted token keeps its coefficients (this is what makes the bound apply)
    at = (jnp.maximum(ev_pos, 0), 0)
    coeff = jax.lax.dynamic_update_slice(state.coeff, ev_c[None, :], at)
    coeff_v = jax.lax.dynamic_update_slice(state.coeff_v, ev_cv[None, :], at)
    return state._replace(
        coeff=coeff, coeff_v=coeff_v,
        win_k=state.win_k.at[slot].set(k_post), win_v=state.win_v.at[slot].set(v),
        keep_k=keep_k, keep_v=keep_v, keep_rho=keep_rho, keep_pos=keep_pos,
        comp_live=comp_live, t=state.t + 1,
    )


def attend(state: TKVState, q, q_pre, scale):
    """
    One decode step for a whole GQA group: q, q_pre are (g, d) (post-/pre-RoPE),
    the output is (g, d). The group shares a single read of the coefficient array --
    scoring all g queries in one matmul -- which is what makes the rank-r cache
    actually cheaper than the dense one at long context.

    Exact tokens (sinks, window, keep-set) are scored directly; the rest through the
    frozen basis at O(r(T + d)). The softmax is fused across the four segments, so
    nothing of length T is concatenated or re-scanned, and no attention
    probabilities are materialized.
    """
    s_sink = (state.sink_k @ q.T) * scale                     # (S, g)
    s_win = (state.win_k @ q.T) * scale                       # (W, g)
    s_keep = jnp.where((state.keep_pos >= 0)[:, None], (state.keep_k @ q.T) * scale, -jnp.inf)
    s_comp = (state.coeff @ (state.basis @ q_pre.T)) * scale  # (T_max, g), one read of coeff

    live = state.comp_live[:, None]
    m = jnp.maximum(jnp.maximum(s_sink.max(0), s_win.max(0)),
                    jnp.maximum(s_keep.max(0), jnp.where(live, s_comp, -jnp.inf).max(0)))
    e_sink, e_win = jnp.exp(s_sink - m), jnp.exp(s_win - m)
    e_keep = jnp.where((state.keep_pos >= 0)[:, None], jnp.exp(s_keep - m), 0.0)
    e_comp = jnp.where(live, jnp.exp(s_comp - m), 0.0)

    denom = e_sink.sum(0) + e_win.sum(0) + e_keep.sum(0) + e_comp.sum(0)   # (g,)
    out = (e_sink.T @ state.sink_v + e_win.T @ state.win_v + e_keep.T @ state.keep_v
           + (e_comp.T @ state.coeff_v) @ state.basis_v)                   # (g, d)
    return out / denom[:, None]


def memory_ratio(t, d, rank, sink_len, window_len, keep_len):
    """Per-head KV entries relative to the full cache, matching the paper's accounting."""
    compressed = (t - keep_len) * rank + rank * d
    exact = (sink_len + window_len + keep_len) * d
    return (compressed + exact) / (t * d)
