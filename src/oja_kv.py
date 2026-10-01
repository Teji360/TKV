"""
OjaKV baseline: online low-rank KV cache compression via Oja's rule.

After every token, the right singular basis V is updated via one step of
the normalized Oja rule (eta_t = 1/((t+1)*||x||^2), gradient ascent on
explained variance) followed by QR re-orthonormalization.

Streaming semantics: coefficients are always expressed in the *current*
basis.  When the basis moves V_old -> V_new, the stored coefficients are
rotated K_proj <- K_proj @ (V_old @ V_new.T) (the Oja analogue of Brand's
U-rotation, O(T R^2)); the new token is then stored as V_new @ x.  So each
reconstructed token is its projection onto the chain of bases seen since it
arrived — realizable online, no access to past raw K/V.  The error vs
BrandOnly comes from the subspace lagging the exact SVD (information
discarded by earlier projections), not from a frame mismatch.

Interface mirrors kv_state.py so benchmark.py can call both identically.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple

from kv_state import tkv_cold_start


class OjaKVState(NamedTuple):
    V_k:     jnp.ndarray  # (R_max, d) orthonormal right basis for keys
    V_v:     jnp.ndarray  # (R_max, d) orthonormal right basis for values
    K_proj:  jnp.ndarray  # (T_max, R_max) compressed keys (V_k @ k at insertion time)
    V_proj:  jnp.ndarray  # (T_max, R_max) compressed values
    seq_len: jnp.ndarray  # scalar int32


def init_oja_kv(T_max: int, R_max: int, d: int, seed: int = 0) -> OjaKVState:
    key = jax.random.PRNGKey(seed)
    # Random orthonormal rows as starting basis (same He-like scale as Phi_init)
    V_raw = jax.random.normal(key, (d, R_max)) * jnp.sqrt(2.0 / d)
    Q, _ = jnp.linalg.qr(V_raw)          # (d, R_max)
    V_init = Q[:, :R_max].T              # (R_max, d)
    return OjaKVState(
        V_k=V_init,
        V_v=V_init,
        K_proj=jnp.zeros((T_max, R_max), dtype=jnp.float32),
        V_proj=jnp.zeros((T_max, R_max), dtype=jnp.float32),
        seq_len=jnp.array(0, dtype=jnp.int32),
    )


def init_oja_kv_from_warmup(
    K_warm: jnp.ndarray, V_warm: jnp.ndarray, R_max: int, T_max: int,
) -> OjaKVState:
    """
    Initialize OjaKV from an SVD of the warmup window — the same basis and
    coefficients TKV's tkv_cold_start produces — so OjaKV and TKV start the
    decode phase from an identical state.  Rank is min(R_max, w, d).
    Subsequent append_token_oja calls continue the step schedule at t = w.
    """
    w = K_warm.shape[0]
    tkv = tkv_cold_start(K_warm, V_warm, R_max, warmup_len=w)
    R = tkv.rank
    return OjaKVState(
        V_k=tkv.basis_k,
        V_v=tkv.basis_v,
        K_proj=jnp.zeros((T_max, R), dtype=jnp.float32).at[:w].set(tkv.proj_k),
        V_proj=jnp.zeros((T_max, R), dtype=jnp.float32).at[:w].set(tkv.proj_v),
        seq_len=jnp.array(w, dtype=jnp.int32),
    )


def _oja_step(
    V: jnp.ndarray,        # (R_max, d) current orthonormal basis
    buf: jnp.ndarray,      # (T_max, R_max) projection buffer
    x: jnp.ndarray,        # (d,) new token vector
    seq_len: jnp.ndarray,  # scalar int32, position to write
    t: int,                # step count (for 1/(t+1) step size)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    One normalized Oja step + QR orthonormalization, rotate the stored
    coefficients into the new basis, then store x's projection onto the new
    basis at position seq_len in buf.

    eta = 1/((t+1)*||x||^2): the 1/(t+1) schedule of Oja (1982) made
    scale-invariant, so large-norm keys (||k|| ~ 20-60) do not snap the
    basis onto the latest token.
    """
    h = V @ x                               # (R_max,)
    eta = 1.0 / ((t + 1) * (x @ x + 1e-9))
    V_new = V + eta * jnp.outer(h, x)      # (R_max, d) gradient ascent step
    # Re-orthonormalize rows via economy QR on the transpose
    Q, R = jnp.linalg.qr(V_new.T)          # Q: (d, R_max), R: (R_max, R_max)
    signs = jnp.sign(jnp.diag(R))          # canonical sign (positive diagonal)
    V_orth = (Q * signs).T                 # (R_max, d)

    buf = buf @ (V @ V_orth.T)             # rotate old coefficients into new basis
    buf = jax.lax.dynamic_update_slice(buf, (V_orth @ x)[None, :], (seq_len, 0))
    return V_orth, buf


def append_token_oja(
    state: OjaKVState,
    k_new: jnp.ndarray,   # (d,)
    v_new: jnp.ndarray,   # (d,)
    R_max: int,
) -> OjaKVState:
    t = int(state.seq_len)
    V_k, K_proj = _oja_step(state.V_k, state.K_proj, k_new, state.seq_len, t)
    V_v, V_proj = _oja_step(state.V_v, state.V_proj, v_new, state.seq_len, t)
    return OjaKVState(
        V_k=V_k, V_v=V_v,
        K_proj=K_proj, V_proj=V_proj,
        seq_len=state.seq_len + 1,
    )


def reconstruct_kv_oja(
    state: OjaKVState, T_max: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Reconstruct K and V using the current basis (coefficients are kept in
    the current basis by _oja_step's rotation, so frames always match).
    """
    mask = (jnp.arange(T_max) < state.seq_len)[:, None]   # (T_max, 1)
    K = state.K_proj @ state.V_k                           # (T_max, d)
    V = state.V_proj @ state.V_v                           # (T_max, d)
    return jnp.where(mask, K, 0.0), jnp.where(mask, V, 0.0)


# --- Unit test ---
if __name__ == "__main__":
    import numpy as np

    T_MAX, R_MAX, D = 256, 16, 64
    key = jax.random.PRNGKey(0)

    A = jax.random.normal(key, (64, D)) * jnp.array([1.0 / (i + 1) for i in range(D)])
    K_true = A[:64]
    V_true = jax.random.normal(jax.random.PRNGKey(1), (64, D))

    state = init_oja_kv(T_MAX, R_MAX, D)
    for i in range(64):
        state = append_token_oja(state, K_true[i], V_true[i], R_MAX)

    K_rec, V_rec = reconstruct_kv_oja(state, T_MAX)
    K_rec = K_rec[:64]

    err_k = float(jnp.linalg.norm(K_true - K_rec) / jnp.linalg.norm(K_true))
    print(f"seq_len={state.seq_len}")
    print(f"K reconstruction rel error: {err_k:.4f}")
    print("oja_kv.py: unit test complete")
