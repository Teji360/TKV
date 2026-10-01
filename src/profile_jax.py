"""
End-to-end decode profiling for the XLA TKV-R cache (tkv_jax.py).

Measures one decode step -- cache update plus attention read -- for a whole
layer (n_kv KV heads x GQA groups), batched, against two baselines at the same
KV memory:

  full    keep every token exactly (no compression)
  evict   sinks + recent window only, window widened to spend the same bytes
  tkv_r   sinks + window + top-M residual tokens exact, everything else rank-R

Reports ms/token, tokens/s and per-head KV bytes, sweeping context length and
batch size. Everything is jitted with static shapes, so the timed region is the
compiled step, not tracing.
"""

import argparse
import time
from functools import partial

import jax
import jax.numpy as jnp

from tkv_jax import add_token, attend, init_state, memory_ratio


def _timeit(fn, iters, warmup=3):
    try:
        for _ in range(warmup):
            jax.block_until_ready(fn())
    except Exception:            # e.g. the dense baseline running out of memory
        return float("nan")
    t0 = time.perf_counter()
    for _ in range(iters):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / iters * 1e3     # ms


@partial(jax.jit, donate_argnums=(0, 1))
def _full_step(K, V, k_new, v_new, q, slot):
    """Baseline: write the new token into a ring-buffer cache and attend over all of it.
    K/V are donated so XLA updates them in place -- no full-cache copy per step."""
    K = jax.lax.dynamic_update_slice(K, k_new[:, :, None], (0, 0, slot, 0))
    V = jax.lax.dynamic_update_slice(V, v_new[:, :, None], (0, 0, slot, 0))
    s = jnp.einsum("bhgd,bhtd->bhgt", q, K) * (q.shape[-1] ** -0.5)
    return K, V, jnp.einsum("bhgt,bhtd->bhgd", jax.nn.softmax(s, -1), V)


def bench_full(B, n_kv, g, T, d, dtype):
    K = jnp.zeros((B, n_kv, T, d), dtype)
    V = jnp.zeros((B, n_kv, T, d), dtype)
    q = jnp.zeros((B, n_kv, g, d), dtype)
    k_new = jnp.zeros((B, n_kv, d), dtype)
    state = [K, V]

    def run():
        state[0], state[1], out = _full_step(state[0], state[1], k_new, k_new, q,
                                             jnp.asarray(0, jnp.int32))
        return out
    return run


def bench_evict(B, n_kv, g, T, d, dtype, keep):
    """Sinks + window only: a dense cache of `keep` tokens, independent of T."""
    return bench_full(B, n_kv, g, keep, d, dtype)


def bench_tkv(B, n_kv, g, T, d, dtype, rank, sink, window, keep):
    """One TKV-R state per (batch, KV head), vmapped; one add_token + one attend."""
    k_warm = jnp.zeros((B, n_kv, max(sink + window, rank), d), dtype)
    v_warm = jnp.zeros_like(k_warm)
    mk = jax.vmap(jax.vmap(lambda k, v: init_state(k, v, rank, T, sink, window, keep)))
    state = mk(k_warm, v_warm)
    k_new = jnp.zeros((B, n_kv, d), dtype)
    q = jnp.zeros((B, n_kv, g, d), dtype)
    scale = d ** -0.5

    @partial(jax.jit, donate_argnums=(0,))
    def step(state, k_new, q):
        state = jax.vmap(jax.vmap(add_token))(state, k_new, k_new, k_new)
        out = jax.vmap(jax.vmap(attend, in_axes=(0, 0, 0, None)),
                       in_axes=(0, 0, 0, None))(state, q, q, scale)
        return state, out

    box = [state]

    def run():
        box[0], out = step(box[0], k_new, q)
        return out
    return run


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", nargs="+", type=int, default=[4096, 32768, 131072])
    p.add_argument("--batch", nargs="+", type=int, default=[1, 8])
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--sink-len", type=int, default=64)
    p.add_argument("--window-len", type=int, default=512)
    p.add_argument("--n-kv", type=int, default=8)
    p.add_argument("--groups", type=int, default=4)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--budget", type=float, default=0.594)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = p.parse_args()
    dt = getattr(jnp, a.dtype)
    d, n_kv, g = a.head_dim, a.n_kv, a.groups

    print(f"device: {jax.devices()[0].device_kind}  dtype={a.dtype}  "
          f"layer = {n_kv} KV heads x {g} groups, d={d}, rank={a.rank}")
    hdr = f"{'ctx':>8} {'batch':>6} {'full ms':>9} {'evict ms':>9} {'tkv_r ms':>9} {'tkv tok/s':>10} {'KV ratio':>9}"
    print(hdr); print("-" * len(hdr))
    for T in a.ctx:
        # M so that TKV-R spends `budget` of the full cache (paper's accounting)
        M = max(0, round((a.budget * T - T * a.rank / d - a.rank - a.sink_len - a.window_len)
                         / (1 - a.rank / d)))
        evict_keep = max(1, round(a.budget * T))
        for B in a.batch:
            t_full = _timeit(bench_full(B, n_kv, g, T, d, dt), a.iters)
            t_ev = _timeit(bench_evict(B, n_kv, g, T, d, dt, evict_keep), a.iters)
            t_tkv = _timeit(bench_tkv(B, n_kv, g, T, d, dt, a.rank, a.sink_len, a.window_len, M),
                            a.iters)
            ratio = memory_ratio(T, d, a.rank, a.sink_len, a.window_len, M)
            fmt = lambda x: "     oom" if x != x else f"{x:>9.3f}"
            print(f"{T:>8} {B:>6} {fmt(t_full)} {fmt(t_ev)} {fmt(t_tkv)} "
                  f"{B / (t_tkv / 1e3):>10.1f} {ratio:>9.3f}", flush=True)
