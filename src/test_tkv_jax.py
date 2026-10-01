"""Check tkv_jax (XLA decode) against a dense reference with identical cache semantics.

Run: cd src && python test_tkv_jax.py
"""
import numpy as np, jax, jax.numpy as jnp
from tkv_jax import init_state, add_token, attend

rng = np.random.default_rng(0)
d, r, S, W, M, T0, N = 16, 4, 2, 6, 5, 24, 40
K = rng.standard_normal((T0 + N, d)).astype(np.float32)
V = rng.standard_normal((T0 + N, d)).astype(np.float32)
q = rng.standard_normal(d).astype(np.float32); scale = d ** -0.5

st = init_state(jnp.array(K[:T0]), jnp.array(V[:T0]), r, T0 + N, S, W, M)
for i in range(T0, T0 + N):
    st = add_token(st, jnp.array(K[i]), jnp.array(V[i]), jnp.array(K[i]))
out = np.array(attend(st, jnp.array(q)[None], jnp.array(q)[None], scale))[0]

# dense reference: same basis, same keep-set, scored directly
B = np.array(st.basis); Bv = np.array(st.basis_v); T = int(st.t)
keep = {int(p) for p in np.array(st.keep_pos) if p >= 0}
exact_idx = list(range(S)) + list(range(max(T - W, 0), T)) + sorted(keep)
comp_idx = [i for i in range(S, max(T - W, 0)) if i not in keep]
Kx = np.concatenate([K[exact_idx], (K[comp_idx] @ B.T) @ B])
Vx = np.concatenate([V[exact_idx], (V[comp_idx] @ Bv.T) @ Bv])
s = Kx @ q * scale
w = np.exp(s - s.max()); w /= w.sum()
ref = w @ Vx
err = np.abs(out - ref).max() / np.abs(ref).max()
print(f"max rel err vs dense reference: {err:.2e}   T={T} kept={len(keep)} compressed={len(comp_idx)}")
assert err < 1e-4, "JAX decode does not match the reference"

# rank = d and everything exact -> must equal full attention
st2 = init_state(jnp.array(K[:T0]), jnp.array(V[:T0]), d, T0 + N, S, W, M)
for i in range(T0, T0 + N):
    st2 = add_token(st2, jnp.array(K[i]), jnp.array(V[i]), jnp.array(K[i]))
out2 = np.array(attend(st2, jnp.array(q)[None], jnp.array(q)[None], scale))[0]
s = K[:T] @ q * scale; w = np.exp(s - s.max()); w /= w.sum(); full = w @ V[:T]
err2 = np.abs(out2 - full).max() / np.abs(full).max()
print(f"rank=d equals full attention: {err2:.2e}")
assert err2 < 1e-4
print("PASS")
