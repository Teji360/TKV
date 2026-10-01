"""
Smoke tests for Chris's contributions:
  1. loop.py bug fixes  (inner_loop_update + query)
  2. sinked_add_token   (kv_state.py)
  3. sketch_cold_start  (loop.py)
  4. OjaKV fixes        (oja_kv.py) and Oja+Sink attention hook (oja_sink_hook.py)
"""

import jax
import jax.numpy as jnp
import numpy as np

from kv_state import (
    sinked_cold_start,
    sinked_add_token,
    tkv_reconstruct,
)
from loop import JaxTensorSketchStore, sketch_cold_start
from attention import sinked_attention


# ---------------------------------------------------------------------------
# 1. loop.py bug fixes — inner_loop_update and query
# ---------------------------------------------------------------------------

def test_inner_loop_update():
    store = JaxTensorSketchStore(depth=4, width=8, embedding_dim=16)
    state = store.init_state()

    token = jax.random.normal(jax.random.PRNGKey(0), (16,))
    state2 = JaxTensorSketchStore.inner_loop_update(state, token, store.d, store.w)

    assert state2.S.shape == (4, 8), f"bad shape: {state2.S.shape}"
    assert not jnp.allclose(state2.S, state.S), "S should change after update"
    print("PASS  inner_loop_update — shape correct, state mutated")


def test_query():
    store = JaxTensorSketchStore(depth=4, width=8, embedding_dim=16)
    state = store.init_state()

    # Insert two tokens then query one of them back
    t1 = jax.random.normal(jax.random.PRNGKey(1), (16,))
    t2 = jax.random.normal(jax.random.PRNGKey(2), (16,))
    state = JaxTensorSketchStore.inner_loop_update(state, t1, store.d, store.w)
    state = JaxTensorSketchStore.inner_loop_update(state, t2, store.d, store.w)

    score_t1 = JaxTensorSketchStore.query(state, t1, store.d, store.w)
    score_zero = JaxTensorSketchStore.query(state, jnp.zeros(16), store.d, store.w)

    assert score_t1.shape == (), f"query should return scalar, got {score_t1.shape}"
    assert float(score_zero) == 0.0, "zero vector should score 0"
    print(f"PASS  query — scalar output, t1 score={float(score_t1):.4f}, zero score={float(score_zero):.4f}")


# ---------------------------------------------------------------------------
# 2. sinked_add_token
# ---------------------------------------------------------------------------

def test_sinked_add_token_streaming():
    """After cold_start with a full sequence, stream 10 tokens and verify
    that seq_len grows, window stays fixed, and middle grows by 10."""
    D, RANK, SINK, WINDOW = 32, 4, 8, 8
    T = 50
    key = jax.random.PRNGKey(0)
    K = jax.random.normal(key, (T, D))
    V = jax.random.normal(jax.random.PRNGKey(1), (T, D))

    state = sinked_cold_start(K, V, RANK, sink_len=SINK, window_len=WINDOW, warmup_len=16)
    assert state.middle is not None
    mid_start = state.middle.seq_len

    for i in range(10):
        k_new = jax.random.normal(jax.random.PRNGKey(100 + i), (D,))
        v_new = jax.random.normal(jax.random.PRNGKey(200 + i), (D,))
        state = sinked_add_token(state, k_new, v_new, RANK, window_capacity=WINDOW)

    assert state.seq_len == T + 10,         f"seq_len={state.seq_len}, expected {T+10}"
    assert state.sink_len == SINK,           f"sink changed: {state.sink_len}"
    assert state.window_len == WINDOW,       f"window changed: {state.window_len}"
    assert len(state.recent_k) == WINDOW,    f"recent_k len={len(state.recent_k)}"
    assert state.middle.seq_len == mid_start + 10, \
        f"middle grew by {state.middle.seq_len - mid_start}, expected 10"
    print(f"PASS  sinked_add_token streaming — seq_len={state.seq_len}, middle={state.middle.seq_len}")


def test_sinked_add_token_first_eviction():
    """Start with a short sequence (no middle), fill the window, trigger
    the first eviction, and confirm a middle is created with seq_len=1."""
    D, RANK, SINK, WINDOW = 32, 4, 8, 8
    T = 12  # T < SINK + WINDOW so middle starts as None
    key = jax.random.PRNGKey(10)
    K = jax.random.normal(key, (T, D))
    V = jax.random.normal(jax.random.PRNGKey(11), (T, D))

    state = sinked_cold_start(K, V, RANK, sink_len=SINK, window_len=WINDOW)
    assert state.middle is None, "middle should be None for short sequence"
    initial_window = state.window_len

    # Add enough tokens to fill the window and cause one eviction
    n_to_fill = WINDOW - initial_window + 1
    for i in range(n_to_fill):
        k_new = jax.random.normal(jax.random.PRNGKey(300 + i), (D,))
        v_new = jax.random.normal(jax.random.PRNGKey(400 + i), (D,))
        state = sinked_add_token(state, k_new, v_new, RANK, window_capacity=WINDOW)

    assert state.window_len == WINDOW,   f"window={state.window_len}"
    assert state.middle is not None,     "middle should exist after first eviction"
    assert state.middle.seq_len == 1,    f"middle.seq_len={state.middle.seq_len}, expected 1"
    print(f"PASS  sinked_add_token first eviction — middle created, seq_len={state.middle.seq_len}")


def test_sinked_add_token_attention():
    """Verify sinked_attention runs without error on a streamed state."""
    D, RANK, SINK, WINDOW = 32, 4, 8, 8
    T = 50
    key = jax.random.PRNGKey(20)
    K = jax.random.normal(key, (T, D))
    V = jax.random.normal(jax.random.PRNGKey(21), (T, D))

    state = sinked_cold_start(K, V, RANK, sink_len=SINK, window_len=WINDOW, warmup_len=16)
    for i in range(5):
        k_new = jax.random.normal(jax.random.PRNGKey(500 + i), (D,))
        v_new = jax.random.normal(jax.random.PRNGKey(600 + i), (D,))
        state = sinked_add_token(state, k_new, v_new, RANK, window_capacity=WINDOW)

    Q = jax.random.normal(jax.random.PRNGKey(99), (4, D)) * 0.1
    out = sinked_attention(Q, state)
    assert out.shape == (4, D), f"bad output shape: {out.shape}"
    assert not jnp.any(jnp.isnan(out)), "NaN in attention output"
    print(f"PASS  sinked_add_token attention — output shape {out.shape}, no NaNs")


# ---------------------------------------------------------------------------
# 3. sketch_cold_start
# ---------------------------------------------------------------------------

def test_sketch_cold_start_shape():
    """sketch_cold_start should return a CompressedKVState with the right shapes."""
    from kv_state import CompressedKVState, reconstruct_kv

    D, R_MAX, T_MAX = 32, 4, 64
    T = 20
    DEPTH, WIDTH = 2, R_MAX  # depth*width >= R_MAX

    key = jax.random.PRNGKey(0)
    K = jax.random.normal(key, (T, D))
    V = jax.random.normal(jax.random.PRNGKey(1), (T, D))

    store = JaxTensorSketchStore(depth=DEPTH, width=WIDTH, embedding_dim=D)
    sketch_state = store.init_state()

    # Feed K rows into the sketch so it has seen the same tokens
    for i in range(T):
        sketch_state = JaxTensorSketchStore.inner_loop_update(
            sketch_state, K[i], store.d, store.w
        )

    state = sketch_cold_start(K, V, store, sketch_state, R_MAX, T_MAX)
    assert isinstance(state, CompressedKVState), "wrong return type"
    assert state.U_k.shape == (T_MAX, R_MAX), f"U_k shape {state.U_k.shape}"
    assert state.V_k.shape == (R_MAX, D),     f"V_k shape {state.V_k.shape}"
    assert int(state.seq_len) == T,            f"seq_len={state.seq_len}"
    print(f"PASS  sketch_cold_start shapes — U_k={state.U_k.shape}, rank={int(state.active_rank)}")


def test_sketch_cold_start_reconstruction():
    """Reconstruction error should be reasonable (< 0.8) for low-rank data."""
    from kv_state import reconstruct_kv

    D, R_MAX, T_MAX = 32, 8, 64
    T = 30
    DEPTH, WIDTH = 2, R_MAX

    # Low-rank K so compression should work well
    key = jax.random.PRNGKey(5)
    base = jax.random.normal(key, (4, D))
    K = jax.random.normal(jax.random.PRNGKey(6), (T, 4)) @ base
    V = jax.random.normal(jax.random.PRNGKey(7), (T, D))

    store = JaxTensorSketchStore(depth=DEPTH, width=WIDTH, embedding_dim=D)
    sketch_state = store.init_state()
    for i in range(T):
        sketch_state = JaxTensorSketchStore.inner_loop_update(
            sketch_state, K[i], store.d, store.w
        )

    state = sketch_cold_start(K, V, store, sketch_state, R_MAX, T_MAX)
    K_rec, _ = reconstruct_kv(state, T_MAX)
    err = float(jnp.linalg.norm(K - K_rec[:T]) / (jnp.linalg.norm(K) + 1e-9))
    assert err < 0.8, f"reconstruction error too large: {err:.4f}"
    print(f"PASS  sketch_cold_start reconstruction — rel error={err:.4f}")


def test_sketch_cold_start_importance_effect():
    """With full-rank data (lossy compression), sketch_cold_start should
    produce valid output and not diverge vs plain cold_start on the same phi."""
    from kv_state import cold_start, reconstruct_kv

    D, R_MAX, T_MAX = 32, 4, 64  # R_MAX < D so compression is genuinely lossy
    T = 30
    DEPTH, WIDTH = 2, R_MAX

    key = jax.random.PRNGKey(9)
    K = jax.random.normal(key, (T, D))  # full-rank: compression must approximate
    V = jax.random.normal(jax.random.PRNGKey(11), (T, D))

    store = JaxTensorSketchStore(depth=DEPTH, width=WIDTH, embedding_dim=D, seed=12)
    sketch_state = store.init_state()
    for i in range(T):
        sketch_state = JaxTensorSketchStore.inner_loop_update(
            sketch_state, K[i], store.d, store.w
        )

    state_plain  = cold_start(K, V, store.Phi_init, R_MAX, T_MAX, D)
    state_sketch = sketch_cold_start(K, V, store, sketch_state, R_MAX, T_MAX)

    K_plain,  _ = reconstruct_kv(state_plain,  T_MAX)
    K_sketch, _ = reconstruct_kv(state_sketch, T_MAX)
    err_plain  = float(jnp.linalg.norm(K - K_plain[:T])  / (jnp.linalg.norm(K) + 1e-9))
    err_sketch = float(jnp.linalg.norm(K - K_sketch[:T]) / (jnp.linalg.norm(K) + 1e-9))

    # Both are lossy — sketch-weighted should be in the same ballpark as plain.
    assert err_sketch < err_plain + 0.15, \
        f"sketch err={err_sketch:.4f} is more than 0.15 above plain err={err_plain:.4f}"
    print(f"PASS  sketch importance effect — plain={err_plain:.4f}, sketch={err_sketch:.4f}")


# ---------------------------------------------------------------------------
# 4. OjaKV (normalized step, rotated coefficients, warmup init) + Oja+Sink hook
# ---------------------------------------------------------------------------

def _oja_stream_err(K, V, R, warm=0):
    from oja_kv import init_oja_kv, init_oja_kv_from_warmup, append_token_oja, reconstruct_kv_oja
    T, D = K.shape
    if warm:
        state = init_oja_kv_from_warmup(K[:warm], V[:warm], R, T)
    else:
        state = init_oja_kv(T, R, D)
    for i in range(warm, T):
        state = append_token_oja(state, K[i], V[i], R)
    K_rec, _ = reconstruct_kv_oja(state, T)
    return float(jnp.linalg.norm(K - K_rec) / jnp.linalg.norm(K))


def _decaying_kv(T=128, D=32, seed=0):
    # Large-norm keys with a mean offset (like real keys, ||k|| ~ 40) and a decaying spectrum
    decay = jnp.array([1.0 / (i + 1) ** 0.5 for i in range(D)])
    K = 30.0 * jax.random.normal(jax.random.PRNGKey(seed), (T, D)) * decay + 20.0 * decay
    V = jax.random.normal(jax.random.PRNGKey(seed + 1), (T, D))
    return K, V


def test_oja_error_at_most_one():
    K, V = _decaying_kv()
    errs = [_oja_stream_err(K, V, R) for R in (2, 4, 8, 16)]
    assert all(e <= 1.0 for e in errs), f"Oja rel error > 1: {errs}"
    print(f"PASS  OjaKV rel error <= 1 (random init, streaming) — {[round(e, 4) for e in errs]}")


def test_oja_error_decreases_with_rank():
    K, V = _decaying_kv()
    for warm in (0, 32):
        errs = [_oja_stream_err(K, V, R, warm) for R in (2, 4, 8, 16, 32)]
        assert all(a > b for a, b in zip(errs, errs[1:])), f"warm={warm}: not decreasing {errs}"
    print(f"PASS  OjaKV error decreases with rank — warm=32: {[round(e, 4) for e in errs]}")


def test_oja_exact_when_rank_covers():
    K, V = _decaying_kv(T=48)
    err_warm = _oja_stream_err(K[:24], V[:24], R=24, warm=24)  # R >= T, all tokens in the SVD warmup
    err_full = _oja_stream_err(K, V, R=32)                     # R = d, pure streaming from random init
    err_mix = _oja_stream_err(K, V, R=32, warm=32)             # R = d, warmup then streaming
    assert max(err_warm, err_full, err_mix) < 1e-4, f"{err_warm}, {err_full}, {err_mix}"
    print(f"PASS  OjaKV exact at full rank — R>=T warm={err_warm:.1e}, R=d stream={err_full:.1e}, R=d warm+stream={err_mix:.1e}")


def _exact_and_hooked_logits(model, ids, **hook_kwargs):
    import torch
    from model_hooks import unpatch_model
    from oja_sink_hook import patch_model_oja_sink
    with torch.no_grad():
        ref = model(ids, use_cache=False).logits
        orig = patch_model_oja_sink(model, **hook_kwargs)
        out = model(ids, use_cache=False).logits
        unpatch_model(model, orig)
    return ref, out


def _tiny_gqa_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=100, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    cfg._attn_implementation = "sdpa"
    return LlamaForCausalLM(cfg).eval(), torch.randint(0, 100, (2, 96))


def test_oja_sink_hook_full_rank_exact():
    """rank = head_dim: compressed K/V are exact, so logits must match exact attention."""
    from model_hooks import load_gpt2
    model, tok = load_gpt2()
    ids = tok("The quick brown fox jumps over the lazy dog. " * 20, return_tensors="pt")["input_ids"][:, :160]
    ref, out = _exact_and_hooked_logits(model, ids, ranks=64, warmup_len=32, sink_len=4,
                                        window_len=16, update_every=8)
    diff = float((out - ref).abs().max() / ref.abs().max())
    assert diff < 1e-4, f"GPT-2 rank=d rel logit diff {diff:.2e}"

    llama, ids = _tiny_gqa_llama()
    ref, out = _exact_and_hooked_logits(llama, ids, ranks=[16, 16], warmup_len=16, sink_len=2,
                                        window_len=8, update_every=1)
    diff_l = float((out - ref).abs().max() / ref.abs().max())
    assert diff_l < 1e-4, f"GQA Llama rank=d rel logit diff {diff_l:.2e}"
    print(f"PASS  Oja+Sink hook rank=d exact — GPT-2 {diff:.1e}, GQA Llama {diff_l:.1e}")


def test_oja_sink_hook_sink_window_exact():
    """sink_len + window_len >= T: every key is exact regardless of rank."""
    llama, ids = _tiny_gqa_llama()
    ref, out = _exact_and_hooked_logits(llama, ids, ranks=2, warmup_len=16, sink_len=48,
                                        window_len=48, update_every=4)
    diff = float((out - ref).abs().max() / ref.abs().max())
    assert diff < 1e-4, f"S+W>=T rel logit diff {diff:.2e}"
    print(f"PASS  Oja+Sink hook S+W>=T exact — rel logit diff {diff:.1e}")


def test_sink_hooks_pre_rope_full_rank_exact():
    """pre_rope=True with rank = head_dim: both hooks reproduce exact logits on a RoPE model."""
    import torch
    from model_hooks import unpatch_model
    from oja_sink_hook import patch_model_oja_sink
    from tkv_sink_hook import patch_model_tkv_sink
    llama, ids = _tiny_gqa_llama()
    with torch.no_grad():
        ref = llama(ids, use_cache=False).logits
        for patch in (patch_model_oja_sink, patch_model_tkv_sink):
            orig = patch(llama, [16, 16], warmup_len=16, sink_len=2, window_len=8, pre_rope=True)
            out = llama(ids, use_cache=False).logits
            unpatch_model(llama, orig)
            diff = float((out - ref).abs().max() / ref.abs().max())
            assert diff < 1e-4, f"{patch.__name__} pre_rope rank=d rel logit diff {diff:.2e}"
    print("PASS  Oja+Sink / TKV+Sink pre_rope rank=d exact")


def test_streaming_llm_hook_exact_limit():
    """StreamingLLM with a window covering the whole sequence is exact attention."""
    import torch
    from model_hooks import unpatch_model
    from streaming_llm_hook import patch_model_streaming_llm
    llama, ids = _tiny_gqa_llama()
    with torch.no_grad():
        ref = llama(ids, use_cache=False).logits
        orig = patch_model_streaming_llm(llama, 16, sink_len=2, window_len=96, ctx_len=96)
        out = llama(ids, use_cache=False).logits
        unpatch_model(llama, orig)
    diff = float((out - ref).abs().max() / ref.abs().max())
    assert diff < 1e-4, f"StreamingLLM full-window rel logit diff {diff:.2e}"
    print(f"PASS  StreamingLLM hook full window exact — rel logit diff {diff:.1e}")


def test_residual_retention():
    """Retention mask picks the top-residual middle tokens per query; retain_frac=1
    keeps every token exact, so the TKV hook reproduces exact logits."""
    import torch
    from model_hooks import unpatch_model
    from tkv_sink_hook import _retained_mask, patch_model_tkv_sink
    score = torch.tensor([9., 9., 1., 5., 3., 8., 2., 7., 0., 0.])
    m = _retained_mask(score, sink_len=2, window_len=2, frac=0.5)
    # query 9 sees middle [2..7] (6 tokens) -> keeps the top 3 residuals: 5 (8.), 7 (7.), 3 (5.)
    assert m[9].nonzero().flatten().tolist() == [3, 5, 7], m[9]
    # query 5 sees middle [2..3] -> keeps 1 token: 3 (5.)
    assert m[5].nonzero().flatten().tolist() == [3], m[5]
    llama, ids = _tiny_gqa_llama()
    with torch.no_grad():
        ref = llama(ids, use_cache=False).logits
        orig = patch_model_tkv_sink(llama, 2, warmup_len=16, sink_len=2, window_len=8,
                                    pre_rope=True, retain_frac=1.0)
        out = llama(ids, use_cache=False).logits
        unpatch_model(llama, orig)
    diff = float((out - ref).abs().max() / ref.abs().max())
    assert diff < 1e-4, f"retain_frac=1 rel logit diff {diff:.2e}"
    print(f"PASS  residual retention — top-k mask correct, retain_frac=1 exact ({diff:.1e})")


def test_select_hook_exact_limits():
    """Query-agnostic select hook: a window covering the context, or rank = d with every
    token compressed, reproduces exact logits; evicting everything but sinks changes them."""
    import torch
    from model_hooks import unpatch_model
    from select_hook import _SEL, patch_model_select
    llama, ids = _tiny_gqa_llama()
    ids = ids[:1]
    _SEL["compress_at"] = 80
    try:
        with torch.no_grad():
            ref = llama(ids, use_cache=False).logits
            diffs = []
            for crit, lowrank, rank, M, window in [("snapkv", False, 0, 0, 96), ("residual", True, 16, 0, 0),
                                                   ("h2o", False, 0, 0, 0)]:
                orig = patch_model_select(llama, crit, lowrank, rank, M, warmup_len=16, sink_len=2,
                                          window_len=window, pre_rope=True)
                out = llama(ids, use_cache=False).logits
                unpatch_model(llama, orig)
                diffs.append(float((out - ref).abs().max() / ref.abs().max()))
    finally:
        _SEL["compress_at"] = None
    assert diffs[0] < 1e-4 and diffs[1] < 1e-4, diffs
    assert diffs[2] > 1e-3, diffs
    print(f"PASS  select hook exact limits — window {diffs[0]:.1e}, rank=d {diffs[1]:.1e}; evicted {diffs[2]:.1e}")


def test_oja_sink_hook_causal():
    """Changing tokens at positions >= 70 must not change outputs at positions < 70."""
    import torch
    from oja_sink_hook import oja_sink_attention
    torch.manual_seed(0)
    q = torch.randn(1, 4, 120, 16)
    k = torch.randn(1, 2, 120, 16) * 5
    v = torch.randn(1, 2, 120, 16)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 70:] += 3.0
    v2[:, :, 70:] += 3.0
    kw = dict(rank=4, warmup_len=32, sink_len=4, window_len=8, update_every=1)
    o1 = oja_sink_attention(q, k, v, **kw)
    o2 = oja_sink_attention(q, k2, v2, **kw)
    assert torch.equal(o1[:, :, :70], o2[:, :, :70]), "future tokens leaked into past outputs"
    assert not torch.allclose(o1[:, :, 70:], o2[:, :, 70:])
    print("PASS  Oja+Sink hook causal — outputs before a change are bit-identical")


# ---------------------------------------------------------------------------
# 5. TKV+Sink batched hook (tkv_sink_hook.py)
# ---------------------------------------------------------------------------

def test_tkv_sink_hook_matches_streaming():
    """Per-query outputs of the batched teacher-forced hook match streaming:
    exact attention during warmup, then sinked_add_token + frozen-basis middle."""
    import torch
    from kv_state import SinkedKVState, TKVState, tkv_cold_start
    from tkv_sink_hook import _make_tkv_sink_fn

    D, RANK, SINK, WINDOW, T0, T = 16, 4, 4, 6, 16, 40
    rng = np.random.default_rng(0)
    K = rng.standard_normal((T, D)).astype(np.float32)
    V = rng.standard_normal((T, D)).astype(np.float32)
    Q = rng.standard_normal((T, D)).astype(np.float32)
    scale = 0.3

    class _M:
        layer_idx = 0
    t = lambda x: torch.tensor(x)[None, None]
    out, _ = _make_tkv_sink_fn([RANK], T0, SINK, WINDOW)(_M(), t(Q), t(K), t(V), scaling=scale)
    out = out[0, :, 0].numpy()                                   # (T, D)

    ref = np.zeros((T, D), np.float32)
    for i in range(T0):                                          # warmup: exact causal
        s = (Q[i] @ K[:i + 1].T) * scale
        w = np.exp(s - s.max()); w /= w.sum()
        ref[i] = w @ V[:i + 1]

    # cache holding tokens [0, T0): sink | frozen-basis middle | recent window
    tkv = tkv_cold_start(jnp.array(K[:T0]), jnp.array(V[:T0]), RANK, warmup_len=T0)
    mid = slice(SINK, T0 - WINDOW)
    middle = TKVState(basis_k=tkv.basis_k, basis_v=tkv.basis_v,
                      proj_k=np.array(tkv.proj_k[mid]), proj_v=np.array(tkv.proj_v[mid]),
                      seq_len=T0 - WINDOW - SINK, rank=tkv.rank)
    state = SinkedKVState(sink_k=K[:SINK], sink_v=V[:SINK], middle=middle,
                          recent_k=K[T0 - WINDOW:T0], recent_v=V[T0 - WINDOW:T0],
                          sink_len=SINK, window_len=WINDOW, seq_len=T0, d=D)
    for i in range(T0, T):
        state = sinked_add_token(state, jnp.array(K[i]), jnp.array(V[i]), RANK, window_capacity=WINDOW)
        ref[i] = np.array(sinked_attention(jnp.array(Q[i][None]), state, scale=scale))[0]

    err = np.abs(out - ref).max()
    assert err < 1e-4, f"hook vs streaming max abs err {err:.2e}"
    print(f"PASS  tkv_sink hook == streaming sinked_add_token — max abs err {err:.2e}")


def test_tkv_sink_hook_exact_limits_gpt2():
    """On GPT-2, rank=d (warmup >= d, so the basis is complete) and sink+window>=T
    each reproduce exact logits."""
    import torch
    from model_hooks import load_model, unpatch_model
    from tkv_sink_hook import patch_model_tkv_sink

    model, tok = load_model("gpt2", dtype=torch.float32)
    ids = torch.tensor([tok.encode("The quick brown fox jumps over the lazy dog. " * 12)])
    T, d = ids.shape[1], model.config.n_embd // model.config.n_head
    with torch.no_grad():
        ref = model(ids).logits
        for ranks, sink, window in [(d, 4, 8), (4, 8, T - 8)]:
            orig = patch_model_tkv_sink(model, ranks, warmup_len=d, sink_len=sink, window_len=window)
            try:
                err = (model(ids).logits - ref).abs().max().item()
            finally:
                unpatch_model(model, orig)
            assert err < 1e-2, f"ranks={ranks} sink={sink} window={window}: max abs logit err {err:.2e}"
            print(f"PASS  tkv_sink GPT-2 exact limit ranks={ranks} sink+window={sink + window} (T={T}) — err {err:.2e}")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_inner_loop_update,
        test_query,
        test_sinked_add_token_streaming,
        test_sinked_add_token_first_eviction,
        test_sinked_add_token_attention,
        test_sketch_cold_start_shape,
        test_sketch_cold_start_reconstruction,
        test_sketch_cold_start_importance_effect,
        test_oja_error_at_most_one,
        test_oja_error_decreases_with_rank,
        test_oja_exact_when_rank_covers,
        test_oja_sink_hook_full_rank_exact,
        test_oja_sink_hook_sink_window_exact,
        test_oja_sink_hook_causal,
        test_sink_hooks_pre_rope_full_rank_exact,
        test_streaming_llm_hook_exact_limit,
        test_residual_retention,
        test_select_hook_exact_limits,
        test_tkv_sink_hook_matches_streaming,
        test_tkv_sink_hook_exact_limits_gpt2,
    ]

    passed, failed = 0, []
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed.append((t.__name__, e))
            print(f"FAIL  {t.__name__} — {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    if failed:
        raise SystemExit(1)
