# TKV: what a low-rank KV cache can tell you about which tokens to keep

A KV cache has to decide what to throw away. The usual signal is attention — keep
what past queries looked at (H2O, SnapKV). That needs attention probabilities
(which fused kernels don't expose) and says nothing about queries that haven't
arrived yet.

**TKV** fits a small basis to the pre-RoPE keys of the first 512 tokens and freezes
it. The useful product turns out not to be the compressed storage but a *signal*:
each token's **projection residual** `ρ = ‖k − BᵀBk‖`, i.e. how much of the token the
basis cannot explain. Tokens carrying new information are the ones it explains worst.

- **TKV-S** — evict by residual (attention-free, query-agnostic, `O(rd)` per token).
- **TKV-R** — keep the non-selected tokens as rank-R projections instead of evicting
  them, and hold the top-M residual tokens exactly.

## The result

No query-agnostic rule that *evicts* tokens can bound its error: a later query may
align with exactly what was dropped (Prop. 2 in the paper). Keeping projections
admits a bound, and ranking by residual is what minimizes it (Prop. 1) — so TKV-R is
the only configuration here with a worst-case guarantee.

Needle retrieval, context compressed **before** the question, matched KV memory,
30 held-out prompts (top-1 accuracy):

| | TinyLlama-1.1B | Mistral-7B | Qwen2.5-14B |
|---|---|---|---|
| exact (full cache) | 1.00 | 1.00 | 1.00 |
| StreamingLLM | 0.60 | 0.60 | 0.60 |
| H2O | 0.77 | 0.67 | 0.33 |
| SnapKV (query-agnostic) | 0.67 | 0.93 | 0.73 |
| **TKV-S** | **0.93** | **0.93** | 0.63 |
| **TKV-R** (guaranteed) | 0.77 | **0.93** | 0.53 |

A one-number diagnostic predicts where it works: the residual percentile of the
tokens you need (0.66 / 0.64 / 0.56 for the three models) orders the outcomes exactly.

**Where it does not pay off:** as *storage*, a low-rank cache loses to plain eviction
on perplexity at every budget we tested. The basis is worth more as a measurement
than as a container.

## Layout

| Path | What |
|---|---|
| `src/tkv_jax.py` | TKV-R decode in JAX, XLA-compilable, static shapes, fixed-capacity heap |
| `src/profile_jax.py` | decode-step profiling vs full cache and eviction, sweeping context and batch |
| `src/select_hook.py` | query-agnostic selection (residual / SnapKV / H2O / random) as a HF attention hook |
| `src/tkv_sink_hook.py` | streaming TKV / TKV-R cache (sinks + window + projections) |
| `src/streaming_llm_hook.py` | StreamingLLM baseline at matched memory |
| `src/oja_kv.py`, `src/oja_sink_hook.py` | corrected OjaKV baseline |
| `src/select_benchmark.py` | needle retrieval under query-agnostic compression |
| `src/wikitext_benchmark.py` | WikiText-2 / PG-19 / distribution-shift perplexity |
| `src/residual_distribution.py` | residual distribution of needle tokens (paper Figure 1) |
| `modal_eval.py` | all GPU runs (Modal); `--select`, `--tkvr`, `--tkvr-dev`, `--jaxprof`, … |
| `paper/` | paper and 34"×34" poster; `paper/archive/` holds superseded drafts |
| `results/` | every log behind every number in the paper |

## Reproducing

```bash
cd src && python test_contributions.py     # 20 tests: hook equivalences, exact limits, causality
cd src && python test_tkv_jax.py           # XLA decode == dense reference; rank=d == full attention

# local, free (TinyLlama on MPS/CPU)
python select_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device mps \
  --ctx-len 2048 --budget 0.672 --configs residual/evict/8/256 snapkv/evict/0/256 h2o/evict/0/256

# GPU runs (Modal); each writes to results/modal/
modal run --detach modal_eval.py --select --only mistral7b
modal run --detach modal_eval.py --jaxprof
```

Every hook is verified against exact attention in its degenerate limits (full rank,
or a window covering the sequence), and the teacher-forced simulation is checked
against token-by-token streaming (max abs error 7e-7).

## Limits

One synthetic needle format, 2K–4K context, 30 prompts per cell; model-dependent
(weaker than SnapKV on Qwen2.5-14B); no comparison against KVzip; no quantization
baseline. See the paper's Limitations section — none of these are hidden.
