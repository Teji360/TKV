# TKV: deployment notes, open questions, talking points

Prep sheet for COLM. Everything here is either measured or explicitly flagged as
unmeasured.

## 30 seconds

> Every KV cache has to decide what to throw away, and the standard signal is
> attention — keep what past queries looked at. That needs attention probabilities
> fused kernels don't expose, and it tells you nothing about the query that hasn't
> arrived yet. We score tokens instead by what a small frozen projection of the keys
> *can't* reproduce: the residual. It's free, it needs no attention, and it works
> before the question exists. There's also a reason to prefer it: if you evict a
> token you can't promise anything to a future query, but if you keep it as a
> projection, keeping the largest residuals is exactly the bound-minimizing choice.

## Three minutes

1. **Setting.** Compression happens *before* the question: cached documents,
   multi-turn, shared prefixes. Attention-based scores are extrapolation there.
2. **Signal.** Fit rank-8 basis on the first 512 pre-RoPE keys, freeze it.
   `ρ = ‖k − BᵀBk‖`, one projection per token, `√(‖k‖² − ‖Bk‖²)` in practice.
3. **Theory.** Evicting admits no query-agnostic bound (a query can align with what
   you dropped). Keeping projections does, and the bound depends only on the largest
   residual you didn't keep — so top-M residual is the simultaneous minimizer.
4. **Results.** At matched memory: ties the best query-agnostic method on Mistral-7B
   (0.93), matches the best baseline on TinyLlama (0.77), loses on Qwen (0.53 vs
   0.73) — and the residual percentile diagnostic predicts that ordering in advance.
5. **Honest negative.** As storage, low-rank loses to eviction on perplexity at every
   budget. The basis is a measuring device, not a container.

## What a serving engineer will ask, and the current answer

| Question | Answer today |
|---|---|
| Compression overhead? | Selection is `O(rd)` per token (r=8) plus one `O(M)` heap compare; no attention probabilities. Prefill SVD is one `512×d` SVD per layer/head. |
| Does it work with FlashAttention / fused kernels? | Yes, that's the point — the criterion never needs attention probabilities. H2O and SnapKV do. |
| Batching? | The JAX implementation is vmapped over batch and KV heads with static shapes. Per-sequence keep-sets mean ragged exact-token sets across a batch; we pad to a fixed M, which is what `profile_jax.py` measures. **Not yet measured: throughput at large batch with realistic mixed sequence lengths.** |
| Paged/blocked KV (vLLM-style)? | Untested. The coefficient array is contiguous per sequence; a paged layout would need block-wise coefficients. Open. |
| Distributed / TP sharding? | Untested. Per-head state shards naturally with heads, and the basis is per-head, so TP looks straightforward, but nothing is measured. |
| Quantization interaction? | Untested and the obvious combination: coefficients are small and probably tolerate low precision better than raw keys. |
| Quality at deployment ratios? | **Weakest point.** We tested 0.59× and 0.21×; serving wants 0.1× or less. At 0.21× retrieval outside the window collapsed in our runs. |
| Long context? | Untested beyond 4K. The frozen basis is fit on 512 tokens — at 128K that's 0.4% of the context and likely stale; periodic refit is the obvious fix and changes the guarantee to per-segment. |
| Prefill cost? | One SVD per layer/head over the warmup window; not yet profiled against prefill attention. |

## What our latency numbers are measured against (read this before quoting them)

`src/profile_jax.py` compares three JAX implementations we wrote:

| | what it is |
|---|---|
| `full` | dense contiguous ring-buffer cache, attention as an einsum + softmax |
| `evict` | the same, with the cache truncated to sinks + a widened window |
| `tkv_r` | the XLA TKV-R cache (`src/tkv_jax.py`) |

**None of these is a production baseline.** There is no PagedAttention, no block
tables, no FlashAttention-class fused kernel. Numbers are reported in fp32 and
bf16; production serves bf16/fp16 with fused paged kernels, which would make the
dense baseline substantially faster than ours. So "TKV-R beats the full cache at
4K-32K" means it beats *our naive dense implementation*, and nothing stronger.

What a fair serving comparison needs, in order: a FlashAttention-class decode
kernel as the baseline; a paged layout for the coefficient array (it is currently
contiguous per sequence, which is the main structural unknown for adoption); and
end-to-end throughput with mixed sequence lengths rather than a single decode step.

At matched KV memory the eviction baseline is the faster kernel in our own
measurements -- both read the same bytes by construction, but eviction is one
contiguous matmul while TKV-R touches four segments and two matmuls. TKV-R's case
rests on retrieval quality and the guarantee, not on decode speed.

### Measured decode step, L40S, one layer (8 KV heads x 4 groups, d=128, rank 32)

ms per decode step at 0.594x KV memory; `full` = our dense ring-buffer baseline.

| ctx | batch | full fp32 | tkv_r fp32 | full bf16 | evict bf16 | **tkv_r bf16** |
|---|---|---|---|---|---|---|
| 4K | 1 | 0.44 | 0.25 | 0.40 | 0.40 | **0.23** |
| 4K | 32 | 2.08 | 1.57 | 1.22 | 0.91 | **0.84** |
| 32K | 1 | 0.85 | 0.65 | 0.61 | 0.44 | **0.44** |
| 32K | 8 | 3.70 | 4.24 | 2.05 | 1.41 | 2.28 |
| 32K | 32 | 13.54 | 10.31 | 6.97 | 4.36 | 5.33 |
| 128K | 1 | 2.02 | 2.27 | 1.25 | 0.93 | 1.18 |
| 128K | 8 | 13.34 | 16.18 | 6.88 | 4.34 | 8.33 |
| 128K | 32 | oom | 39.66 | 26.44 | 15.99 | 20.22 |

bf16 roughly halves both paths, so the ordering is not a dtype artifact. TKV-R beats
our dense baseline at 4K-32K and at 128K/batch-32, loses at batch 8, and eviction is
the fastest kernel almost everywhere. Note the prefill SVD must run in fp32 even for a
bf16 cache -- cuSolver has no bf16 SVD path (`gesvdj`), so the basis is computed in
fp32 and cast down.

## Optimizations, measured

| Change | Effect |
|---|---|
| Fuse the softmax across cache segments (no concat, no re-scan of a length-T array) | 10-35% |
| Donate the state buffers so XLA updates coefficients in place | large; without it every step copied a 33 MB buffer and TKV-R lost to dense everywhere |
| Share the coefficient read across the GQA group | no measurable effect (vmap already shared it) |
| Remove nested `jit`s so donation reaches the buffers | helped on CPU, no effect on GPU |

## Questions to ask them

- Where does your serving stack actually spend KV bandwidth: prefill, decode read,
  or copies on preemption/scheduling?
- Does anything in your stack expose attention probabilities cheaply, or is an
  attention-free criterion a real advantage?
- At what compression ratio would this be interesting? If it's 10×, our current
  numbers don't reach it and I'd want to know before optimizing further.
- Would a per-segment refit (basis re-fit every N tokens) break your caching or
  prefix-sharing assumptions?

## Next experiments, in order

1. 16–32K context at 0.25–0.5× budget (needs the tiled attention path; the
   O(T²) scoring matrix is the current blocker).
2. KVzip as a baseline — the strongest query-agnostic method, currently uncited in
   our numbers.
3. RULER or LongBench instead of a single synthetic needle.
4. Batch-throughput profiling with mixed sequence lengths.
