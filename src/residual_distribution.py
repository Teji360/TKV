"""
Residual distribution of needle tokens (figure data, no attention needed).

Captures each layer's pre-RoPE keys (k_proj of the attention input), fits the
rank-r TKV basis on the first warmup_len tokens per KV head, and reports where
the needle-sentence tokens fall in the residual distribution of the middle
tokens. Prints histogram bin counts of the needle percentile, ready to paste
into pgfplots.
"""

import argparse

import torch

from model_hooks import load_model
from needle_benchmark import NEEDLE_WORDS, build_prompt
from wikitext_benchmark import load_wikitext2


@torch.no_grad()
def needle_percentiles(model, tok, haystack, ctx, rank, warmup, sink, window, depths, words):
    """Percentile of each needle token among middle-token residuals, per layer/head/prompt."""
    layers = model.model.layers
    captured = {}

    def hook(idx):
        def fn(module, args, kwargs, output):
            h = kwargs.get("hidden_states", args[0] if args else None)
            captured[idx] = module.k_proj(h).detach()          # (1, T, n_kv*d), pre-RoPE
        return fn

    handles = [l.self_attn.register_forward_hook(hook(i), with_kwargs=True) for i, l in enumerate(layers)]
    cfg = model.config
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    pct = []
    try:
        for depth in depths:
            for t, word in enumerate(words):
                ids, _, a, e = build_prompt(tok, haystack, ctx, word, depth, offset=t * ctx)
                captured.clear()
                model(torch.tensor([ids], device=model.device), use_cache=False)
                for idx in range(len(layers)):
                    k = captured[idx][0].float().view(-1, captured[idx].shape[-1] // d, d).transpose(0, 1)
                    T0, r = min(warmup, k.shape[1]), min(rank, d)
                    B = torch.linalg.svd(k[:, :T0].cpu(), full_matrices=False)[2][:, :r]
                    res = (k.cpu() - (k.cpu() @ B.mT) @ B).norm(dim=-1)        # (n_kv, T)
                    mid = res[:, sink:k.shape[1] - window]
                    needle = res[:, a:e]
                    pct.append((mid[:, None, :] < needle[:, :, None]).float().mean(-1))  # (n_kv, n_needle)
    finally:
        for h in handles:
            h.remove()
    return torch.cat([p.flatten() for p in pct])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--ctx-len", type=int, default=2048)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--warmup-len", type=int, default=512)
    p.add_argument("--sink-len", type=int, default=64)
    p.add_argument("--window-len", type=int, default=256)
    p.add_argument("--depths", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--bins", type=int, default=20)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    a = p.parse_args()

    model, tok = load_model(a.model, device=a.device)
    haystack = tok.encode(load_wikitext2(), add_special_tokens=False)
    pct = needle_percentiles(model, tok, haystack, a.ctx_len, a.rank, a.warmup_len, a.sink_len,
                             a.window_len, a.depths, NEEDLE_WORDS[:a.trials])
    counts = torch.histc(pct, bins=a.bins, min=0.0, max=1.0)
    frac = (counts / counts.sum()).tolist()
    print(f"\n{a.model}  ctx={a.ctx_len} rank={a.rank}  n={pct.numel():,} (layer x kv-head x needle token)")
    print(f"mean percentile = {pct.mean():.3f}   above 0.5: {(pct > 0.5).float().mean():.3f}   "
          f"above 0.9: {(pct > 0.9).float().mean():.3f}")
    print("pgfplots coordinates (bin center, fraction):")
    print(" ".join(f"({(i + 0.5) / a.bins:.3f},{f:.4f})" for i, f in enumerate(frac)))
