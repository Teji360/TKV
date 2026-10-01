"""
Long-context perplexity benchmark on WikiText-2 for KV cache compression.

Standard HF strided protocol: fixed ctx_len windows moved by stride; labels on
the context portion are ignored (only the newly revealed tokens are scored).
Every method scores exactly the same token set. PPL is reported over all
scored tokens and over post-freeze tokens (in-window position > warmup_len,
i.e. predicted by a query that sees the frozen-basis compressed cache).

Methods:
  - exact:    unmodified attention (baseline)
  - tkv_sink: frozen-basis TKV + exact sink + sliding recent window (tkv_sink_hook.py)
  - oja_sink: Oja + sink (oja_sink_hook.py), only if that module exists
  - streaming_llm: eviction baseline, sink + recent window memory-matched to
    each rank config (streaming_llm_hook.py)

Datasets (--dataset):
  - wikitext2: WikiText-2 test (standard)
  - pg19:      PG-19 test books concatenated (real long-form text)
  - shift:     distribution shift — every window is warmup_len WikiText-2
               tokens followed by code (codeparrot); needs stride == ctx_len,
               so post-freeze PPL is measured entirely on the shifted domain

Lower perplexity = better.
"""

import argparse
import math
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from typing import List

from model_hooks import load_model, unpatch_model
from streaming_llm_hook import patch_model_streaming_llm
from tkv_sink_hook import patch_model_tkv_sink

# name -> patch fn(model, ranks, warmup_len, sink_len, window_len) -> restore handle; None = exact
METHODS = {"exact": None, "tkv_sink": patch_model_tkv_sink, "streaming_llm": patch_model_streaming_llm}
try:
    from oja_sink_hook import patch_model_oja_sink
    METHODS["oja_sink"] = patch_model_oja_sink
except ImportError:
    pass


def load_wikitext2(split: str = "test") -> str:
    """Standard HF text: the full split joined with blank lines."""
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return "\n\n".join(ds["text"])


def _stream_text(name: str, field: str, n_chars: int, split: str = "train") -> List[str]:
    """First documents of a streamed HF dataset until n_chars characters are collected."""
    docs, total = [], 0
    for ex in load_dataset(name, split=split, streaming=True):
        docs.append(ex[field])
        total += len(ex[field])
        if total >= n_chars:
            break
    return docs


def load_corpus_ids(dataset: str, tok, ctx_len: int, stride: int, warmup_len: int,
                    max_windows: int) -> torch.Tensor:
    """Token ids of the evaluation stream for --dataset."""
    if dataset == "wikitext2":
        return torch.tensor(tok(load_wikitext2()).input_ids, dtype=torch.long)
    n_windows = max_windows or 40
    n_chars = 6 * (ctx_len + stride * n_windows)   # ~4 chars/token, with margin
    if dataset == "pg19":
        text = "\n\n".join(_stream_text("emozilla/pg19-test", "text", n_chars, split="test"))
        return torch.tensor(tok(text).input_ids, dtype=torch.long)
    assert dataset == "shift" and stride == ctx_len, "shift needs --stride == --ctx-len"
    wiki = tok(load_wikitext2()).input_ids
    code = tok("\n\n".join(_stream_text("codeparrot/codeparrot-clean-valid", "content",
                                           6 * ctx_len * n_windows))).input_ids
    n_code = ctx_len - warmup_len
    stream = []
    for w in range(n_windows):
        stream += wiki[w * warmup_len:(w + 1) * warmup_len] + code[w * n_code:(w + 1) * n_code]
    return torch.tensor(stream, dtype=torch.long)


@torch.no_grad()
def sliding_perplexity(
    model,
    ids: torch.Tensor,       # (N,) token ids of the whole corpus
    ctx_len: int = 2048,
    stride: int = 512,
    warmup_len: int = 512,
    device: str = "cpu",
    max_windows: int = None,
) -> dict:
    """
    HF strided perplexity. Window w covers ids[begin:begin+ctx_len]; its targets
    are the tokens not already scored by the previous window (the whole first
    window, then the last `stride` tokens). Returns NLL sums / token counts for
    all scored targets and for post-freeze targets (in-window position > warmup_len).
    """
    N = ids.shape[0]
    nll_all = nll_post = 0.0
    n_all = n_post = n_windows = 0
    prev_end = 0
    t0 = time.time()
    for begin in range(0, N, stride):
        end = min(begin + ctx_len, N)
        trg_len = end - prev_end
        chunk = ids[begin:end].to(device)
        L = end - begin

        logits = model(chunk[None], use_cache=False).logits[0].float()
        nll = F.cross_entropy(logits[:-1], chunk[1:], reduction="none")   # target at pos 1..L-1
        pos = torch.arange(1, L, device=device)
        scored = pos >= L - trg_len
        post = scored & (pos > warmup_len)
        nll_all += nll[scored].sum().item()
        n_all += int(scored.sum())
        nll_post += nll[post].sum().item()
        n_post += int(post.sum())

        n_windows += 1
        prev_end = end
        if end == N or (max_windows is not None and n_windows >= max_windows):
            break

    return {
        "ppl": math.exp(nll_all / n_all),
        "ppl_post": math.exp(nll_post / n_post) if n_post else float("nan"),
        "n_all": n_all,
        "n_post": n_post,
        "n_windows": n_windows,
        "sec_per_window": (time.time() - t0) / n_windows,
    }


def memory_ratio(ranks: List[int], n_layers: int, T: int, d: int, sink_len: int, window_len: int,
                 retain_frac: float = 0.0) -> float:
    """Per-head KV bytes relative to exact: ((T*R + R*d)*2 + (S+W+kept)*d*2) / (T*d*2), averaged
    over layers; kept = retain_frac * (T - S - W) exact middle tokens (charged on top of their coefficients)."""
    per_layer = [ranks[min(l, len(ranks) - 1)] for l in range(n_layers)]
    kept = retain_frac * max(T - sink_len - window_len, 0)
    comp = sum((T * R + R * d) * 2 + (sink_len + window_len + kept) * d * 2 for R in per_layer)
    return comp / (n_layers * T * d * 2)


def quarter_ranks(quarters: List[int], n_layers: int) -> List[int]:
    """Expand 4 per-quarter ranks to a per-layer list (first quarters take the remainder)."""
    q, r = divmod(n_layers, 4)
    out = []
    for i, rank in enumerate(quarters):
        out += [rank] * (q + (1 if r > i else 0))
    return out


def run_ppl(args) -> list:
    print(f"\nLoading {args.model} on {args.device} ({args.dtype})...")
    model, tok = load_model(args.model, device=args.device, dtype=getattr(torch, args.dtype))
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    print(f"Loading {args.dataset}...")
    ids = load_corpus_ids(args.dataset, tok, args.ctx_len, args.stride, args.warmup_len,
                          args.max_windows)
    print(f"  corpus tokens: {ids.shape[0]:,}  layers={n_layers}  head_dim={d}")
    print(f"  ctx={args.ctx_len} stride={args.stride} warmup={args.warmup_len} "
          f"sink={args.sink_len} window={args.window_len} max_windows={args.max_windows} pre_rope={args.pre_rope}")

    rank_cfgs = [(f"r={r}", [r]) for r in args.ranks]
    if args.pyramid:
        rank_cfgs.append((f"pyr{args.pyramid}", quarter_ranks(args.pyramid, n_layers)))

    runs = []
    for method in args.methods:
        if method not in METHODS:
            print(f"  skipping {method}: not available")
            continue
        for label, ranks in ([("-", None)] if METHODS[method] is None else rank_cfgs):
            orig = None
            if METHODS[method] is not None:
                extra = {"pre_rope": True} if args.pre_rope and method != "streaming_llm" else {}
                if method == "streaming_llm":
                    extra["ctx_len"] = args.ctx_len
                if method == "tkv_sink" and args.retain_frac > 0:
                    extra.update(retain_frac=args.retain_frac, retain_by=args.retain_by)
                if method == "tkv_sink" and args.center:
                    extra["center"] = True
                orig = METHODS[method](model, ranks, warmup_len=args.warmup_len,
                                       sink_len=args.sink_len, window_len=args.window_len, **extra)
            try:
                res = sliding_perplexity(model, ids, args.ctx_len, args.stride, args.warmup_len,
                                         args.device, args.max_windows)
            finally:
                if orig is not None:
                    unpatch_model(model, orig)
            keep = args.retain_frac if method == "tkv_sink" else 0.0
            res["method"] = (method + (f"+{args.retain_by[:3]}{keep:g}" if keep else "")
                             + ("+c" if args.center and method == "tkv_sink" else ""))
            res["ranks"] = label
            res["mem"] = 1.0 if ranks is None else memory_ratio(
                ranks, n_layers, args.ctx_len, d, args.sink_len, args.window_len, keep)
            runs.append(res)
            print(f"  {res['method']:>9} {label:>22}  ppl={res['ppl']:.3f}  ppl_post={res['ppl_post']:.3f}  "
                  f"mem={res['mem']:.3f}  ({res['sec_per_window']:.2f}s/window)", flush=True)
    return runs


def print_ppl_table(runs: list):
    header = f"{'method':>9} {'ranks':>22} {'PPL(all)':>9} {'PPL(post)':>10} {'mem':>6} {'s/win':>6}"
    print("\n" + "=" * len(header))
    print("Perplexity (lower = better); mem = KV bytes / exact")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in runs:
        print(f"{r['method']:>9} {r['ranks']:>22} {r['ppl']:>9.3f} {r['ppl_post']:>10.3f} "
              f"{r['mem']:>6.3f} {r['sec_per_window']:>6.2f}")
    print("=" * len(header))
    r = runs[0]
    print(f"tokens scored: all={r['n_all']:,}  post-freeze={r['n_post']:,}  windows={r['n_windows']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--methods", nargs="+", default=["exact", "tkv_sink", "oja_sink"])
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "pg19", "shift"])
    parser.add_argument("--ctx-len", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--sink-len", type=int, default=64)
    parser.add_argument("--window-len", type=int, default=256)
    parser.add_argument("--warmup-len", type=int, default=512)
    parser.add_argument("--ranks", nargs="*", type=int, default=[32], help="uniform ranks to sweep")
    parser.add_argument("--pyramid", nargs=4, type=int, default=None,
                        help="also run per-layer-quarter ranks, e.g. --pyramid 96 64 48 32")
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-windows", type=int, default=None, help="deterministic first-N windows")
    parser.add_argument("--pre-rope", action="store_true", help="tkv_sink: compress keys before RoPE")
    parser.add_argument("--retain-frac", type=float, default=0.0,
                        help="tkv_sink: also keep this fraction of middle tokens exact")
    parser.add_argument("--retain-by", default="residual", choices=["residual", "random"])
    parser.add_argument("--center", action="store_true", help="tkv_sink: mean-center bases (warmup mean)")
    args = parser.parse_args()

    print_ppl_table(run_ppl(args))
