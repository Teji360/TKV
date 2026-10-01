"""
Needle-in-a-Haystack retrieval for KV cache compression methods.

A one-sentence needle ("The special code word is <word>.") is inserted into
real WikiText-2 text at several depths; the prompt ends with a question whose
answer is the needle word. Reports P(needle token) and top-1 accuracy of the
next-token prediction, per method / rank / depth.

This is where eviction and low-rank compression differ: StreamingLLM drops
tokens outside its sink + recent window entirely, while TKV / OjaKV keep a
compressed copy of every token. Methods come from wikitext_benchmark.METHODS
(exact, tkv_sink, oja_sink, streaming_llm), with the same memory matching.
"""

import argparse
from typing import List, Tuple

import torch

from model_hooks import load_model, unpatch_model
from tkv_sink_hook import _DIAG
from wikitext_benchmark import METHODS, load_wikitext2, memory_ratio

NEEDLE_WORDS = ["blue", "gold", "dark", "soft", "wild", "cold"]
NEEDLE_TEMPLATE = " The special code word is {word}. Remember it. "
QUESTION = "\n\nQuestion: what is the special code word? Answer: The special code word is"


def build_prompt(tok, haystack: List[int], ctx_len: int, word: str, depth: float,
                 offset: int) -> Tuple[List[int], int, int, int]:
    """Returns (input_ids, needle_token_id, needle_start, needle_end)."""
    needle = tok.encode(NEEDLE_TEMPLATE.format(word=word), add_special_tokens=False)
    question = tok.encode(QUESTION, add_special_tokens=False)
    bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
    n_fill = ctx_len - len(bos) - len(needle) - len(question)
    fill = haystack[offset:offset + n_fill]
    pos = int(n_fill * depth)
    ids = bos + fill[:pos] + needle + fill[pos:] + question
    answer = tok.encode(" " + word, add_special_tokens=False)[0]
    return ids, answer, len(bos) + pos, len(bos) + pos + len(needle)


@torch.no_grad()
def retrieve(model, ids: List[int], answer: int, device: str) -> Tuple[float, bool]:
    logits = model(torch.tensor([ids], device=device), use_cache=False).logits[0, -1].float()
    return torch.softmax(logits, -1)[answer].item(), int(logits.argmax()) == answer


def run(args):
    print(f"\nLoading {args.model} on {args.device}...")
    model, tok = load_model(args.model, device=args.device)
    cfg = model.config
    n_layers = cfg.num_hidden_layers
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    haystack = tok.encode(load_wikitext2(), add_special_tokens=False)
    words = NEEDLE_WORDS[:args.trials]
    print(f"  ctx={args.ctx_lens} depths={args.depths} words={words} warmup={args.warmup_len} "
          f"sink={args.sink_len} window={args.window_len} pre_rope={args.pre_rope}")

    configs = [("exact", "-", None)]
    for method in args.methods:
        ranks = args.oja_ranks if method == "oja_sink" else args.ranks
        configs += [(method, f"r={r}", [r]) for r in ranks]

    header = f"{'ctx':>6} {'method':>13} {'ranks':>6} {'mem':>6} " + " ".join(f"{'d=' + str(x):>11}" for x in args.depths) + f" {'mean P':>7} {'acc':>5}"
    rows = []
    for ctx in args.ctx_lens:
        for method, label, ranks in configs:
            orig = None
            if ranks is not None:
                extra = {"ctx_len": ctx} if method == "streaming_llm" else {"pre_rope": args.pre_rope}
                if method == "tkv_sink" and args.retain_frac > 0:
                    extra.update(retain_frac=args.retain_frac, retain_by=args.retain_by)
                if method == "tkv_sink" and args.center:
                    extra["center"] = True
                orig = METHODS[method](model, ranks, warmup_len=args.warmup_len,
                                       sink_len=args.sink_len, window_len=args.window_len, **extra)
            try:
                per_depth, hits = [], 0
                _DIAG["stats"] = []
                for depth in args.depths:
                    ps = []
                    for t, word in enumerate(words):
                        ids, ans, a, e = build_prompt(tok, haystack, ctx, word, depth, offset=t * ctx)
                        _DIAG["span"] = (a, e) if args.diag and ranks is not None else None
                        p, hit = retrieve(model, ids, ans, args.device)
                        _DIAG["span"] = None
                        ps.append(p)
                        hits += hit
                    per_depth.append(sum(ps) / len(ps))
            finally:
                if orig is not None:
                    unpatch_model(model, orig)
            keep = args.retain_frac if method == "tkv_sink" else 0.0
            mem = 1.0 if ranks is None else memory_ratio(ranks, n_layers, ctx, d, args.sink_len,
                                                         args.window_len, keep)
            if keep:
                method = f"{method}+{args.retain_by[:3]}{keep:g}"
            if args.center and ranks is not None and method.startswith("tkv_sink"):
                method += "+c"
            if _DIAG["stats"]:
                kept, pct = zip(*_DIAG["stats"])
                print(f"  DIAG {ctx} {method} {label}: needle tokens kept exact = {sum(kept) / len(kept):.3f}, "
                      f"mean residual percentile = {sum(pct) / len(pct):.3f}", flush=True)
            acc = hits / (len(words) * len(args.depths))
            row = (f"{ctx:>6} {method:>20} {label:>6} {mem:>6.3f} "
                   + " ".join(f"{p:>11.3f}" for p in per_depth)
                   + f" {sum(per_depth) / len(per_depth):>7.3f} {acc:>5.2f}")
            rows.append(row)
            print(f"  NIAH {row}", flush=True)

    print("\n" + "=" * len(header))
    print("Needle-in-a-Haystack: mean P(needle) per depth (higher = better); acc = top-1 accuracy")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    print("\n".join(rows))
    print("=" * len(header))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--ctx-lens", nargs="+", type=int, default=[2048, 4096])
    p.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--trials", type=int, default=3, help="needle words per depth")
    p.add_argument("--methods", nargs="+", default=["tkv_sink", "oja_sink", "streaming_llm"])
    p.add_argument("--ranks", nargs="+", type=int, default=[16, 64])
    p.add_argument("--oja-ranks", nargs="+", type=int, default=[64])
    p.add_argument("--warmup-len", type=int, default=512)
    p.add_argument("--sink-len", type=int, default=64)
    p.add_argument("--window-len", type=int, default=256)
    p.add_argument("--pre-rope", action="store_true")
    p.add_argument("--retain-frac", type=float, default=0.0)
    p.add_argument("--retain-by", default="residual", choices=["residual", "random"])
    p.add_argument("--center", action="store_true", help="tkv_sink: mean-center bases (warmup mean)")
    p.add_argument("--diag", action="store_true", help="report needle-token retention per config")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    run(p.parse_args())
