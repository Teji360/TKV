"""
Needle retrieval under query-agnostic compression: the context is compressed
once, before the question arrives (select_hook.py), at a fixed KV budget.

Config spec: criterion/mode/rank/window
  criterion: residual | snapkv | h2o | random | snapkv_q | none
  mode:      lowrank (non-selected middle tokens kept as rank-R coefficients) | evict (dropped)
The number of exact selected tokens M follows from --budget (fraction of the full cache).
Example: residual/lowrank/32/512 (TKV-R), snapkv/evict/0/256 (SnapKV), none/evict/0/2368 (StreamingLLM).
"""

import argparse

import torch

from model_hooks import load_model, unpatch_model
from needle_benchmark import NEEDLE_WORDS, QUESTION, build_prompt, retrieve
from select_hook import _SEL, patch_model_select, select_budget
from wikitext_benchmark import load_wikitext2


def run(args):
    print(f"\nLoading {args.model} on {args.device}...")
    model, tok = load_model(args.model, device=args.device)
    cfg = model.config
    d = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    haystack = tok.encode(load_wikitext2(), add_special_tokens=False)
    q_len = len(tok.encode(QUESTION, add_special_tokens=False))
    words = NEEDLE_WORDS[:args.trials]
    T = args.ctx_len
    print(f"  ctx={T} budget={args.budget} depths={args.depths} words={words} warmup={args.warmup_len} "
          f"sink={args.sink_len} question_len={q_len} offset_base={args.offset_base}")

    header = (f"{'config':>26} {'M':>5} " + " ".join(f"{'d=' + str(x):>7}" for x in args.depths)
              + f" {'mean P':>7} {'acc':>5}")
    rows = []
    for spec in ["exact"] + args.configs:
        orig = None
        M = 0
        if spec != "exact":
            crit, mode, rank, window = spec.split("/")
            rank, window, lowrank = int(rank), int(window), mode == "lowrank"
            M = 0 if crit == "none" else select_budget(args.budget, T, d, rank, lowrank, args.sink_len, window)
            orig = patch_model_select(model, crit if crit != "none" else "random", lowrank, rank, M,
                                      args.warmup_len, args.sink_len, window, pre_rope=True)
        try:
            per_depth, hits = [], 0
            for depth in args.depths:
                ps = []
                for t, word in enumerate(words):
                    ids, ans, _, _ = build_prompt(tok, haystack, T, word, depth,
                                                 offset=(t + args.offset_base) * T)
                    _SEL["compress_at"] = len(ids) - q_len
                    p, hit = retrieve(model, ids, ans, args.device)
                    ps.append(p)
                    hits += hit
                per_depth.append(sum(ps) / len(ps))
        finally:
            _SEL["compress_at"] = None
            if orig is not None:
                unpatch_model(model, orig)
        acc = hits / (len(words) * len(args.depths))
        row = (f"{spec:>26} {M:>5} " + " ".join(f"{p:>7.3f}" for p in per_depth)
               + f" {sum(per_depth) / len(per_depth):>7.3f} {acc:>5.2f}")
        rows.append(row)
        print(f"  SEL {row}", flush=True)

    print("\n" + "=" * len(header))
    print(f"Query-agnostic needle retrieval at {args.budget}x KV memory (mean P per depth; acc = top-1)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    print("\n".join(rows))
    print("=" * len(header))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--ctx-len", type=int, default=4096)
    p.add_argument("--budget", type=float, default=0.594)
    p.add_argument("--depths", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9])
    p.add_argument("--trials", type=int, default=6, help="needle words per depth")
    p.add_argument("--offset-base", type=int, default=0,
                   help="shift haystack windows by this many contexts (>0 gives a disjoint prompt set)")
    p.add_argument("--warmup-len", type=int, default=512)
    p.add_argument("--sink-len", type=int, default=64)
    p.add_argument("--configs", nargs="+", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    run(p.parse_args())
