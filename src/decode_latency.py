"""
GPU decode-step latency: per-token cache update + attention read for one layer
(n_kv KV heads, GQA groups), for exact, TKV (frozen basis) and OjaKV (normalized
Oja step + QR + coefficient rotation). Replaces the paper's CPU/eager timings.

  exact: append k, v; scores q K^T, out = softmax @ V               O(T d)
  tkv:   c = B k (append coeffs); scores (q B^T) C^T, out via C, B  O(R (T + d))
  oja:   tkv read + basis step, QR, C <- C (B_old B_new^T)          O(T R^2)
"""

import argparse

import torch


def _time(fn, iters: int = 50) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters   # ms


def bench(T: int, R: int, d: int = 128, n_kv: int = 8, groups: int = 4, dtype=torch.float16):
    dev = "cuda"
    q = torch.randn(n_kv, groups, 1, d, device=dev, dtype=dtype)
    k_new = torch.randn(n_kv, 1, d, device=dev, dtype=dtype)
    K = torch.randn(n_kv, T, d, device=dev, dtype=dtype)
    V = torch.randn(n_kv, T, d, device=dev, dtype=dtype)
    B = torch.linalg.qr(torch.randn(n_kv, d, R, device=dev))[0].mT.to(dtype)   # (n_kv, R, d)
    Ck = torch.randn(n_kv, T, R, device=dev, dtype=dtype)
    Cv = torch.randn(n_kv, T, R, device=dev, dtype=dtype)
    scale = d ** -0.5

    def exact():
        K[:, -1:] = k_new
        w = torch.softmax((q @ K[:, None].mT) * scale, -1)
        return w @ V[:, None]

    def tkv_read():
        Ck[:, -1:] = k_new @ B.mT
        w = torch.softmax(((q @ B[:, None].mT) @ Ck[:, None].mT) * scale, -1)
        return (w @ Cv[:, None]) @ B[:, None]

    def oja():
        out = tkv_read()
        x = k_new.float()
        eta = 1.0 / (T * ((x * x).sum(-1, keepdim=True) + 1e-9))
        new = B.float() + eta * (x @ B.float().mT).mT @ x
        Bn = torch.linalg.qr(new.mT)[0].mT
        Ck.copy_(Ck @ (B.float() @ Bn.mT).to(dtype))
        return out

    return {"exact": _time(exact), "tkv": _time(tkv_read), "oja": _time(oja)}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", nargs="+", type=int, default=[4096, 32768, 131072])
    p.add_argument("--ranks", nargs="+", type=int, default=[32, 64])
    a = p.parse_args()
    print(f"GPU: {torch.cuda.get_device_name()}  (one layer, 8 KV heads x 4 groups, d=128, fp16)")
    print(f"{'T':>8} {'R':>4} {'exact ms':>9} {'tkv ms':>8} {'oja ms':>8}")
    for T in a.ctx:
        for R in a.ranks:
            r = bench(T, R)
            print(f"{T:>8} {R:>4} {r['exact']:>9.4f} {r['tkv']:>8.4f} {r['oja']:>8.4f}", flush=True)
