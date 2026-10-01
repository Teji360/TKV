"""
Runs the TKV evaluation matrix on Modal GPUs, sized for a ~$30 budget.

  modal run modal_eval.py --smoke          # 2 windows per config: checks the pipeline, measures s/window
  modal run modal_eval.py                  # full matrix (20 windows per config)
  modal run modal_eval.py --windows 10     # cheaper
  modal run modal_eval.py --needle         # needle-in-a-haystack retrieval jobs only
  modal run modal_eval.py --hybrid         # rank-vs-window allocation sweep at a fixed memory budget
  modal run modal_eval.py --retention      # residual-guided exact retention at the same budget
  modal run modal_eval.py --diag           # needle-token retention diagnostics + mean-centered bases
  modal run modal_eval.py --select --only mistral7b   # query-agnostic selection: residual vs SnapKV/H2O
  modal run modal_eval.py --jaxprof        # XLA decode profiling of the JAX TKV-R cache (GPU)

Don't run two `modal run` invocations of this app at once: when one finishes it
can stop the other (observed: APP_STATE_STOPPED on the concurrent run).

One container per (model, dataset) runs its configs back to back, so each
model loads once. Timeouts cap the worst case: 3 x 1 h L40S (Mistral) +
3 x 1.5 h A100-80GB (Qwen) ~= $17 at list prices; partial output is kept on timeout.

Needs a Modal secret named `huggingface` with HF_TOKEN (Mistral is gated).
Writes one log per job to results/modal/ and a summary to results/modal/summary.md.
"""

import pathlib
import re
import subprocess

import modal

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "results" / "modal"

app = modal.App("tkv-eval")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.10.0", "transformers==5.9.0", "datasets==4.8.5", "accelerate",
                 "jax==0.9.0", "numpy", "sentencepiece", "protobuf")
    .env({"HF_HOME": "/hf"})
    .add_local_dir(ROOT / "src", "/root/src", ignore=["__pycache__", "*.pdf"])
)
# JAX-on-GPU image for the XLA decode profiler (torch not needed there)
jax_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jax[cuda12]==0.9.0", "numpy")
    .add_local_dir(ROOT / "src", "/root/src", ignore=["__pycache__", "*.pdf"])
)

hf_cache = modal.Volume.from_name("tkv-hf-cache", create_if_missing=True)
# job logs are also written here remotely, so results survive a dead local client:
#   modal volume get tkv-results / results/modal/
results_vol = modal.Volume.from_name("tkv-results", create_if_missing=True)
common = dict(image=image, volumes={"/hf": hf_cache, "/results": results_vol},
              secrets=[modal.Secret.from_name("huggingface")])

MODELS = {"mistral7b": "mistralai/Mistral-7B-v0.1", "qwen14b": "Qwen/Qwen2.5-14B"}
BASE = ["--ctx-len", "4096", "--warmup-len", "512", "--sink-len", "64", "--window-len", "256",
        "--device", "cuda", "--pre-rope"]


def jobs(windows: int) -> list:
    """(model key, job name, [argument lists run sequentially in one container])."""
    w = ["--max-windows", str(windows)]
    out = []
    for m, hf in MODELS.items():
        wiki = ["--model", hf, "--dataset", "wikitext2", "--stride", "1024"] + BASE + w
        out.append((m, f"{m}_wikitext2", [
            wiki + ["--methods", "exact", "tkv_sink", "streaming_llm",
                    "--ranks", "16", "32", "64", "--pyramid", "96", "64", "48", "32"],
            wiki + ["--methods", "oja_sink", "--ranks", "64"],
            # reference: the paper's original post-RoPE TKV
            [a for a in wiki if a != "--pre-rope"] + ["--methods", "tkv_sink", "--ranks", "64"],
        ]))
        out.append((m, f"{m}_pg19", [
            ["--model", hf, "--dataset", "pg19", "--stride", "1024"] + BASE + w
            + ["--methods", "exact", "tkv_sink", "streaming_llm", "--ranks", "64"],
        ]))
        out.append((m, f"{m}_shift", [
            ["--model", hf, "--dataset", "shift", "--stride", "4096"] + BASE + w
            + ["--methods", "exact", "tkv_sink", "streaming_llm", "--ranks", "16", "64"],
            ["--model", hf, "--dataset", "shift", "--stride", "4096"] + BASE + w
            + ["--methods", "oja_sink", "--ranks", "64"],
        ]))
    return out


def needle_jobs() -> list:
    common_args = ["--depths", "0.1", "0.3", "0.5", "0.7", "0.9", "--trials", "3", "--warmup-len", "512",
                   "--sink-len", "64", "--window-len", "256", "--pre-rope", "--device", "cuda"]
    return [
        ("mistral7b", "mistral7b_needle", [["needle", "--model", MODELS["mistral7b"], "--ctx-lens", "2048", "4096",
                                            "--methods", "tkv_sink", "oja_sink", "streaming_llm"] + common_args]),
        ("qwen14b", "qwen14b_needle", [["needle", "--model", MODELS["qwen14b"], "--ctx-lens", "4096", "8192",
                                        "--methods", "tkv_sink", "streaming_llm"] + common_args]),
    ]


def hybrid_jobs(windows: int, budget_rank: int = 64, ctx: int = 4096, d: int = 128) -> list:
    """
    Fixed KV budget = TKV+Sink at rank budget_rank with a 256-token window
    (= StreamingLLM matched window). Spend it as rank R for the compressed
    middle plus an exact window W: each unit of rank costs ctx/d + 1 tokens.
    R=0 (StreamingLLM) and R=budget_rank (TKV+Sink) are the existing endpoints.
    """
    out = []
    for m, hf in MODELS.items():
        cmds = []
        for R in (16, 32, 48):
            W = 256 + (budget_rank - R) * (ctx // d + 1)
            cfg = ["--model", hf, "--ctx-len", str(ctx), "--warmup-len", "512", "--sink-len", "64",
                   "--window-len", str(W), "--device", "cuda", "--pre-rope"]
            cmds.append(cfg + ["--dataset", "wikitext2", "--stride", "1024", "--max-windows", str(windows),
                               "--methods", "tkv_sink", "--ranks", str(R)])
            cmds.append(["needle"] + cfg + ["--ctx-lens", str(ctx), "--depths", "0.1", "0.3", "0.5", "0.7",
                                            "0.9", "--trials", "3", "--methods", "tkv_sink", "--ranks", str(R)])
        out.append((m, f"{m}_hybrid", cmds))
    return out


def retention_jobs(windows: int) -> list:
    """
    Residual retention at the rank-64 budget (0.594x, ctx 4096, d=128): rank R costs
    T*R/d + R token-equivalents, sink + window S + W, the rest is spent on keeping
    retain_frac of the middle tokens exact (largest key residual, or random as control).
    """
    configs = [(16, 768, 0.3284, "residual"), (16, 768, 0.3284, "random"),
               (16, 256, 0.4195, "residual"), (32, 512, 0.2273, "residual")]
    out = []
    for m, hf in MODELS.items():
        cmds = []
        for R, W, f, by in configs:
            cfg = ["--model", hf, "--warmup-len", "512", "--sink-len", "64", "--window-len", str(W),
                   "--device", "cuda", "--pre-rope", "--methods", "tkv_sink", "--ranks", str(R),
                   "--retain-frac", str(f), "--retain-by", by]
            cmds.append(cfg + ["--dataset", "wikitext2", "--ctx-len", "4096", "--stride", "1024",
                               "--max-windows", str(windows)])
            cmds.append(["needle"] + cfg + ["--ctx-lens", "4096", "--depths", "0.1", "0.3", "0.5",
                                            "0.7", "0.9", "--trials", "3"])
        out.append((m, f"{m}_retention", cmds))
    return out


def diag_jobs(windows: int) -> list:
    """r16 / window 768 / 32.84% retention (0.594x): does retention keep the needle's tokens,
    residual vs random, and does mean-centering the basis change that (and perplexity)?"""
    out = []
    for m, hf in MODELS.items():
        base = ["--model", hf, "--warmup-len", "512", "--sink-len", "64", "--window-len", "768",
                "--device", "cuda", "--pre-rope", "--methods", "tkv_sink", "--ranks", "16",
                "--retain-frac", "0.3284"]
        needle = ["--ctx-lens", "4096", "--depths", "0.1", "0.3", "0.5", "0.7", "0.9", "--trials", "3", "--diag"]
        cmds = [["needle"] + base + ["--retain-by", by] + needle for by in ("residual", "random")]
        cmds.append(["needle"] + base + ["--retain-by", "residual", "--center"] + needle)
        cmds.append(base + ["--retain-by", "residual", "--center", "--dataset", "wikitext2", "--ctx-len", "4096",
                            "--stride", "1024", "--max-windows", str(windows)])
        out.append((m, f"{m}_diag", cmds))
    return out


TKVR_CONFIGS = ["residual/lowrank/8/256", "residual/lowrank/16/256", "residual/lowrank/32/512",
                "residualkv/lowrank/8/256"]

SELECT_CONFIGS = ["none/evict/0/2368", "snapkv/evict/0/256", "h2o/evict/0/256", "random/evict/0/256",
                  "residual/evict/8/256", "residual/evict/16/256", "residual/evict/32/256",
                  "residual/lowrank/32/512", "snapkv_q/evict/0/256"]


# reduced set for Qwen2.5-14B (budget)
SELECT_CONFIGS_QWEN = ["none/evict/0/2368", "snapkv/evict/0/256", "h2o/evict/0/256", "residual/evict/8/256"]


def tkvr_dev_jobs() -> list:
    """Same allocations on a DISJOINT prompt set (haystack windows shifted by 20 contexts):
    picks the rank/window split without touching the prompts the paper reports."""
    return [(m, f"{m}_tkvr_dev", [["select", "--model", hf, "--device", "cuda", "--ctx-len", "4096",
                                   "--budget", "0.594", "--trials", "6", "--offset-base", "20",
                                   "--configs", "residual/lowrank/8/256", "residual/lowrank/16/256",
                                   "residual/lowrank/32/512"]])
            for m, hf in MODELS.items()]


def tkvr_jobs() -> list:
    """TKV-R (projection + residual retention) under the corrected budget: a retained token
    no longer pays for coefficients it does not need."""
    return [(m, f"{m}_tkvr", [["select", "--model", hf, "--device", "cuda", "--ctx-len", "4096",
                               "--budget", "0.594", "--trials", "6", "--configs"] + TKVR_CONFIGS])
            for m, hf in MODELS.items()]


def select_jobs() -> list:
    return [(m, f"{m}_select", [["select", "--model", hf, "--device", "cuda", "--ctx-len", "4096",
                                 "--budget", "0.594", "--trials", "6", "--configs"]
                                + (SELECT_CONFIGS if m == "mistral7b" else SELECT_CONFIGS_QWEN)])
            for m, hf in MODELS.items()]


def _run_all(name: str, cmds: list, budget_s: int) -> tuple:
    """Runs cmds sequentially within budget_s seconds; keeps partial output on timeout."""
    import time
    logs, t_end = [], time.time() + budget_s
    for args in cmds:
        script = {"latency": "decode_latency.py", "needle": "needle_benchmark.py",
                  "select": "select_benchmark.py"}.get(args[0], "wikitext_benchmark.py")
        args = args[1:] if args[0] in ("latency", "needle", "select") else args
        try:
            p = subprocess.run(["python", script] + args, cwd="/root/src", capture_output=True,
                               text=True, timeout=max(60, t_end - time.time()))
            logs.append(p.stdout + (f"\n[stderr]\n{p.stderr[-4000:]}" if p.returncode else ""))
        except subprocess.TimeoutExpired as e:
            logs.append((e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""))
            logs.append("\n[TIMEOUT: remaining configs skipped]\n")
            break
        finally:
            pathlib.Path(f"/results/{name}.log").write_text("\n".join(logs))
            results_vol.commit()
    return name, "\n".join(logs)


@app.function(image=jax_image, gpu="L40S", timeout=45 * 60)
def jax_profile(args: list) -> tuple:
    p = subprocess.run(["python", "profile_jax.py"] + args, cwd="/root/src",
                       capture_output=True, text=True)
    return "jax_profile", p.stdout + ("\n[stderr]\n" + p.stderr[-4000:] if p.returncode else "")


@app.function(**common, timeout=3600)
def download(repo: str):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"])
    hf_cache.commit()


@app.function(**common, gpu="L40S", timeout=45 * 60)
def run_mistral(name: str, cmds: list) -> tuple:
    return _run_all(name, cmds, 43 * 60)


@app.function(**common, gpu="A100-80GB", timeout=30 * 60)
def run_qwen(name: str, cmds: list) -> tuple:
    return _run_all(name, cmds, 28 * 60)


RESULT = re.compile(r"^\s+(\S+)\s+(pyr\[[^\]]*\]|\S+)\s+ppl=([\d.]+)\s+ppl_post=([\d.]+)\s+mem=([\d.]+)\s+\(([\d.]+)s", re.M)


def summarize(logs: dict) -> str:
    rows = ["| job | method | ranks | PPL (all) | PPL (post-freeze) | mem | s/window |",
            "|---|---|---|---|---|---|---|"]
    for name in sorted(logs):
        hits = RESULT.findall(logs[name])
        if not hits and name != "latency":
            rows.append(f"| {name} | FAILED / no result | | | | | |")
        for meth, rk, ppl, post, mem, spw in hits:
            rows.append(f"| {name} | {meth} | {rk} | {ppl} | {post} | {mem} | {spw} |")
    return "\n".join(rows) + "\n"


@app.local_entrypoint()
def main(smoke: bool = False, windows: int = 20, needle: bool = False, hybrid: bool = False,
         retention: bool = False, diag: bool = False, select: bool = False, tkvr: bool = False,
         tkvr_dev: bool = False, jaxprof: bool = False, only: str = ""):
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if smoke else ""
    if jaxprof:
        base = ["--ctx", "4096", "32768", "131072", "--batch", "1", "8", "32",
                "--rank", "32", "--iters", "20"]
        logs = []
        for dt in ("float32", "bfloat16"):
            _, log = jax_profile.remote(base + ["--dtype", dt])
            logs.append(log)
            print(log, flush=True)
        (OUT / "jax_profile.log").write_text("\n".join(logs))
        return
    list(download.map(MODELS.values()))   # fill the shared cache once, before the fan-out
    if needle:
        todo = needle_jobs()
    elif hybrid:
        todo = hybrid_jobs(windows)
    elif retention:
        todo = retention_jobs(windows)
    elif diag:
        todo = diag_jobs(windows)
    elif select:
        todo = select_jobs()
    elif tkvr:
        todo = tkvr_jobs()
    elif tkvr_dev:
        todo = tkvr_dev_jobs()
    else:
        todo = jobs(2 if smoke else windows)
        todo[0][2].append(["latency"])   # GPU decode latency rides along in the first Mistral job
    if only:
        todo = [t for t in todo if t[0] == only]
    calls = [(run_mistral if m == "mistral7b" else run_qwen).spawn(n, c) for m, n, c in todo]
    print(f"launched {len(calls)} jobs")
    logs = {}
    for (_, name, _), call in zip(todo, calls):
        try:
            name, log = call.get()
        except Exception as e:   # keep collecting the other jobs
            log = f"[remote failure] {type(e).__name__}: {e}"
        logs[name] = log
        (OUT / f"{name}{tag}.log").write_text(log)
        print(f"done: {name}")
    if not (needle or hybrid or retention or diag or select or tkvr or tkvr_dev):
        (OUT / f"summary{tag}.md").write_text(summarize(logs))
        print(summarize(logs))
