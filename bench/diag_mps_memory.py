"""Where does the batched sealing path's memory go on MPS? (13.09.2026: 1.5B GSM8K at batch 4 peaked at 16 GB; after
the allocator-cache fix, Phi-3 sealing still grew from batch to batch and reached 16 GB at its tenth batch.)

Seals questions exactly as bench/gpu/seal_corpus.py does (chat template, guard 0.9 by default) and prints, every N
model calls and after each batch: live MPS tensors (current_allocated), what the MPS driver holds for the process
(driver_allocated: live tensors + the allocator's cache), the process footprint (top's MEM: everything, including
graph caches and CPU memory), and how many distinct input shapes the model has seen.

Usage: PYTHONPATH=src python3 bench/diag_mps_memory.py --model Qwen/Qwen2.5-1.5B-Instruct --batch 4 [--batches 3]
       [--gen-only] [--plain] [--data prose_offline --field prompt --temperature 0.7 --min-new 250]
"""
import argparse
import hashlib
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gpu"))


def footprint_mb() -> int:
    """phys_footprint of this process from the kernel (task_info TASK_VM_INFO), in MB: what top's MEM shows, to the
    byte (top prints whole gigabytes above 10 GB). Falls back to top when the call is unavailable."""
    try:
        import ctypes
        libc = ctypes.CDLL(None)
        task = ctypes.c_uint.in_dll(libc, "mach_task_self_")
        buf, count = (ctypes.c_uint32 * 128)(), ctypes.c_uint32(128)
        if libc.task_info(task, 22, ctypes.byref(buf), ctypes.byref(count)) == 0 and count.value >= 38:
            return int(ctypes.c_uint64.from_buffer(buf, 144).value / 2 ** 20)     # task_vm_info.phys_footprint
    except Exception:  # noqa: BLE001
        pass
    out = subprocess.run(["top", "-l", "1", "-pid", str(os.getpid()), "-stats", "mem"], capture_output=True, text=True)
    s = out.stdout.strip().splitlines()[-1].strip().rstrip("+-")
    n = float("".join(c for c in s if c.isdigit() or c == "."))
    return int(n * 1024 if "G" in s else n / 1024 if "K" in s else n if "M" in s else n / 2 ** 20)


def mem(tag: str) -> None:
    gb = 2 ** 30
    print(f"[{time.strftime('%H:%M:%S')}] {tag:40s} live {torch.mps.current_allocated_memory() / gb:6.2f} GB | "
          f"driver {torch.mps.driver_allocated_memory() / gb:6.2f} GB | footprint {footprint_mb() / 1024:6.2f} GB",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--data", default="gsm8k")
    ap.add_argument("--field", default="question")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--batches", type=int, default=1, help="consecutive batches (memory printed after each)")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--min-new", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--empty-cache", action="store_true", help="torch.mps.empty_cache() every --every model calls")
    ap.add_argument("--plain", action="store_true", help="generate without sealing (no canonical generation)")
    ap.add_argument("--gen-only", action="store_true", help="the sealed generation alone: no measurement pass")
    ap.add_argument("--repeat", action="store_true", help="the same first batch every time (the same shapes)")
    ap.add_argument("--no-tidy", action="store_true", help="keep the MPS cache (HFBackend.MPS_SLACK = infinity)")
    ap.add_argument("--slack-gb", type=float, default=None, help="override HFBackend.MPS_SLACK")
    a = ap.parse_args()
    from common import DATA, load_backend, prompt_input, read_jsonl
    from entropy_protocol.batch import generate_batch, seal_batch
    from entropy_protocol.seal import SealParams, decider, margins
    mem("start")
    b = load_backend(a.model, "mps", a.dtype)
    if a.no_tidy:
        b.MPS_SLACK = float("inf")
    if a.slack_gb is not None:
        b.MPS_SLACK = int(a.slack_gb * 2 ** 30)
    mem(f"model loaded (slack {getattr(b, 'MPS_SLACK', 'n/a')})")
    calls, shapes = [0], set()

    def hook(module, args, kwargs, output):
        calls[0] += 1
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is not None:
            shapes.add(tuple(ids.shape))
        if calls[0] % a.every == 0:
            mem(f"model call {calls[0]} {tuple(ids.shape) if ids is not None else ''}")
            if a.empty_cache:
                torch.mps.empty_cache()
    b.model.register_forward_hook(hook, with_kwargs=True)
    items = read_jsonl(DATA / f"{a.data}.jsonl")[:a.batch * a.batches]
    params = SealParams(max_new_tokens=a.max_new, min_new_tokens=a.min_new, temperature=a.temperature, guard=0.9)
    master, digests, t_all = bytes([7]) * 32, [], time.time()
    for bi in range(a.batches):
        group = items[:a.batch] if a.repeat else items[bi * a.batch:(bi + 1) * a.batch]
        inputs = [prompt_input(b, it, a.field, True) for it in group]
        docs = [f"diag-{bi}-{i}" for i in range(len(inputs))]
        t0 = time.time()
        if a.plain:
            outs = generate_batch(b, inputs, [lambda j, g: None] * len(inputs), a.max_new, a.temperature, 0.9, 1, 3,
                                  seed=1, min_new_tokens=a.min_new)
            n, texts = sum(len(ids) for ids, _ in outs), [b.tok.decode(ids) for ids, _ in outs]
        elif a.gen_only:
            _, _, hm = margins(params)
            outs = generate_batch(b, inputs, [decider(params, master, d) for d in docs], params.max_new_tokens,
                                  params.temperature, params.top_p, params.k, params.high_window, 1,
                                  params.min_new_tokens, hm, params.align, params.free_after, gate=params.gate,
                                  guard=params.guard, canon_gen=params.canon_gen)
            n, texts = sum(len(ids) for ids, _ in outs), [str(ids) for ids, _ in outs]
        else:
            res = seal_batch(b, inputs, [master] * len(inputs), params, doc_ids=docs, seed=1)
            n, texts = sum(s.tokens for s in res), [f"{s.text}|{s.enforced}|{s.entries}" for s in res]
        digests.append(hashlib.sha256("\x00".join(texts).encode()).hexdigest()[:16])
        mem(f"batch {bi + 1}: {n} tok, {time.time() - t0:.0f}s, {len(shapes)} shapes, {digests[-1]}")
    mem(f"done: {calls[0]} calls, {len(shapes)} distinct input shapes, {time.time() - t_all:.0f}s")
    torch.mps.empty_cache()
    mem("after empty_cache")


if __name__ == "__main__":
    main()
