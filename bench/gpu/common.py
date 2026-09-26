"""Shared helpers of the GPU evaluation program: resumable JSONL stores, backends, prompts, keys, provenance.

Every script appends one JSON line per finished item and skips items already present, so a spot interruption
costs only the item in flight. Test keys of the program are stored with the runs (they protect nothing).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")      # worker processes fork after tokenization
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "bench"))

import torch  # noqa: E402

from entropy_protocol import crypto  # noqa: E402
from entropy_protocol.backend_hf import HFBackend  # noqa: E402
from _meta import code_digest  # noqa: E402

DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
RUNS = Path(os.environ.get("EP_RUNS", ROOT / "runs"))
DATA = Path(os.environ.get("EP_DATA", ROOT / "data"))


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def slug(model: str) -> str:
    return model.replace("/", "_")


def load_backend(model: str, device: str = "cuda", dtype: str = "bfloat16", attn: str | None = None) -> HFBackend:
    local = os.environ.get("HF_HUB_OFFLINE") == "1"
    b = HFBackend(model, device=device, dtype=DTYPES[dtype], local_files_only=local, attn=attn or None)
    _cached_digest(b)
    return b


def _cached_digest(b: HFBackend) -> None:
    """Hashing 15-60 GB of weights takes a minute; keep the digest per (file, size, mtime) between scripts."""
    from entropy_protocol.backend_hf import weight_files
    try:
        from huggingface_hub import snapshot_download
        root = b.model_id if os.path.isdir(b.model_id) else snapshot_download(b.model_id, local_files_only=True)
    except Exception:  # noqa: BLE001 - the digest is then computed on first use as usual
        return
    sig = [[os.path.basename(p), os.path.getsize(p), int(os.path.getmtime(p))] for p in weight_files(root)]
    cache_path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "ep_digest_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    hit = cache.get(root)
    if hit and hit["files"] == sig:
        b.__dict__["weights_digest"] = hit["digest"]          # cached_property storage
        return
    cache[root] = {"files": sig, "digest": b.weights_digest}
    cache_path.write_text(json.dumps(cache, indent=1))


def add_backend_args(ap, dtype: str = "bfloat16") -> None:
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=dtype, choices=list(DTYPES))
    ap.add_argument("--attn", default="", help="attention kernel: sdpa | eager | flash_attention_2 (default: library)")


# ------------------------------------------------------------------------------------------- JSONL stores
def read_jsonl(path) -> list[dict]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with p.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:          # a torn last line after an interruption
                continue
    return out


class Store:
    """Append-only JSONL keyed by "id"; ``done`` holds the ids already written."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = {r["id"] for r in read_jsonl(self.path) if "id" in r}

    def write(self, row: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.done.add(row["id"])

    def rows(self) -> list[dict]:
        return read_jsonl(self.path)


def write_json(path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(p)


# ------------------------------------------------------------------------------------------------ keys
def program_keys(n: int = 8) -> list[bytes]:
    """The program's master keys: deterministic test keys (key i = SHA-256 of a fixed label and i), so every
    machine derives the same keys without copying secrets."""
    return [hashlib.sha256(f"ep-gpu-eval/2026-09/master/{i}".encode()).digest() for i in range(n)]


def null_keys(n: int) -> list[bytes]:
    """Detection keys never used to seal anything: the empirical null is scored under these."""
    return [crypto.wm_key(hashlib.sha256(f"ep-gpu-eval/2026-09/null/{i}".encode()).digest()) for i in range(n)]


# --------------------------------------------------------------------------------------------- prompts
def prompt_input(backend: HFBackend, item: dict, field: str, chat: bool):
    """A string (raw continuation, tokenized like the single path) or chat-template token ids."""
    text = item[field]
    if not chat:
        return text
    ids = backend.tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                          tokenize=True)
    return list(ids)


def provenance(backend: HFBackend, **extra) -> dict:
    return {"code": code_digest(), "model": backend.model_ref, "weights": backend.weights_digest,
            "env": backend.environment, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]
