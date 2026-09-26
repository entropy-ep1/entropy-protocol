"""Model identity: the weights digest covers exactly the files the loader reads."""
import json

import pytest

pytest.importorskip("torch")

from entropy_protocol.backend_hf import HFBackend, weight_files  # noqa: E402


def _digest(root) -> str:
    b = HFBackend.__new__(HFBackend)          # identity only, no model loaded
    b.model_id, b.local_files_only = str(root), True
    return b.weights_digest


def test_sharded_checkpoint_ignores_other_formats(tmp_path):
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        (tmp_path / name).write_bytes(name.encode() * 10)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors",
                        "c": "model-00001-of-00002.safetensors"}}))
    before = _digest(tmp_path)
    (tmp_path / "consolidated.safetensors").write_bytes(b"the same weights in another layout")
    assert _digest(tmp_path) == before                     # a second format next to the shards changes nothing
    assert [p.split("/")[-1] for p in weight_files(str(tmp_path))] == [
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]


def test_single_file_checkpoint(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"w")
    (tmp_path / "pytorch_model.bin").write_bytes(b"same weights, pickle")
    assert [p.split("/")[-1] for p in weight_files(str(tmp_path))] == ["model.safetensors"]
