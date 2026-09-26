"""The tolerance is measured where checkpoints can be: the first tokens after BOS carry no record entry and no
detector count (15.09.2026, Qwen2.5-7B on an L40S: one position-3 outlier made the fp16 table 14 times wider)."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from entropy_protocol.calibrate import calibrate  # noqa: E402
from entropy_protocol.seal import SealParams, first_checkpoint, table_covers  # noqa: E402

V = 11


class Scorer:
    """A deterministic stand-in for a backend: logits depend on the position only; ``bump`` raises the logit of the
    token present at chosen positions, which moves its margin there and nowhere else."""
    start_id = 0
    device = "cpu"

    def __init__(self, dtype, bump: dict[int, float] | None = None):
        self.dtype = dtype
        self.bump = bump or {}

    def model(self, inp):
        ids = inp[0].tolist()
        logits = torch.stack([torch.linspace(0.0, 1.0 + t, V) for t in range(len(ids))])[None]
        for pos, v in self.bump.items():
            logits[0, pos, ids[pos + 1]] += v          # row pos predicts ids[pos + 1] (BOS first)
        return SimpleNamespace(logits=logits)


def test_positions_before_the_first_checkpoint_are_not_measured():
    corpus = [[3, 5, 7, 9, 2, 4, 6, 8], [1, 2, 3, 4, 5, 6]]
    ref = Scorer(torch.float32)
    configs = {"cpu:float16": Scorer(torch.float16, {2: 5.0}),     # an outlier at position 2 only
               "cpu:bfloat16": Scorer(torch.bfloat16, {2: 5.0, 5: 0.25})}
    every = calibrate(ref, configs, corpus, k=1, from_position=0)
    used = calibrate(ref, configs, corpus, k=1, from_position=first_checkpoint(SealParams()))
    assert every.delta["cpu:float16"] == pytest.approx(5.0) and every.delta["cpu:bfloat16"] == pytest.approx(5.0)
    assert used.delta["cpu:float16"] == 0.0 and used.delta["cpu:bfloat16"] == pytest.approx(0.25)
    assert used.from_position == 4 and used.positions == (8 - 4) + (6 - 4)
    assert used.table(["fp16", "bf16"]) == {"fp16": 0.0, "bf16": 0.75}


def test_a_table_must_cover_every_position_the_parameters_use():
    content, position = SealParams(), SealParams(selection="position")
    assert first_checkpoint(content) == content.context and first_checkpoint(position) == position.interval
    assert table_covers({}, content) is None                            # measured everywhere (older files)
    assert table_covers({"from_position": 4}, content) is None
    assert "from 5 on" in table_covers({"from_position": 5}, content)
    assert table_covers({"from_position": 5}, position) is None
