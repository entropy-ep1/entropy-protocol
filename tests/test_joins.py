"""The join check never rejects a canonical tokenization: every join of a real text's tokens is one the verifier
reads back as it is (a false rejection would steer generation away from text the verifier reads correctly)."""
import pytest

pytest.importorskip("torch")

from entropy_protocol.canon import canonicalize  # noqa: E402
from entropy_protocol.canongen import CanonGuard  # noqa: E402

TEXTS = [
    'The city council met on Tuesday to discuss the new budget. "We can\'t wait," said the mayor -- and the vote '
    "passed.\n\nNext steps:\n- publish the minutes\n- 2 more meetings",
    "Совет собрался во вторник. \"Мы не можем ждать\", - сказал мэр. П'ять хвилин... і все.\n\nДалее: итоги.",
    "北京是中国的首都。它有很多历史名胜。你去过吗？我们明天见！",
    "def parse(line):\n    key, _, value = line.partition('=')\n    return key.strip(), value.strip()\n\n"
    'print(parse("a = 1"))',
    "| name | value |\n|---|---|\n| a | 1 |\n\n**Bold** and *italic*, `code` and [a link](http://example.com).\n\n"
    "1. first\n2. second",
]
MODELS = ["distilgpt2", "Qwen/Qwen2.5-0.5B-Instruct", "microsoft/Phi-3-mini-4k-instruct",
          "mistralai/Mistral-7B-Instruct-v0.2"]


@pytest.mark.parametrize("name", MODELS)
def test_canonical_tokenizations_always_join(name):
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{name} tokenizer not available offline: {exc}")
    g = CanonGuard(tok, len(tok))
    for text in TEXTS:
        ids = tok(canonicalize(text), add_special_tokens=False)["input_ids"]
        state = 0
        for i, t in enumerate(ids):
            assert not bool(g.mask(state)[t]), (name, i)          # nothing a canonical text needs is banned
            state = g.state_after(state, t)
            if i:
                assert g.joinable(ids[:i], t), (name, tok.convert_ids_to_tokens(ids[max(0, i - 3):i + 1]))
