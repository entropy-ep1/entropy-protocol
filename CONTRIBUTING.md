# Contributing

Issues and pull requests are welcome. Before a change is merged:

- `python -m pytest tests -q` passes (`-m "not model"` needs no model; the model tests use distilgpt2 on the CPU);
- if the change touches a format, `spec/make_vectors.py` regenerates `spec/test-vectors/ep1-vectors.json`, the new
  vectors are committed, and `node impl-js/check-vectors.mjs` passes — both implementations must agree;
- a claim in the documentation comes with the file and the command that reproduce it.

By contributing you agree that your contribution is licensed under the Apache License 2.0.
