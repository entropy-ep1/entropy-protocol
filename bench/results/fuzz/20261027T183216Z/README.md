# Coverage-guided campaign, 27.10.2026 (code 4add982)

`fuzz/campaign_gcp.sh 60 e2-highcpu-32`: every target of `fuzz/targets.py` for 60 minutes in parallel with atheris,
`-timeout=10 -rss_limit_mb=2048`. `summary.json`: executions, coverage and findings per target (979,964,023
executions in all); `results.tgz` (SHA-256 9da069e4eced70b3fbc2e452f7f38243d51b5fdcae369f61b617f30c9a232703):
libFuzzer logs, merge logs and every finding.

Findings, each fixed and kept in `fuzz/corpus/<target>/finding-*`: `event_log` (a bank an event lacked),
`hcl` (a JWK member that was not base64url), `receipt` (an inclusion proof of the wrong shape), `rim` (a schema
error inside signxml), `rim_payload` and `tpm_quote` (the inputs CI's Hypothesis run had also found), and
`timestamp` — a 1,995-octet input that took 13 s in the DER walk. `timestamp_reply` did not start (no seeds).

The merge step failed — libFuzzer re-executes `run.py`, which was not executable — so this campaign returned no
grown corpus; `fuzz/run.py` and the script are fixed for the next one.
