# Check run of the campaign script, 27.10.2026 (code 8240113)

`GCP_PROJECT=… fuzz/campaign_gcp.sh 3 e2-highcpu-8`, after the launch, merge and runner fixes: every target ran
(20,469,742 executions in 3 minutes, `summary.json`), the launch returned at once, and the merge produced
1506 corpus files with no `MERGE FAILED` (1887 raw new inputs under 64 KB). One finding: a RIM certificate with X.509
version 10 (`x509.InvalidVersion`), fixed and kept in `fuzz/corpus/rim/`. The archive (`results.tgz`, 11 MB,
SHA-256 b515dc9a46738bff2c3ed5900f53e465e59363a9161fd65686e41d971c092709) stays with the operator: a three-minute corpus is not worth the repository's weight.
