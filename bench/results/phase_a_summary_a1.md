# Phase A — Qwen/Qwen2.5-7B-Instruct (a1)

- code `c96620107c0cc3b2`, weights `fd024e4fed1dfb65`
- hardware: NVIDIA L40S; torch 2.7.0+cu128; attention sdpa

## Steps

| step | result | min:s |
|---|---|---|
| data | skipped (done earlier) | |
| unit_tests | skipped (done earlier) | |
| tiny_model | skipped (done earlier) | |
| model_tests | skipped (done earlier) | |
| cal_seal | skipped (done earlier) | |
| dump_float16_b1 | skipped (done earlier) | |
| dump_float16_b32 | skipped (done earlier) | |
| dump_bfloat16_b32 | skipped (done earlier) | |
| dump_float32_b1 | skipped (done earlier) | |
| table | ok | 0:04 |
| sealed | ok | 2:44 |
| quality | ok | 2:43 |
| sealed_bf | ok | 1:06 |
| plain | ok | 1:28 |
| kgw | skipped (done earlier) | |
| synthid | skipped (done earlier) | |
| chat_seal | ok | 1:29 |
| chat_plain | ok | 0:49 |
| gsm_seal | ok | 2:38 |
| gsm_plain | ok | 1:20 |
| gsm_qual | ok | 2:19 |
| verify_fp16 | ok | 1:39 |
| verify_bf16 | ok | 0:56 |
| verify_fp32 | ok | 1:19 |
| verify_bfg | ok | 0:34 |
| verify_q | ok | 1:27 |
| attacks | ok | 0:13 |
| detect | ok | 26:13 |
| detect_q | ok | 1:40 |
| base_kgw | ok | 1:31 |
| base_sid | FAILED rc=143 | 13:29 |
| q_div | ok | 0:59 |
| q_div_q | ok | 0:58 |
| q_gsm | ok | 0:04 |
| q_gsm_q | ok | 0:05 |
| speed | ok | 2:27 |
- 15:05:21 PHASE A DONE (GPU 44 GB, fp32 steps: run)

## Corpora

| corpus | texts | mean tokens | tok/s | batch | enforced/text | record points/text | texts with 0 points |
|---|---|---|---|---|---|---|---|
| A_c4_kgw | 64 | 400 | 555 | 32 | | | |
| A_c4_plain | 128 | 400 | 676 | 32 | | | |
| A_c4_quality | 128 | 400 | 337 | 32 | 59.2 | 2.4 | 18 |
| A_c4_quality_fp16cert | 128 | 400 | 337 | 32 | 59.2 | 35.0 | 0 |
| A_c4_sealed | 128 | 400 | 337 | 32 | 78.9 | 14.9 | 0 |
| A_c4_sealed_bf16gen | 32 | 400 | 248 | 32 | 78.7 | 12.4 | 0 |
| A_c4_sealed_fp16cert | 128 | 400 | 335 | 32 | 78.9 | 53.3 | 0 |
| A_c4_synthid | 64 | 400 | 258 | 16 | | | |
| A_cal | 64 | 393 | 297 | 32 | 80.0 | 80.0 | 0 |
| A_dolly_plain | 64 | 296 | 509 | 32 | | | |
| A_dolly_sealed | 64 | 321 | 264 | 32 | 64.3 | 18.7 | 0 |
| A_gsm8k_plain | 128 | 259 | 485 | 32 | | | |
| A_gsm8k_quality | 128 | 261 | 264 | 32 | 9.0 | 1.5 | 31 |
| A_gsm8k_sealed | 128 | 322 | 282 | 32 | 64.5 | 31.3 | 0 |

## Calibration (logit space, k = 1)

- reference `NVIDIA-L40S_float16_default_b1`, 64 texts, 24920 positions
- certified table: {'fp32': 2.044862, 'fp16': 0.421875}; record margin needed 4.0897; positions kept 18.9%

| configuration | max |Δ| (delta) | p99.99 | flips (all) | flips at kept positions |
|---|---|---|---|---|
| NVIDIA-L40S_bfloat16_default_b32 | 8.3438 | 2.2500 | 315 | 1 |
| NVIDIA-L40S_float16_default_b32 | 0.1406 | 0.1094 | 31 | 0 |
| NVIDIA-L40S_float32_default_b1 | 0.6816 | 0.5253 | 34 | 0 |

## Verification by record

- **verify_A_c4_quality_fp16_default_b32**: 128 texts, decisive 128, same → VERIFIED 100.0%, wrong MODIFIED 0, INDETERMINATE 0; transport wrong MODIFIED 0
  - edits: replace_word 100.0% (localized 100.0%, false VERIFIED 0); one_char 100.0% (localized 100.0%, false VERIFIED 0); delete_sentence 100.0% (localized 100.0%, false VERIFIED 0); insert_sentence 100.0% (localized 100.0%, false VERIFIED 0); truncate_tail 100.0% (localized 100.0%, false VERIFIED 0)
  - unrelated false VERIFIED 0, privacy leaks 0
- **verify_A_c4_quality_fp16cert_fp16_default_b32**: 128 texts, decisive 128, same → VERIFIED 100.0%, wrong MODIFIED 0, INDETERMINATE 0; transport wrong MODIFIED 0
  - edits: replace_word 100.0% (localized 100.0%, false VERIFIED 0); one_char 100.0% (localized 100.0%, false VERIFIED 0); delete_sentence 100.0% (localized 100.0%, false VERIFIED 0); insert_sentence 100.0% (localized 100.0%, false VERIFIED 0); truncate_tail 100.0% (localized 100.0%, false VERIFIED 0)
  - unrelated false VERIFIED 0, privacy leaks 0
- **verify_A_c4_sealed_bf16_default_b32**: 128 texts, decisive 128, same → VERIFIED 100.0%, wrong MODIFIED 0, INDETERMINATE 0; transport wrong MODIFIED 0
- **verify_A_c4_sealed_bf16gen_fp16_default_b32**: 32 texts, decisive 32, same → VERIFIED 100.0%, wrong MODIFIED 0, INDETERMINATE 0; transport wrong MODIFIED 0
  - edits: replace_word 100.0% (localized 100.0%, false VERIFIED 0); one_char 100.0% (localized 100.0%, false VERIFIED 0); delete_sentence 100.0% (localized 100.0%, false VERIFIED 0); insert_sentence 100.0% (localized 100.0%, false VERIFIED 0); truncate_tail 100.0% (localized 100.0%, false VERIFIED 0)
  - unrelated false VERIFIED 0, privacy leaks 0
- **verify_A_c4_sealed_fp16_default_b32**: 128 texts, decisive 128, same → VERIFIED 100.0%, wrong MODIFIED 0, INDETERMINATE 0; transport wrong MODIFIED 0
  - edits: replace_word 100.0% (localized 100.0%, false VERIFIED 0); one_char 100.0% (localized 100.0%, false VERIFIED 0); delete_sentence 99.219% (localized 99.213%, false VERIFIED 0); insert_sentence 100.0% (localized 100.0%, false VERIFIED 0); truncate_tail 100.0% (localized 99.219%, false VERIFIED 0)
  - unrelated false VERIFIED 0, privacy leaks 0
- **verify_A_c4_sealed_fp16cert_fp16_default_b32**: 128 texts, decisive 128, same → VERIFIED 100.0%, wrong MODIFIED 0, INDETERMINATE 0; transport wrong MODIFIED 0
  - edits: replace_word 100.0% (localized 100.0%, false VERIFIED 0); one_char 100.0% (localized 100.0%, false VERIFIED 0); delete_sentence 99.219% (localized 99.213%, false VERIFIED 0); insert_sentence 100.0% (localized 100.0%, false VERIFIED 0); truncate_tail 100.0% (localized 99.219%, false VERIFIED 0)
  - unrelated false VERIFIED 0, privacy leaks 0
- **verify_A_c4_sealed_fp32_default_b1**: 32 texts, decisive 32, same → VERIFIED 100.0%, wrong MODIFIED 0, INDETERMINATE 0; transport wrong MODIFIED 0

## Detection without a record

**detect_A** (tol 0.4219, 26.1 min)

| length | texts | at 1e-6 | at 1e-4 | at 1e-2 | checkpoints | match |
|---|---|---|---|---|---|---|
| 50 | 128 (128 full) | 0.0% | 1.6% | 62.5% | 7.3 | 100.0% |
| 100 | 128 (128 full) | 14.8% | 68.0% | 99.2% | 15.4 | 100.0% |
| 200 | 128 (128 full) | 96.1% | 100.0% | 100.0% | 31.3 | 100.0% |
| 400 | 128 (127 full) | 100.0% | 100.0% | 100.0% | 60.8 | 100.0% |

| null | pairs | false positives at 1e-4 | at 1e-2 | min p |
|---|---|---|---|---|
| same_model_other_key | 128 | 0 | 1 (0.8%) | 0.00674464686559593 |
| data:c4 | 20000 | 1 | 155 (0.8%) | 7.694451122217116e-05 |
| data:wiki_uk | 6000 | 0 | 39 (0.7%) | 0.000702247315556978 |
| runs:A_c4_plain | 2560 | 0 | 16 (0.6%) | 0.00022014250325698418 |

| attack | texts | at 1e-6 | at 1e-2 | segment at 1e-6 |
|---|---|---|---|---|
| chars_10 | 128 | 0.0% | 1.6% | 0.0% |
| chars_5 | 128 | 0.0% | 27.3% | 0.0% |
| collapse_whitespace | 128 | 98.4% | 100.0% | 96.9% |
| crlf | 128 | 100.0% | 100.0% | 98.4% |
| delete_sentence | 128 | 99.2% | 100.0% | 91.4% |
| embed_10 | 128 | 0.0% | 30.5% | 98.4% |
| embed_25 | 128 | 3.9% | 73.4% | 98.4% |
| embed_3spans_25 | 128 | 2.3% | 66.4% | 3.1% |
| homoglyphs_5 | 128 | 0.0% | 20.3% | 0.0% |
| long_doc | 128 | 0.0% | 2.3% | 60.2% |
| markdown_rendered | 128 | 99.2% | 100.0% | 97.7% |
| nfkc | 128 | 100.0% | 100.0% | 98.4% |
| smart_quotes | 128 | 100.0% | 100.0% | 98.4% |
| truncate_half | 128 | 96.9% | 100.0% | 51.6% |
| words_10 | 128 | 25.0% | 90.6% | 1.6% |
| words_25 | 128 | 0.0% | 21.9% | 0.0% |
| words_5 | 128 | 85.9% | 99.2% | 37.5% |
| words_50 | 128 | 0.0% | 0.8% | 0.0% |
| zero_width | 128 | 100.0% | 100.0% | 98.4% |

**detect_A_quality** (tol 0.4219, 1.6 min)

| length | texts | at 1e-6 | at 1e-4 | at 1e-2 | checkpoints | match |
|---|---|---|---|---|---|---|
| 50 | 128 (128 full) | 0.0% | 0.0% | 28.9% | 6.1 | 95.9% |
| 100 | 128 (128 full) | 0.8% | 25.0% | 82.0% | 12.1 | 96.5% |
| 200 | 128 (128 full) | 61.7% | 89.1% | 97.7% | 24.0 | 96.9% |
| 400 | 128 (127 full) | 95.3% | 96.1% | 98.4% | 43.7 | 97.2% |

| null | pairs | false positives at 1e-4 | at 1e-2 | min p |
|---|---|---|---|---|
| same_model_other_key | 128 | 0 | 1 (0.8%) | 0.00685909252933925 |
| data:c4 | 10000 | 0 | 64 (0.6%) | 0.0001776405593157393 |

## Baselines (same model, same prompts)

- kgw score L=100: TPR at empirical FPR 1e-2 {'tpr': 0.7344, 'threshold': 3.117691453623979, 'null_n': 1128, 'resolvable': True}, 1e-3 {'tpr': 0.0, 'threshold': 7.428242072256628, 'null_n': 1128, 'resolvable': True}; exact p ≤ 1e-4 0.625
- kgw score L=200: TPR at empirical FPR 1e-2 {'tpr': 0.9375, 'threshold': 3.117691453623979, 'null_n': 1128, 'resolvable': True}, 1e-3 {'tpr': 0.2188, 'threshold': 7.428242072256628, 'null_n': 1128, 'resolvable': True}; exact p ≤ 1e-4 0.6094
- kgw score L=400: TPR at empirical FPR 1e-2 {'tpr': 0.9688, 'threshold': 3.117691453623979, 'null_n': 1128, 'resolvable': True}, 1e-3 {'tpr': 0.6562, 'threshold': 7.428242072256628, 'null_n': 1128, 'resolvable': True}; exact p ≤ 1e-4 0.3125
- synthid mean L=100: TPR at empirical FPR 1e-2 {'tpr': 1.0, 'threshold': 0.5120822622107969, 'null_n': 1128, 'resolvable': True}, 1e-3 {'tpr': 0.9688, 'threshold': 0.5175531914893617, 'null_n': 1128, 'resolvable': True}; exact p ≤ 1e-4 0.8438
- synthid weighted L=100: TPR at empirical FPR 1e-2 {'tpr': 1.0, 'threshold': 0.5125747919082642, 'null_n': 1128, 'resolvable': True}, 1e-3 {'tpr': 1.0, 'threshold': 0.5188238024711609, 'null_n': 1128, 'resolvable': True}; exact p ≤ 1e-4 None
- synthid mean L=200: TPR at empirical FPR 1e-2 {'tpr': 1.0, 'threshold': 0.5120822622107969, 'null_n': 1128, 'resolvable': True}, 1e-3 {'tpr': 1.0, 'threshold': 0.5175531914893617, 'null_n': 1128, 'resolvable': True}; exact p ≤ 1e-4 0.9375
- synthid weighted L=200: TPR at empirical FPR 1e-2 {'tpr': 1.0, 'threshold': 0.5125747919082642, 'null_n': 1128, 'resolvable': True}, 1e-3 {'tpr': 1.0, 'threshold': 0.5188238024711609, 'null_n': 1128, 'resolvable': True}; exact p ≤ 1e-4 None
- synthid mean L=400: TPR at empirical FPR 1e-2 {'tpr': 1.0, 'threshold': 0.5120822622107969, 'null_n': 1128, 'resolvable': True}, 1e-3 {'tpr': 1.0, 'threshold': 0.5175531914893617, 'null_n': 1128, 'resolvable': True}; exact p ≤ 1e-4 1.0
- synthid weighted L=400: TPR at empirical FPR 1e-2 {'tpr': 1.0, 'threshold': 0.5125747919082642, 'null_n': 1128, 'resolvable': True}, 1e-3 {'tpr': 1.0, 'threshold': 0.5188238024711609, 'null_n': 1128, 'resolvable': True}; exact p ≤ 1e-4 None

## Quality

- quality_diversity_A_c4_quality_vs_A_c4_plain: {"A_c4_quality": {"distinct_1": 0.5066, "distinct_2": 0.8277, "distinct_3": 0.9132, "self_bleu4": 0.1188}, "A_c4_plain": {"distinct_1": 0.5091, "distinct_2": 0.8291, "distinct_3": 0.9165, "self_bleu4": 0.1175}}
- quality_diversity_A_c4_sealed_vs_A_c4_plain: {"A_c4_sealed": {"distinct_1": 0.5079, "distinct_2": 0.8354, "distinct_3": 0.9246, "self_bleu4": 0.1124}, "A_c4_plain": {"distinct_1": 0.5091, "distinct_2": 0.8291, "distinct_3": 0.9165, "self_bleu4": 0.1175}}
- quality_gsm8k_A_gsm8k_quality_vs_A_gsm8k_plain: {"items": 128, "accuracy_a": 0.8516, "accuracy_b": 0.875, "only_a_correct": 5, "only_b_correct": 8}
- quality_gsm8k_A_gsm8k_sealed_vs_A_gsm8k_plain: {"items": 128, "accuracy_a": 0.3906, "accuracy_b": 0.875, "only_a_correct": 4, "only_b_correct": 66}

## Speed

- batch 1: plain 40.2 tok/s, sealed 32.6 tok/s (overhead 23.3%), peak 14.670386791229248 GB
- batch 8: plain 264.6 tok/s, sealed 217.2 tok/s (overhead 21.8%), peak 14.682396411895752 GB
- batch 32: plain 840.9 tok/s, sealed 477.0 tok/s (overhead 76.3%), peak 15.409183025360107 GB
- verify batch 1: 9275.2 tok/s, 0.1078 s/text
- verify batch 32: 8958.3 tok/s, 0.1116 s/text
