# Results: synthetic drift data (pipeline demonstration, not a detection result)

Generated 2026-10-13T08:39:48Z, feature schema v1, dataset contract v2, seed 42.

Temporal split: train 5,040 (until 2026-03-25T00:00:00), validation 1,080 (until 2026-04-12T00:00:00), test 1,080. SHA-256 overlap between partitions: 0.

Test AUC 0.6869, Brier score 0.1378, ECE (10 bins) 0.1226.

Thresholds are fixed on the validation partition and applied unchanged to the later test partition. Bounds are one-sided 95% Clopper-Pearson.

| Target TPR | Threshold | Test TPR | TPR lower 95% | Test FPR | FPR upper 95% | FP per 10k benign | Unseen-family TPR (n) | Unseen-publisher FPR (n) |
|---|---|---|---|---|---|---|---|---|
| 90% | 0.0016 | 78.40% | 72.40% | 53.4858% | 56.2370% | 5348.6 | 78.40% (162) | 53.4858% (918) |
| 95% | 0.0002 | 93.83% | 89.75% | 77.6688% | 79.9161% | 7766.9 | 93.83% (162) | 77.6688% (918) |
| 99% | 0.0000 | 99.38% | 97.11% | 90.0871% | 91.6667% | 9008.7 | 99.38% (162) | 90.0871% (918) |

Most used features:

- `mid_entropy` (1167)
- `header_entropy` (1131)
- `max_entropy` (1102)
- `overlay_ratio` (1084)
- `tail_entropy` (1066)
- `hist_24` (945)
- `hist_20` (931)
- `hist_31` (915)
- `hist_02` (915)
- `hist_21` (912)
