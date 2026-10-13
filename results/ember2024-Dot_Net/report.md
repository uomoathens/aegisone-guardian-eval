# Results: EMBER2024 Dot_Net

Generated 2026-10-13T12:26:03Z, seed 0.

## Data

| Partition | Weeks first seen | Benign | Malicious |
|---|---|---|---|
| Fit | 2023-09-24 to 2024-07-27 | 220,000 | 220,000 |
| Validation | 2024-07-28 to 2024-09-21 | 40,000 | 40,000 |
| Test | 2024-09-22 to 2024-12-14 | 60,000 | 60,000 |
| Challenge (evasive) | | | 829 |

Malware families seen in fit: 1,603. Test malware from families never seen in fit: 1,264.
SHA-256 duplicates removed: {'fit_to_validation': 0, 'train_to_test': 0, 'train_to_challenge': 0}.

Test ROC AUC 0.9978; standardised partial AUC up to 1% FPR 0.9505; Brier 0.0171; ECE 0.0111.

## Promised versus delivered false-positive rate

Each row fixes a threshold for the target FPR on a different data set, then measures the later test weeks. Bounds are one-sided 95% Clopper-Pearson.

### Target FPR 1.0%

| Threshold chosen on | FPR where chosen | Test FPR | FPR upper 95% | Test TPR | TPR lower 95% | Challenge TPR | Unseen-family TPR |
|---|---|---|---|---|---|---|---|
| random 15% hold-out of training weeks | 0.998% | 4.137% | 4.273% | 99.18% | 99.12% | 92.76% | 97.94% |
| temporal: last 8 training weeks | 1.000% | 1.110% | 1.183% | 95.68% | 95.54% | 79.61% | 94.30% |
| oracle: threshold read off the test set | 1.000% | 1.000% | 1.069% | 95.25% | 95.11% | 78.53% | 93.51% |

Test weeks, temporal threshold:

| Week starting | Benign | FPR | Malicious | TPR |
|---|---|---|---|---|
| 2024-09-22 | 5,000 | 1.320% | 5,000 | 96.44% |
| 2024-09-29 | 5,090 | 0.904% | 5,000 | 95.96% |
| 2024-10-06 | 5,180 | 1.042% | 5,000 | 95.80% |
| 2024-10-13 | 4,922 | 0.488% | 5,000 | 95.44% |
| 2024-10-20 | 4,988 | 0.521% | 5,000 | 96.00% |
| 2024-10-27 | 5,062 | 1.580% | 5,000 | 95.88% |
| 2024-11-03 | 4,844 | 3.262% | 5,000 | 95.80% |
| 2024-11-10 | 5,136 | 1.285% | 5,000 | 95.40% |
| 2024-11-17 | 4,958 | 1.089% | 5,000 | 95.96% |
| 2024-11-24 | 5,032 | 0.914% | 5,000 | 95.36% |
| 2024-12-01 | 4,876 | 0.451% | 5,000 | 95.20% |
| 2024-12-08 | 4,912 | 0.489% | 5,000 | 94.96% |

### Target FPR 0.1%

| Threshold chosen on | FPR where chosen | Test FPR | FPR upper 95% | Test TPR | TPR lower 95% | Challenge TPR | Unseen-family TPR |
|---|---|---|---|---|---|---|---|
| random 15% hold-out of training weeks | 0.098% | 1.547% | 1.632% | 97.43% | 97.32% | 85.16% | 95.57% |
| temporal: last 8 training weeks | 0.100% | 0.083% | 0.105% | 82.76% | 82.50% | 62.24% | 76.27% |
| oracle: threshold read off the test set | 0.100% | 0.100% | 0.124% | 83.44% | 83.19% | 62.85% | 76.58% |

Test weeks, temporal threshold:

| Week starting | Benign | FPR | Malicious | TPR |
|---|---|---|---|---|
| 2024-09-22 | 5,000 | 0.080% | 5,000 | 81.60% |
| 2024-09-29 | 5,090 | 0.079% | 5,000 | 79.56% |
| 2024-10-06 | 5,180 | 0.077% | 5,000 | 83.40% |
| 2024-10-13 | 4,922 | 0.000% | 5,000 | 84.44% |
| 2024-10-20 | 4,988 | 0.040% | 5,000 | 82.56% |
| 2024-10-27 | 5,062 | 0.158% | 5,000 | 84.04% |
| 2024-11-03 | 4,844 | 0.289% | 5,000 | 84.36% |
| 2024-11-10 | 5,136 | 0.117% | 5,000 | 82.68% |
| 2024-11-17 | 4,958 | 0.040% | 5,000 | 84.12% |
| 2024-11-24 | 5,032 | 0.040% | 5,000 | 82.56% |
| 2024-12-01 | 4,876 | 0.000% | 5,000 | 83.20% |
| 2024-12-08 | 4,912 | 0.081% | 5,000 | 80.60% |

## Software

- python: 3.13.16
- thrember: 0.1.0 (0ef753e81d98)
- lightgbm: 4.7.0
- numpy: 2.5.3
- scikit-learn: 1.9.1
- signify: 0.7.1
- oscrypto: 1.3.0 (76bfdcb7285d)
