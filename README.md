# AegisOne Guardian: leakage-resistant evaluation of a static malware classifier

**Dimitrios Sargiotis** · National Technical University of Athens · [users.ntua.gr/dims](https://users.ntua.gr/dims/) · [uomoathens.github.io](https://uomoathens.github.io/)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23565223.svg)](https://doi.org/10.5281/zenodo.23565223)

This repository contains the evaluation protocol, feature extractor and tooling behind the static file classifier of AegisOne, an endpoint protection prototype for Windows. It is published as a research artefact. The product itself is not included.

The central claim is methodological. Static malware classifiers are routinely reported with random train/test splits, accuracy or AUC, and point estimates of the false-positive rate. Each of these flatters the model. This repository implements the alternative end to end and makes every step checkable:

- **Temporal splits only.** Train, validation and test partitions are separated by observation time, never by random row, and equal timestamps never straddle a boundary.
- **No binary in two partitions.** Full-file SHA-256 identities are checked across partitions, and the run aborts on any overlap.
- **Thresholds chosen on validation, reported on a later test set.** The test partition is touched once.
- **Generalisation measured directly.** Detection rate on malware families absent from training, and false-positive rate on benign publishers absent from training, are reported separately.
- **False-positive rates reported as upper confidence bounds.** An observed zero on a small test set is not evidence of a low production rate. Every rate carries a one-sided 95% Clopper-Pearson bound.
- **Calibration reported.** Brier score, expected calibration error and a reliability table accompany every run.
- **One feature definition, two implementations, verified equal.** The extractor exists in Python (training) and C# (deployment). A parity check runs in CI on real and adversarially malformed files.

The reasoning behind each choice is in [METHODOLOGY.md](METHODOLOGY.md).

## Status

| Component | State |
|---|---|
| Feature schema v1 (48 features), Python and C# | Implemented; parity verified in CI |
| Release dataset contract v2 (provenance per sample) | Implemented; self-tested |
| Temporal evaluation with confidence bounds | Implemented; tested on synthetic drift data |
| Public benchmark: EMBER2024 under the temporal protocol | Run on the .NET subset; results below and in [results/ember2024-Dot_Net](results/ember2024-Dot_Net/report.md) |
| Results on the AegisOne schema v1 corpus | Not yet published |

## Findings: EMBER2024, .NET subset

**A threshold set on a random hold-out promised a 1% false-positive rate and delivered 4.1% on the following weeks; at a 0.1% target it delivered 1.55%, fifteen times the promise. Thresholds set on the eight weeks before the test period stayed within 0.11 percentage points of the target.**

Data: 520,000 labelled .NET files first seen between 24 September 2023 and 21 September 2024 (fit: the first 44 weeks, 440,000 files; validation: the last 8 weeks, 80,000 files), tested on 120,000 files from the 12 following weeks (60,000 benign, 60,000 malicious) and on the 829 .NET files of EMBER2024's evasive challenge set. EMBER feature version 3, LightGBM with the dataset's reference settings, seed 0. No SHA-256 identity crossed a partition boundary.

| Target FPR | Threshold chosen on | FPR where chosen | Test FPR (95% upper) | False alarms per 60,000 benign | Test TPR | Challenge TPR | Unseen-family TPR |
|---|---|---|---|---|---|---|---|
| 1% | Random 15% hold-out | 1.00% | **4.14%** (4.27%) | 2,482 | 99.2% | 92.8% | 97.9% |
| 1% | Last 8 training weeks | 1.00% | **1.11%** (1.18%) | 666 | 95.7% | 79.6% | 94.3% |
| 1% | Test set (oracle) | 1.00% | 1.00% (1.07%) | 600 | 95.3% | 78.5% | 93.5% |
| 0.1% | Random 15% hold-out | 0.10% | **1.55%** (1.63%) | 928 | 97.4% | 85.2% | 95.6% |
| 0.1% | Last 8 training weeks | 0.10% | **0.083%** (0.105%) | 50 | 82.8% | 62.2% | 76.3% |
| 0.1% | Test set (oracle) | 0.10% | 0.10% (0.12%) | 60 | 83.4% | 62.8% | 76.6% |

What the table shows:

1. **Random validation misstates the cost of deployment.** The random hold-out shares weeks with the training data, so it looks easier than the future. The threshold it produces is far lower (0.155 against 0.925 at the 1% target), and on later weeks that threshold raises four to fifteen times the promised number of false alarms.
2. **The higher detection rate of the random row is not a better model.** It is bought with those false alarms. At equal false-positive rates the two models are close: the temporal threshold detects 95.7% at 1.11% FPR, and the oracle 95.3% at 1.00%.
3. **Eight weeks of recent data are a usable proxy for the future.** The temporal threshold lands within 0.11 percentage points of its target at 1% and below its target at 0.1%, with detection within 0.7 percentage points of the oracle.
4. **Evasive and novel malware remain the hard cases.** At a 0.1% FPR the same threshold detects 82.8% of ordinary test malware, 76.3% of malware from families absent from training, and 62.2% of the challenge set.
5. **Drift is visible week by week.** At the 1% target the weekly test FPR ranges from 0.45% to 3.26%, with the peak in the week of 3 November 2024. A threshold that is correct on average still needs monitoring.

Limitations: one run, one seed and one file type, so the figures carry no estimate of run-to-run variation. The random hold-out model is trained on 85% of all 52 training weeks, including the eight used for temporal validation, so it differs from the temporal model in training data as well as in threshold. EMBER2024 is balanced between classes, unlike real endpoints, which is why false-positive rate rather than precision is reported. Labels come from VirusTotal detections and ClarAVy family assignment and may contain noise.

The synthetic results in [results/synthetic-demo](results/synthetic-demo/report.md) only demonstrate the pipeline and must not be cited as detection results.

## Public benchmark: EMBER2024

[EMBER2024](https://github.com/FutureComputing4AI/EMBER2024) (Joyce et al., KDD 2025) publishes features and labels for 3.2 million files first seen on VirusTotal between September 2023 and December 2024, organised by week: 52 training weeks, 12 later test weeks, and a challenge set of malware that roughly 70 antivirus engines initially missed.

`training/ember2024.py` applies this repository's protocol to it and answers one question: **when a threshold is set to deliver a 1% or 0.1% false-positive rate, what rate is actually delivered on later weeks?** It compares three ways of choosing the threshold, using EMBER's own LightGBM configuration and feature vectors:

1. a random 15% hold-out of the training weeks, the common practice;
2. the last 8 training weeks, strictly before the test period, this repository's protocol;
3. the test set itself, an oracle that shows the best attainable figure and is never a legitimate choice.

For each, it reports test FPR and TPR with 95% Clopper-Pearson bounds, detection on the challenge set, detection of malware families never seen in training, and FPR and TPR week by week across the test period.

The workflow `.github/workflows/ember2024.yml` runs this end to end on GitHub's servers: it downloads the .NET subset (about 2.3 GB), vectorises it, runs the benchmark and commits `report.md` and `report.json`. It starts automatically on the first push to `main` and can be re-run from the Actions tab. If Hugging Face requires authentication for the dataset, add a read token as the repository secret `HF_TOKEN`. The Win64 and Win32 subsets need more memory and disk than GitHub's free runner provides.

## Quick start

Requires Python 3.13 and, for the parity check, the .NET 8 SDK.

    python -m pip install -r requirements.txt
    python -m pytest -v
    python training/release_experiment.py --synthetic --out results/synthetic-demo

On a real dataset that follows contract v2:

    python training/release_validation.py --dataset dataset-v2.npz
    python training/release_experiment.py --dataset dataset-v2.npz --out results/my-run

EMBER2024 locally (needs `requirements-ember.txt`; the .NET subset runs on a standard 16 GB GitHub runner):

    python training/ember2024.py download --data ember-data --file-type Dot_Net
    python training/ember2024.py vectorize --data ember-data
    python training/ember2024.py run --data ember-data --file-type Dot_Net --out results/ember2024-Dot_Net

Add `--onnx guardian.onnx` to export the trained model (needs `requirements-export.txt`).

## Feature parity between Python and C#

    dotnet run --project tools/FeatureDump -c Release -- path/to/corpus csharp_vectors.csv
    python training/verify_parity.py --csharp csharp_vectors.csv

`tests/make_corpus.py` generates 27 edge cases: empty and one-byte files, window-boundary sizes, truncated and malformed PE headers, negative and out-of-range `e_lfanew`, section tables beyond end of file, more than 96 sections, timestamps at each side of the plausibility rule, and payloads appended to zeroed stubs. Before publication the check was also run on 301 real Windows PE files from a .NET 8 build output: 328 files, zero mismatches, largest absolute difference 5.6 × 10⁻⁸, which is float32 rounding.

## Layout

    training/
      aegis_features.py          feature schema v1, Python reference implementation
      dataset_contract.py        release dataset contract v2
      collect_benign.py          benign collection with publisher provenance (Windows)
      collect_malicious.py       malicious collection from MalwareBazaar; isolated VM only
      merge_release_datasets.py  SHA-256 deduplication, label-conflict detection
      release_validation.py      temporal split, leakage audit, thresholds, confidence bounds
      release_experiment.py      train, choose thresholds on validation, report on test
      ember2024.py               EMBER2024 benchmark under the temporal protocol
      train_guardian.py          LightGBM training and ONNX export
      verify_parity.py           C# against Python feature comparison
    src/AegisOne.Guardian.Features/
      PeFeatureVector.cs         feature schema v1, C# deployment implementation
    tools/FeatureDump/           writes C# vectors for the parity check
    tests/                       unit, protocol and parity tests
    results/synthetic-demo/      pipeline demonstration on synthetic data

## Handling malware safely

`collect_malicious.py` downloads live malware. Run it only inside a disposable virtual machine with no shared folders, no network drives and no personal accounts, and delete the machine afterwards. It asks for confirmation before it starts. Only the resulting `.npz` file leaves the VM: 48 numbers and metadata per sample, from which no executable can be reconstructed.

Read the licence of every data source before training. Several public malware corpora restrict commercial use.

## Limitations

Schema v1 uses static content and header features only. It can separate packed, tampered or oddly assembled binaries; it cannot see fileless attacks, living-off-the-land techniques or malicious scripts, none of which produce a suspicious PE file. Known weaknesses of individual features, including one found while preparing this release, are listed in [METHODOLOGY.md](METHODOLOGY.md#known-limitations-of-schema-v1).

## Citation

This repository is archived on Zenodo:

> Sargiotis, D. (2026). *AegisOne Guardian: leakage-resistant evaluation of a static malware classifier* (Version v0.1.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23565223

See also [CITATION.cff](CITATION.cff).

## Licence

MIT. See [LICENSE](LICENSE).
