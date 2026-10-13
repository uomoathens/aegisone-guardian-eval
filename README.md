# AegisOne Guardian: leakage-resistant evaluation of a static malware classifier

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
| Public benchmark: EMBER2024 under the temporal protocol | Implemented and integration-tested; runs automatically in GitHub Actions and commits its results to `results/ember2024-Dot_Net/` |
| Results on the AegisOne schema v1 corpus | Not yet published |

Until the EMBER2024 run has completed, the only results in this repository are from synthetic data with built-in temporal drift ([results/synthetic-demo](results/synthetic-demo/report.md)). They demonstrate the pipeline and the effect of drift. They are not detection results and must not be cited as such.

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

EMBER2024 locally (needs `requirements-ember.txt`, about 7 GB of disk and 12 GB of memory for the .NET subset):

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

See [CITATION.cff](CITATION.cff).

## Licence

MIT. See [LICENSE](LICENSE).
