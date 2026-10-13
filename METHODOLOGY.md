# Methodology

This document records why the evaluation is built the way it is. The code enforces each rule; the reasons live here.

## 1. Why not accuracy, and not really AUC

On a real endpoint almost every file is benign. A classifier that labels everything clean reaches very high accuracy and a respectable AUC on any realistic class mix, while detecting nothing. The quantity that decides whether a static classifier is usable is the false-positive rate at a fixed detection rate. One false alarm per thousand clean files still means dozens of alarms on an ordinary Windows installation, and a security tool that raises them is switched off before it catches anything.

Every run therefore reports, for target detection rates of 90%, 95% and 99%:

- the threshold that reaches the target on the validation partition,
- the detection rate (TPR) and false-positive rate (FPR) that threshold produces on the later test partition,
- false positives per 10,000 benign files,
- confidence bounds on both rates (section 5).

AUC is still printed, as a secondary diagnostic.

## 2. Temporal splits

Malware changes faster than benign software. A random split lets the model see the future: samples from the same campaign, built days apart, land on both sides, and the test score measures recall of known campaigns rather than detection of new ones.

`release_validation.temporal_split` orders samples by `observed_at_utc` and cuts at the 70% and 85% positions by default. Samples sharing a timestamp are kept in the same partition, and the function verifies strict chronology:

$$\max t_{\text{train}} < \min t_{\text{val}}, \qquad \max t_{\text{val}} < \min t_{\text{test}}$$

```latex
\max t_{\text{train}} < \min t_{\text{val}}, \qquad \max t_{\text{val}} < \min t_{\text{test}}
```

Collection must therefore happen on several dates. A single collection run cannot be split temporally, and the code refuses it.

## 3. Identity leakage

The same binary is often collected twice, from two feeds or on two days. If copies land in training and test, the score is inflated. Contract v2 stores the full-file SHA-256 of every sample. `merge_release_datasets.py` deduplicates on that hash, keeps the earliest observation, and aborts if one hash carries both labels. `assert_no_sha_leakage` verifies before every run that no hash appears in two partitions.

Deduplication is by file hash, never by feature vector. Two different binaries can share a vector, and treating them as one would silently discard data.

## 4. Thresholds and unseen groups

The threshold is chosen once, on validation, as the score of the $\lceil \tau \cdot n_{\text{mal}} \rceil$-th highest-scoring malicious validation sample for target rate $\tau$, and is applied unchanged to the test partition.

```latex
\theta_\tau = s_{(\lceil \tau \, n_{\text{mal}} \rceil)}
```

Two subsets of the test partition are reported on their own:

- **Unseen families:** malicious test samples whose family label never appears among malicious training samples. This is the closest offline proxy for a new campaign.
- **Unseen publishers:** benign test samples whose publisher never appears among benign training samples. This exposes a model that has learned "Microsoft-looking" rather than "benign".

Unknown or blank labels are excluded from both subsets rather than counted as new.

## 5. Confidence bounds

With $k$ false positives among $n$ benign test files, the reported upper bound on the FPR is the one-sided 95% Clopper-Pearson bound:

$$p_U = B^{-1}\!\left(0.95;\; k+1,\; n-k\right)$$

```latex
p_U = B^{-1}\!\left(0.95;\; k+1,\; n-k\right)
```

where $B^{-1}$ is the inverse regularised incomplete beta function. For $k = 0$ it reduces to

$$p_U = 1 - 0.05^{1/n} \approx \frac{3}{n}$$

```latex
p_U = 1 - 0.05^{1/n} \approx \frac{3}{n}
```

So zero false positives among 3,000 benign files supports a claim of at most 0.1%, not 0%. To support one false alarm per 10,000 files at 95% confidence, roughly 30,000 benign test files with no false positives are needed. The detection rate carries the matching lower bound, $B^{-1}(0.05;\, k,\, n-k+1)$. The zero-failure case is checked against the closed form, and both bounds are checked to bracket the point estimate, in `tests/test_release_validation.py`.

## 6. Calibration

Scores are used as risk estimates downstream, so their calibration is reported. On the test partition, with scores grouped into 10 equal-width bins $B_b$:

$$\mathrm{ECE} = \sum_{b=1}^{10} \frac{|B_b|}{n}\,\bigl|\bar{s}_b - \bar{y}_b\bigr|$$

```latex
\mathrm{ECE} = \sum_{b=1}^{10} \frac{|B_b|}{n}\,\bigl|\bar{s}_b - \bar{y}_b\bigr|
```

together with the Brier score and the full reliability table in `report.json`.

## 7. Class balance

Training at 50/50 teaches the model that malware is as common as clean software, and the deployed false-positive rate is then far worse than validation suggested. The benign class should be at least ten times the malicious one. LightGBM's `scale_pos_weight` is capped at 3 so that rebalancing stays mild.

## 8. Feature schema v1 and the parity contract

Each file is read in up to three windows of 256 KiB (head, middle, tail). The middle window is read only when the file exceeds three windows. The tail window is distinct as soon as the file exceeds one window. The tail window is what reveals a high-entropy payload appended to a low-entropy stub.

| Index | Feature | Definition |
|---|---|---|
| 0 | `log_size` | $\log_{10}(\text{size}+1)/10$ |
| 1–3 | `header_entropy`, `mid_entropy`, `tail_entropy` | Shannon entropy of each window, divided by 8 |
| 4 | `max_entropy` | maximum of the three, divided by 8 |
| 5 | `entropy_gap` | $(H_{\text{tail}} - H_{\text{head}} + 8)/16$ |
| 6–37 | `hist_00` … `hist_31` | byte histogram in 32 buckets of 8 values, over the distinct windows |
| 38 | `printable_ratio` | share of bytes 0x20–0x7E plus tab, LF, CR |
| 39 | `null_ratio` | share of zero bytes |
| 40 | `is_pe` | valid MZ and PE signatures within the file |
| 41 | `section_count` | section count divided by 16 |
| 42 | `is_dll` | COFF characteristic 0x2000 |
| 43 | `timestamp_bad` | link timestamp zero, before 1995, or two or more days in the future |
| 44 | `is_64bit` | machine AMD64 or ARM64 |
| 45 | `is_gui_subsystem` | optional-header subsystem equals 2 |
| 46 | `relocs_stripped` | COFF characteristic 0x0001 |
| 47 | `overlay_ratio` | bytes after the end of the last section, as a share of the file |

Every value is clamped to $[0, 1]$, and NaN becomes 0.

The model is trained on vectors from `aegis_features.py` and queried with vectors from `PeFeatureVector.cs`. If the two drift apart, nothing fails: detection quality degrades silently while every check stays green. The rules are therefore:

- any change to the feature set is a new schema version, added alongside the old one; an existing index is never reordered or repurposed;
- `train_guardian.py` loads every exported ONNX model back and warns if its input width differs from the schema;
- `verify_parity.py` compares both implementations file by file at tolerance $10^{-6}$ and runs in CI on Linux and Windows.

## 9. Known limitations of schema v1

**`timestamp_bad` is dominated by deterministic builds.** In the 301 real PE files used for the parity check (framework and NuGet libraries from a .NET 8 build output), 249 carried a link timestamp later than the current year, and the feature fired on every one of the 204 libraries named `Microsoft.*` or `System.*`. Deterministic compilation writes a content hash into the COFF timestamp field rather than a date, so the field no longer measures what the feature name says. In a training set this feature will largely encode "built deterministically", which correlates with toolchain and publisher, not with intent. Schema v2 should replace it, for example with a flag for hash-like timestamps plus a separate plausibility check applied only to non-deterministic builds.

**`timestamp_bad` also depends on the evaluation date.** The "future" rule compares against the clock at extraction time, so the same file can change value as time passes. Vectors should be extracted once, at collection time, and stored; re-extraction months later is not guaranteed to reproduce them.

**Overlapping windows for mid-sized files.** For files between one and two windows long, the head and tail windows overlap and the shared bytes are counted twice in the histogram. This is deterministic and identical in both implementations, but it means the histogram is not a pure byte distribution for that size range.

**Scope.** These are static features. Fileless attacks, abuse of legitimate system tools and malicious scripts produce no suspicious PE file and are outside what this classifier can see. They need behavioural telemetry.

## 10. EMBER2024 benchmark design

EMBER2024 is organised by the week in which each file was first submitted to VirusTotal, which makes a strictly temporal protocol possible on public data. `training/ember2024.py` uses the 2,568-dimensional EMBER feature version 3 vectors produced by the dataset's own `thrember` package, and LightGBM settings taken from the dataset's reference configuration, so that differences in the results come from the protocol rather than from the model. One setting differs: LightGBM builds its feature bins from a sample of 50,000 rows instead of its default 200,000. With 2,568 features the default sample alone needs several gigabytes and exhausts the memory of a standard GitHub runner; on data of the .NET subset's size, peak memory falls from an estimated 10 GB (extrapolated from measurements at 30,000 and 60,000 rows) to a measured 5.8 GB for the whole benchmark. The same setting is used under all three threshold protocols, so the comparison between them is unaffected.

The two practices it is compared with appear in the dataset's own reference scripts, which is why they are worth measuring rather than assuming. The reference training function holds out a stratified random 10% of training rows, so validation rows come from the same weeks as training rows. The reference evaluation script reads its operating point, the detection rate at 1% FPR, off the test set.

Partitions:

- **Fit:** training weeks before the last eight.
- **Validation:** the last eight training weeks, used only to choose thresholds.
- **Test:** EMBER's twelve test weeks.
- **Challenge:** EMBER's evasive malware, filtered to the same file type, used only to measure detection.

Rows without a benign/malicious label and rows of other file types are excluded and counted. SHA-256 identities are compared across partitions, and any duplicate in a later partition is removed and counted.

For a target false-positive rate $\alpha$ and benign validation scores sorted in decreasing order $b_{(1)} \ge b_{(2)} \ge \dots \ge b_{(n)}$, the threshold is the smallest value above $b_{(k+1)}$ with $k = \lfloor \alpha n \rfloor$:

$$\theta_\alpha = \operatorname{nextafter}\!\left(b_{(k+1)},\, +\infty\right)$$

```latex
\theta_\alpha = \operatorname{nextafter}\!\left(b_{(k+1)},\, +\infty\right)
```

This guarantees an FPR of at most $\alpha$ on the data where it was chosen, including when scores are tied. The same rule is applied to three score sets: a random 15% hold-out of the training weeks (with its own model), the temporal validation weeks, and the test set (the oracle). The headline comparison is between the FPR promised where the threshold was chosen and the FPR delivered on the later test weeks, reported overall and week by week.

Because EMBER2024 does not record the publisher of benign files, the unseen-publisher analysis of section 4 is not available for it. Families come from the dataset's ClarAVy labels, and unseen-family detection is reported for test malware whose family never occurs in the fit partition.

## 11. What a result must include

A run is reportable only if it states: the collection dates and sources, the number of benign publishers and malware families, the split cut-offs, all three operating points with bounds, the unseen-family and unseen-publisher figures, calibration, and the feature importances. If the most important features are all histogram buckets, the model has probably learned file format rather than intent. An AUC of 1.0 on real data indicates leakage, not success.
