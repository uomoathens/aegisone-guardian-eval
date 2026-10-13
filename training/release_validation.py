from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

_mlcore = os.environ.get("AEGIS_MLCORE_SITE")
if _mlcore:
    sys.path.insert(0, _mlcore)

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aegis_features as F

DATASET_CONTRACT_VERSION = 2
UNKNOWN = {"", "unknown", "none", "null", "n/a", "na", "-"}


@dataclass(frozen=True)
class Dataset:
    X: np.ndarray
    y: np.ndarray
    sample_sha256: np.ndarray
    observed_at_utc: np.ndarray
    source: np.ndarray
    family: np.ndarray
    publisher: np.ndarray


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_cutoff: np.datetime64
    validation_cutoff: np.datetime64


def _as_text(values) -> np.ndarray:
    return np.asarray(values).astype(str)


def _known(values: np.ndarray) -> np.ndarray:
    normalised = np.char.lower(np.char.strip(values.astype(str)))
    mask = np.ones(normalised.shape[0], dtype=bool)
    for token in UNKNOWN:
        mask &= normalised != token
    return mask


def load_dataset(path: str) -> Dataset:
    data = np.load(path, allow_pickle=False)

    required = {
        "X", "y", "schema_version", "dataset_contract_version",
        "sample_sha256", "observed_at_utc", "source", "family", "publisher",
    }
    missing = sorted(required.difference(data.files))
    if missing:
        raise ValueError(
            "Dataset is not release-contract v2; missing keys: "
            + ", ".join(missing)
        )

    schema = int(np.asarray(data["schema_version"]).reshape(-1)[0])
    if schema != F.VERSION:
        raise ValueError(
            f"Feature schema mismatch: dataset v{schema}, code v{F.VERSION}"
        )

    contract = int(np.asarray(data["dataset_contract_version"]).reshape(-1)[0])
    if contract != DATASET_CONTRACT_VERSION:
        raise ValueError(
            f"Dataset contract v{contract}, expected v{DATASET_CONTRACT_VERSION}"
        )

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)

    if X.ndim != 2 or X.shape[1] != F.LENGTH:
        raise ValueError(f"X must be [N,{F.LENGTH}], got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y length does not match X")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("y must contain only 0 and 1")

    n = X.shape[0]
    meta = {}
    for key in (
        "sample_sha256", "observed_at_utc", "source", "family", "publisher"
    ):
        arr = _as_text(data[key])
        if arr.ndim != 1 or arr.shape[0] != n:
            raise ValueError(f"{key} must be a 1-D array of length {n}")
        meta[key] = arr

    sha = np.char.lower(np.char.strip(meta["sample_sha256"]))
    valid_sha = np.fromiter(
        (
            len(x) == 64
            and all(c in "0123456789abcdef" for c in x)
            for x in sha.tolist()
        ),
        dtype=bool,
        count=n,
    )
    if not valid_sha.all():
        raise ValueError(
            f"{int((~valid_sha).sum())} sample_sha256 values are not full SHA-256"
        )

    try:
        observed_text = [
            value[:-1] if value.endswith("Z")
            else value[:-6] if value.endswith("+00:00")
            else value
            for value in meta["observed_at_utc"].astype(str).tolist()
        ]
        observed = np.asarray(observed_text).astype("datetime64[s]")
    except (TypeError, ValueError) as exc:
        raise ValueError("observed_at_utc contains invalid timestamps") from exc

    if np.isnat(observed).any():
        raise ValueError("observed_at_utc contains NaT")

    return Dataset(
        X=X,
        y=y,
        sample_sha256=sha,
        observed_at_utc=observed,
        source=meta["source"],
        family=meta["family"],
        publisher=meta["publisher"],
    )


def temporal_split(
    dataset: Dataset,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> Split:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be between 0 and 1")
    if not (0.0 < validation_fraction < 1.0):
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train + validation fractions must leave a test set")

    n = dataset.X.shape[0]
    if n < 20:
        raise ValueError("At least 20 samples are required even for an audit")

    times = dataset.observed_at_utc
    order = np.argsort(times, kind="stable")

    train_pos = max(1, min(n - 2, int(math.floor(n * train_fraction))))
    val_pos = max(train_pos + 1, min(n - 1, int(
        math.floor(n * (train_fraction + validation_fraction))
    )))

    train_cutoff = times[order[train_pos - 1]]
    validation_cutoff = times[order[val_pos - 1]]

    train = times <= train_cutoff
    validation = (times > train_cutoff) & (times <= validation_cutoff)
    test = times > validation_cutoff

    if not train.any() or not validation.any() or not test.any():
        raise ValueError(
            "Temporal timestamps do not produce three non-empty partitions. "
            "Collect data on more distinct dates."
        )

    if times[train].max() >= times[validation].min():
        raise ValueError("Train/validation chronology is not strict")
    if times[validation].max() >= times[test].min():
        raise ValueError("Validation/test chronology is not strict")

    assert not np.any(train & validation)
    assert not np.any(train & test)
    assert not np.any(validation & test)

    return Split(
        train=train,
        validation=validation,
        test=test,
        train_cutoff=train_cutoff,
        validation_cutoff=validation_cutoff,
    )


def assert_no_sha_leakage(dataset: Dataset, split: Split) -> None:
    groups = {
        "train": set(dataset.sample_sha256[split.train].tolist()),
        "validation": set(dataset.sample_sha256[split.validation].tolist()),
        "test": set(dataset.sample_sha256[split.test].tolist()),
    }

    pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]
    for left, right in pairs:
        overlap = groups[left].intersection(groups[right])
        if overlap:
            raise ValueError(
                f"SHA-256 leakage between {left} and {right}: "
                f"{len(overlap)} duplicate binaries"
            )


def choose_threshold(
    y_validation: np.ndarray,
    scores_validation: np.ndarray,
    target_tpr: float,
) -> float:
    y_validation = np.asarray(y_validation, dtype=np.int64)
    scores_validation = np.asarray(scores_validation, dtype=np.float64)

    if y_validation.shape != scores_validation.shape:
        raise ValueError("Validation labels and scores have different shapes")
    if not np.isfinite(scores_validation).all():
        raise ValueError("Validation scores contain NaN/inf")
    if not (0.0 < target_tpr <= 1.0):
        raise ValueError("target_tpr must be in (0,1]")

    malicious_scores = np.sort(scores_validation[y_validation == 1])[::-1]
    if malicious_scores.size == 0:
        raise ValueError("Validation set contains no malicious samples")

    required = int(math.ceil(target_tpr * malicious_scores.size))
    required = min(max(required, 1), malicious_scores.size)
    return float(malicious_scores[required - 1])


def binomial_upper_95(successes: int, trials: int) -> float:
    if trials <= 0:
        return float("nan")
    if successes < 0 or successes > trials:
        raise ValueError("Invalid binomial counts")
    if successes == trials:
        return 1.0
    return float(beta.ppf(0.95, successes + 1, trials - successes))


def binomial_lower_95(successes: int, trials: int) -> float:
    if trials <= 0:
        return float("nan")
    if successes < 0 or successes > trials:
        raise ValueError("Invalid binomial counts")
    if successes == 0:
        return 0.0
    return float(beta.ppf(0.05, successes, trials - successes + 1))


def metrics_at_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict:
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    if y.shape != scores.shape:
        raise ValueError("Labels and scores have different shapes")
    if not np.isfinite(scores).all():
        raise ValueError("Scores contain NaN/inf")

    pred = scores >= threshold
    benign = y == 0
    malicious = y == 1

    n_benign = int(benign.sum())
    n_malicious = int(malicious.sum())
    fp = int((pred & benign).sum())
    tp = int((pred & malicious).sum())

    fpr = fp / n_benign if n_benign else float("nan")
    tpr = tp / n_malicious if n_malicious else float("nan")

    return {
        "n": int(y.size),
        "benign": n_benign,
        "malicious": n_malicious,
        "fp": fp,
        "tp": tp,
        "fpr": fpr,
        "tpr": tpr,
        "fpr_upper_95": binomial_upper_95(fp, n_benign),
        "tpr_lower_95": binomial_lower_95(tp, n_malicious),
    }


def unseen_family_mask(dataset: Dataset, split: Split) -> np.ndarray:
    known_family = _known(dataset.family)
    train_families = set(
        np.char.lower(np.char.strip(
            dataset.family[split.train & (dataset.y == 1) & known_family]
        )).tolist()
    )
    test_family = np.char.lower(np.char.strip(dataset.family))
    return (
        split.test
        & (dataset.y == 1)
        & known_family
        & np.fromiter(
            (x not in train_families for x in test_family.tolist()),
            dtype=bool,
            count=test_family.size,
        )
    )


def unseen_publisher_mask(dataset: Dataset, split: Split) -> np.ndarray:
    known_publisher = _known(dataset.publisher)
    train_publishers = set(
        np.char.lower(np.char.strip(
            dataset.publisher[split.train & (dataset.y == 0) & known_publisher]
        )).tolist()
    )
    test_publisher = np.char.lower(np.char.strip(dataset.publisher))
    return (
        split.test
        & (dataset.y == 0)
        & known_publisher
        & np.fromiter(
            (x not in train_publishers for x in test_publisher.tolist()),
            dtype=bool,
            count=test_publisher.size,
        )
    )


def audit_dataset(dataset: Dataset, split: Split) -> list[str]:
    assert_no_sha_leakage(dataset, split)

    lines = []
    lines.append(
        f"samples={dataset.X.shape[0]} benign={int((dataset.y == 0).sum())} "
        f"malicious={int((dataset.y == 1).sum())}"
    )
    lines.append(
        f"train={int(split.train.sum())} validation={int(split.validation.sum())} "
        f"test={int(split.test.sum())}"
    )
    lines.append(
        f"train_cutoff={split.train_cutoff} "
        f"validation_cutoff={split.validation_cutoff}"
    )

    for name, mask in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        lines.append(
            f"{name}: benign={int(((dataset.y == 0) & mask).sum())} "
            f"malicious={int(((dataset.y == 1) & mask).sum())}"
        )

    uf = unseen_family_mask(dataset, split)
    up = unseen_publisher_mask(dataset, split)

    lines.append(
        f"test_unseen_family_malicious={int(uf.sum())}"
    )
    lines.append(
        f"test_unseen_publisher_benign={int(up.sum())}"
    )

    known_family_count = len(set(
        np.char.lower(np.char.strip(dataset.family[
            (dataset.y == 1) & _known(dataset.family)
        ])).tolist()
    ))
    known_publisher_count = len(set(
        np.char.lower(np.char.strip(dataset.publisher[
            (dataset.y == 0) & _known(dataset.publisher)
        ])).tolist()
    ))
    lines.append(f"known_malware_families={known_family_count}")
    lines.append(f"known_benign_publishers={known_publisher_count}")
    lines.append("sha256_cross_split_leakage=0")
    return lines


def _synthetic_dataset() -> Dataset:
    n_days = 100
    per_day = 10
    n = n_days * per_day

    X = np.zeros((n, F.LENGTH), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    observed = np.empty(n, dtype="datetime64[s]")
    sha = np.empty(n, dtype="<U64")
    source = np.empty(n, dtype="<U32")
    family = np.empty(n, dtype="<U32")
    publisher = np.empty(n, dtype="<U32")

    idx = 0
    for day in range(n_days):
        date = np.datetime64("2026-01-01", "D") + np.timedelta64(day, "D")
        for j in range(per_day):
            malicious = (j % 5 == 0)
            y[idx] = 1 if malicious else 0
            observed[idx] = date.astype("datetime64[s]")
            X[idx, 0] = (idx % 97) / 97.0
            X[idx, 1] = 0.9 if malicious else 0.1
            sha[idx] = f"{idx:064x}"
            source[idx] = "synthetic"

            if malicious:
                family[idx] = f"family-{day // 10:02d}"
                publisher[idx] = ""
            else:
                family[idx] = ""
                publisher[idx] = f"publisher-{day // 10:02d}"

            idx += 1

    return Dataset(
        X=X,
        y=y,
        sample_sha256=sha,
        observed_at_utc=observed,
        source=source,
        family=family,
        publisher=publisher,
    )


def self_test() -> None:
    dataset = _synthetic_dataset()
    split = temporal_split(dataset)
    assert_no_sha_leakage(dataset, split)

    assert split.train.sum() > 0
    assert split.validation.sum() > 0
    assert split.test.sum() > 0

    scores = np.where(dataset.y == 1, 0.9, 0.1).astype(np.float64)
    threshold = choose_threshold(
        dataset.y[split.validation],
        scores[split.validation],
        target_tpr=0.90,
    )

    report = metrics_at_threshold(
        dataset.y[split.test],
        scores[split.test],
        threshold,
    )

    if report["fp"] != 0:
        raise AssertionError("Synthetic self-test unexpectedly has false positives")
    if report["tp"] != report["malicious"]:
        raise AssertionError("Synthetic self-test unexpectedly misses malware")

    uf = unseen_family_mask(dataset, split)
    up = unseen_publisher_mask(dataset, split)
    if uf.sum() == 0:
        raise AssertionError("Unseen-family subset is empty")
    if up.sum() == 0:
        raise AssertionError("Unseen-publisher subset is empty")

    upper = binomial_upper_95(0, 30_000)
    if not (0.0 < upper < 0.0002):
        raise AssertionError(f"Unexpected 95% upper bound: {upper}")

    tampered = Dataset(
        X=dataset.X,
        y=dataset.y,
        sample_sha256=dataset.sample_sha256.copy(),
        observed_at_utc=dataset.observed_at_utc,
        source=dataset.source,
        family=dataset.family,
        publisher=dataset.publisher,
    )
    train_idx = int(np.flatnonzero(split.train)[0])
    test_idx = int(np.flatnonzero(split.test)[0])
    tampered.sample_sha256[test_idx] = tampered.sample_sha256[train_idx]
    try:
        assert_no_sha_leakage(tampered, split)
    except ValueError:
        pass
    else:
        raise AssertionError("SHA leakage guard did not fire")

    print("RELEASE_VALIDATION_SELF_TEST=PASS")
    print(f"threshold={threshold:.6f}")
    print(f"test_benign={report['benign']}")
    print(f"test_malicious={report['malicious']}")
    print(f"fpr_upper_95_zero_fp_30000={upper:.8f}")
    print(f"unseen_family_samples={int(uf.sum())}")
    print(f"unseen_publisher_samples={int(up.sum())}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Guardian release-validation dataset contract."
    )
    parser.add_argument("--dataset", help="Release-contract v2 .npz to audit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.dataset:
        dataset = load_dataset(args.dataset)
        split = temporal_split(dataset)
        for line in audit_dataset(dataset, split):
            print(line)
        print("RELEASE_DATASET_AUDIT=PASS")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
