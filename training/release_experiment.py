from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aegis_features as F
import release_validation as V

TARGET_TPRS = (0.90, 0.95, 0.99)
CALIBRATION_BINS = 10


def build_model(y_train: np.ndarray, seed: int):
    import lightgbm as lgb

    benign = int((y_train == 0).sum())
    malicious = int((y_train == 1).sum())
    return lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        scale_pos_weight=min(3.0, benign / max(1, malicious)),
        random_state=seed,
        verbose=-1,
    )


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = CALIBRATION_BINS) -> tuple[float, list[dict]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=False), 0, bins - 1)
    total = y.size
    ece = 0.0
    table = []
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            table.append({"bin": b, "lower": float(edges[b]), "upper": float(edges[b + 1]), "count": 0, "mean_score": None, "observed_rate": None})
            continue
        mean_score = float(p[mask].mean())
        observed = float(y[mask].mean())
        ece += (count / total) * abs(mean_score - observed)
        table.append({"bin": b, "lower": float(edges[b]), "upper": float(edges[b + 1]), "count": count, "mean_score": mean_score, "observed_rate": observed})
    return float(ece), table


def subset_rate(y: np.ndarray, scores: np.ndarray, mask: np.ndarray, threshold: float, label: int) -> dict:
    selected = mask & (y == label)
    n = int(selected.sum())
    if n == 0:
        return {"n": 0, "rate": None, "bound_95": None}
    hits = int((scores[selected] >= threshold).sum())
    if label == 1:
        return {"n": n, "rate": hits / n, "bound_95": V.binomial_lower_95(hits, n), "bound": "lower"}
    return {"n": n, "rate": hits / n, "bound_95": V.binomial_upper_95(hits, n), "bound": "upper"}


def run(dataset: V.Dataset, seed: int, train_fraction: float, validation_fraction: float, label: str) -> tuple[dict, object]:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    split = V.temporal_split(dataset, train_fraction=train_fraction, validation_fraction=validation_fraction)
    V.assert_no_sha_leakage(dataset, split)

    X, y = dataset.X, dataset.y
    for name, mask in (("train", split.train), ("validation", split.validation), ("test", split.test)):
        classes = set(np.unique(y[mask]).tolist())
        if classes != {0, 1}:
            raise ValueError(f"{name} partition must contain both classes, has {sorted(classes)}")

    model = build_model(y[split.train], seed)
    model.fit(X[split.train], y[split.train])

    scores = model.predict_proba(X)[:, 1].astype(np.float64)

    unseen_family = V.unseen_family_mask(dataset, split)
    unseen_publisher = V.unseen_publisher_mask(dataset, split)

    y_test = y[split.test]
    s_test = scores[split.test]
    ece, reliability = expected_calibration_error(y_test, s_test)

    operating_points = []
    for target in TARGET_TPRS:
        threshold = V.choose_threshold(y[split.validation], scores[split.validation], target_tpr=target)
        validation = V.metrics_at_threshold(y[split.validation], scores[split.validation], threshold)
        test = V.metrics_at_threshold(y_test, s_test, threshold)
        operating_points.append({
            "target_tpr": target,
            "threshold": threshold,
            "validation": validation,
            "test": test,
            "test_unseen_family_tpr": subset_rate(y, scores, unseen_family, threshold, 1),
            "test_unseen_publisher_fpr": subset_rate(y, scores, unseen_publisher, threshold, 0),
        })

    importances = model.feature_importances_
    order = np.argsort(importances)[::-1][:10]

    report = {
        "label": label,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "feature_schema_version": F.VERSION,
        "dataset_contract_version": V.DATASET_CONTRACT_VERSION,
        "seed": seed,
        "split": {
            "method": "temporal",
            "train": int(split.train.sum()),
            "validation": int(split.validation.sum()),
            "test": int(split.test.sum()),
            "train_cutoff": str(split.train_cutoff),
            "validation_cutoff": str(split.validation_cutoff),
            "sha256_cross_split_leakage": 0,
        },
        "test_auc": float(roc_auc_score(y_test, s_test)),
        "test_brier": float(brier_score_loss(y_test, s_test)),
        "test_ece": ece,
        "reliability": reliability,
        "operating_points": operating_points,
        "top_features": [{"name": F.FEATURE_NAMES[i], "importance": int(importances[i])} for i in order],
    }
    return report, model


def fmt(value, pattern="{:.4%}") -> str:
    if value is None:
        return "n/a"
    return pattern.format(value)


def to_markdown(report: dict) -> str:
    s = report["split"]
    lines = [
        f"# Results: {report['label']}",
        "",
        f"Generated {report['generated_at_utc']}, feature schema v{report['feature_schema_version']}, dataset contract v{report['dataset_contract_version']}, seed {report['seed']}.",
        "",
        f"Temporal split: train {s['train']:,} (until {s['train_cutoff']}), validation {s['validation']:,} (until {s['validation_cutoff']}), test {s['test']:,}. SHA-256 overlap between partitions: 0.",
        "",
        f"Test AUC {report['test_auc']:.4f}, Brier score {report['test_brier']:.4f}, ECE ({CALIBRATION_BINS} bins) {report['test_ece']:.4f}.",
        "",
        "Thresholds are fixed on the validation partition and applied unchanged to the later test partition. Bounds are one-sided 95% Clopper-Pearson.",
        "",
        "| Target TPR | Threshold | Test TPR | TPR lower 95% | Test FPR | FPR upper 95% | FP per 10k benign | Unseen-family TPR (n) | Unseen-publisher FPR (n) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for op in report["operating_points"]:
        t = op["test"]
        uf = op["test_unseen_family_tpr"]
        up = op["test_unseen_publisher_fpr"]
        lines.append(
            f"| {op['target_tpr']:.0%} | {op['threshold']:.4f} | {fmt(t['tpr'], '{:.2%}')} | {fmt(t['tpr_lower_95'], '{:.2%}')} | "
            f"{fmt(t['fpr'])} | {fmt(t['fpr_upper_95'])} | {t['fpr'] * 10_000:.1f} | "
            f"{fmt(uf['rate'], '{:.2%}')} ({uf['n']}) | {fmt(up['rate'])} ({up['n']}) |"
        )
    lines += ["", "Most used features:", ""]
    for item in report["top_features"]:
        lines.append(f"- `{item['name']}` ({item['importance']})")
    lines.append("")
    return "\n".join(lines)


def synthetic_drift_dataset(seed: int = 7, days: int = 120, per_day: int = 60, malicious_rate: float = 0.15) -> V.Dataset:
    rng = np.random.default_rng(seed)
    n = days * per_day
    binary = [F.IDX_IS_PE, F.IDX_IS_DLL, F.IDX_TIMESTAMP_BAD, F.IDX_IS_64BIT, F.IDX_IS_GUI_SUBSYSTEM, F.IDX_RELOCS_STRIPPED]
    continuous = [i for i in range(F.LENGTH) if i not in binary]

    family_offsets = [rng.normal(0.0, 0.06, F.LENGTH) for _ in range(days // 10 + 1)]
    publisher_offsets = [rng.normal(0.0, 0.04, F.LENGTH) for _ in range(days // 10 + 1)]

    X = np.zeros((n, F.LENGTH), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    observed = np.empty(n, dtype="datetime64[s]")
    sha = np.empty(n, dtype="<U64")
    source = np.empty(n, dtype="<U32")
    family = np.empty(n, dtype="<U32")
    publisher = np.empty(n, dtype="<U32")

    idx = 0
    for day in range(days):
        date = (np.datetime64("2026-01-01", "D") + np.timedelta64(day, "D")).astype("datetime64[s]")
        signal = 0.12 * (1.0 - 0.5 * day / days)
        for j in range(per_day):
            malicious = j < max(1, int(round(per_day * malicious_rate)))
            group = day // 10
            row = rng.normal(0.5, 0.15, F.LENGTH)
            if malicious:
                row += family_offsets[group]
                row[[F.IDX_HEADER_ENTROPY, F.IDX_MID_ENTROPY, F.IDX_TAIL_ENTROPY, F.IDX_MAX_ENTROPY]] += signal
                row[F.IDX_OVERLAY_RATIO] += signal
            else:
                row += publisher_offsets[group]
            row = np.clip(row, 0.0, 1.0)
            for b in binary:
                p = 0.5 + (0.1 if malicious and b == F.IDX_TIMESTAMP_BAD else 0.0)
                row[b] = float(rng.random() < p)
            X[idx, continuous] = row[continuous]
            X[idx, binary] = row[binary]
            y[idx] = 1 if malicious else 0
            observed[idx] = date
            sha[idx] = f"{seed:08x}{idx:056x}"
            source[idx] = "synthetic"
            family[idx] = f"family-{group:02d}" if malicious else ""
            publisher[idx] = "" if malicious else f"publisher-{group:02d}"
            idx += 1

    return V.Dataset(X=X, y=y, sample_sha256=sha, observed_at_utc=observed, source=source, family=family, publisher=publisher)


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal, leakage-checked training and evaluation of the Guardian classifier.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="Release-contract v2 .npz")
    group.add_argument("--synthetic", action="store_true", help="Run on generated data with temporal drift; plumbing demonstration only")
    parser.add_argument("--out", default="results", help="Directory for report.json and report.md")
    parser.add_argument("--onnx", help="Also export the trained model to this ONNX path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    if args.synthetic:
        dataset = synthetic_drift_dataset()
        label = "synthetic drift data (pipeline demonstration, not a detection result)"
    else:
        dataset = V.load_dataset(args.dataset)
        label = os.path.basename(args.dataset)

    report, model = run(dataset, args.seed, args.train_fraction, args.validation_fraction, label)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    markdown = to_markdown(report)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(markdown)

    if args.onnx:
        import train_guardian
        train_guardian.export_onnx(model, args.onnx)


if __name__ == "__main__":
    main()
