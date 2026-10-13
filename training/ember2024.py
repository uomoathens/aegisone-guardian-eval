from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import release_experiment as E
import release_validation as V

EMBER_START = np.datetime64("2023-09-24T00:00:00", "s")
FILE_TYPES = ("Dot_Net", "Win64", "Win32")
TARGET_FPRS = (0.01, 0.001)
CATEGORICAL = [2, 3, 4, 5, 6, 701, 702]
EMBER_LGBM_PARAMS = {
    "objective": "binary",
    "boosting": "gbdt",
    "num_iterations": 500,
    "learning_rate": 0.1,
    "max_depth": -1,
    "num_leaves": 64,
    "min_data_in_leaf": 100,
    "min_sum_hessian_in_leaf": 0.001,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "feature_fraction": 0.9,
    "feature_fraction_bynode": 0.9,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "is_unbalance": True,
    "num_threads": 0,
    "bin_construct_sample_cnt": 50000,
    "verbosity": -1,
}
QUICK_OVERRIDES = {"num_iterations": 25, "min_data_in_leaf": 2, "min_data_per_group": 1, "min_sum_hessian_in_leaf": 1e-6}


@dataclass
class Subset:
    source: np.ndarray
    rows: np.ndarray
    y: np.ndarray
    sha256: np.ndarray
    observed: np.ndarray
    week: np.ndarray
    family: np.ndarray

    def matrix(self, rows: np.ndarray | None = None) -> np.ndarray:
        selected = self.rows if rows is None else rows
        return np.ascontiguousarray(self.source[selected], dtype=np.float32)


def log(message: str) -> None:
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        suffix = f"  [peak memory {peak:.1f} GB]"
    except ImportError:
        suffix = ""
    print(f"{datetime.now(timezone.utc).strftime('%H:%M:%S')} {message}{suffix}", flush=True)


def vector_dim() -> int:
    import thrember

    return int(thrember.PEFeatureExtractor().dim)


def open_vectors(data_dir: str, subset: str, dim: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.memmap(os.path.join(data_dir, f"X_{subset}.dat"), dtype=np.float32, mode="r")
    if X.size % dim:
        raise ValueError(f"X_{subset}.dat size is not a multiple of {dim}")
    X = X.reshape(-1, dim)
    y = np.memmap(os.path.join(data_dir, f"y_{subset}.dat"), dtype=np.int32, mode="r")
    return X, y


def jsonl_paths(data_dir: str, subset: str) -> list[str]:
    names = sorted(n for n in os.listdir(data_dir) if n.endswith(".jsonl") and subset in n)
    if not names:
        raise FileNotFoundError(f"No {subset} .jsonl files in {data_dir}")
    return [os.path.join(data_dir, n) for n in names]


def read_metadata(paths: list[str]) -> dict[str, np.ndarray]:
    sha, date, label, family, file_type = [], [], [], [], []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                sha.append(str(record["sha256"]).lower())
                date.append(int(record["first_submission_date"]))
                value = record.get("label")
                label.append(-1 if value is None else int(value))
                family.append(record.get("family") or "")
                file_type.append(record.get("file_type") or "")
    return {
        "sha256": np.asarray(sha),
        "date": np.asarray(date, dtype=np.int64),
        "label": np.asarray(label, dtype=np.int64),
        "family": np.asarray(family),
        "file_type": np.asarray(file_type),
    }


def load_subset(data_dir: str, subset: str, file_type: str, dim: int) -> tuple[Subset, dict]:
    X, y = open_vectors(data_dir, subset, dim)
    meta = read_metadata(jsonl_paths(data_dir, subset))
    n = meta["sha256"].size
    if X.shape[0] != n or y.shape[0] != n:
        raise ValueError(f"{subset}: {X.shape[0]} vectors, {y.shape[0]} labels, {n} metadata rows")
    labelled = meta["label"] >= 0
    if not np.array_equal(np.asarray(y[labelled], dtype=np.int64), meta["label"][labelled]):
        raise ValueError(f"{subset}: vector labels do not match metadata order")

    keep = labelled & (meta["file_type"] == file_type)
    observed = meta["date"][keep].astype("datetime64[s]")
    week = ((observed - EMBER_START) // np.timedelta64(7, "D")).astype(np.int64)
    result = Subset(
        source=X,
        rows=np.flatnonzero(keep).astype(np.int64),
        y=meta["label"][keep],
        sha256=meta["sha256"][keep],
        observed=observed,
        week=week,
        family=np.char.lower(np.char.strip(meta["family"][keep].astype(str))),
    )
    audit = {
        "rows_total": int(n),
        "rows_unlabelled": int((~labelled).sum()),
        "rows_other_file_type": int((labelled & (meta["file_type"] != file_type)).sum()),
        "rows_used": int(keep.sum()),
        "benign": int((result.y == 0).sum()),
        "malicious": int((result.y == 1).sum()),
    }
    return result, audit


def take(s: Subset, mask: np.ndarray) -> Subset:
    return Subset(source=s.source, rows=s.rows[mask], y=s.y[mask], sha256=s.sha256[mask], observed=s.observed[mask], week=s.week[mask], family=s.family[mask])


def predict(model, s: Subset, rows: np.ndarray | None = None, chunk: int = 20_000) -> np.ndarray:
    selected = s.rows if rows is None else rows
    out = np.empty(selected.size, dtype=np.float64)
    for start in range(0, selected.size, chunk):
        part = selected[start:start + chunk]
        out[start:start + part.size] = model.predict(s.matrix(part))
    return out


def threshold_at_fpr(benign_scores: np.ndarray, target_fpr: float) -> float:
    ordered = np.sort(np.asarray(benign_scores, dtype=np.float64))[::-1]
    if ordered.size == 0:
        raise ValueError("No benign scores to set a threshold")
    k = int(np.floor(target_fpr * ordered.size))
    if k >= ordered.size:
        return float(-np.inf)
    return float(np.nextafter(ordered[k], np.inf))


def train(source: np.ndarray, rows: np.ndarray, y: np.ndarray, params: dict):
    import lightgbm as lgb

    class RowSequence(lgb.Sequence):
        batch_size = 4096

        def __init__(self, data: np.ndarray, selected: np.ndarray):
            self.data = data
            self.selected = selected

        def __len__(self) -> int:
            return int(self.selected.size)

        def __getitem__(self, index):
            return np.asarray(self.data[self.selected[index]], dtype=np.float64)

    dataset = lgb.Dataset(RowSequence(source, rows), np.asarray(y, dtype=np.float64), categorical_feature=CATEGORICAL, free_raw_data=True)
    return lgb.train(dict(params), dataset)


def rate_block(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    m = V.metrics_at_threshold(y, scores, threshold)
    return {k: m[k] for k in ("benign", "malicious", "fp", "tp", "fpr", "tpr", "fpr_upper_95", "tpr_lower_95")}


def detection_block(scores: np.ndarray, threshold: float) -> dict:
    n = int(scores.size)
    if n == 0:
        return {"n": 0, "tpr": None, "tpr_lower_95": None}
    tp = int((scores >= threshold).sum())
    return {"n": n, "tpr": tp / n, "tpr_lower_95": V.binomial_lower_95(tp, n)}


def sha_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.isin(b, a)


def versions() -> dict:
    from importlib import metadata

    result = {"python": sys.version.split()[0]}
    for name in ("thrember", "lightgbm", "numpy", "scikit-learn", "signify", "oscrypto"):
        try:
            result[name] = metadata.version(name)
            raw = metadata.distribution(name).read_text("direct_url.json")
            if raw:
                commit = json.loads(raw).get("vcs_info", {}).get("commit_id")
                if commit:
                    result[name] += f" ({commit[:12]})"
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def run(data_dir: str, file_type: str, validation_weeks: int, seed: int, quick: bool, random_baseline: bool) -> dict:
    from sklearn.metrics import brier_score_loss, roc_auc_score

    params = dict(EMBER_LGBM_PARAMS)
    params["seed"] = seed
    if quick:
        params.update(QUICK_OVERRIDES)

    dim = vector_dim()
    log(f"reading train metadata and vectors (dimension {dim})")
    train_all, train_audit = load_subset(data_dir, "train", file_type, dim)
    log(f"train rows used {train_audit['rows_used']:,}; reading test")
    test, test_audit = load_subset(data_dir, "test", file_type, dim)
    log(f"test rows used {test_audit['rows_used']:,}; reading challenge")
    challenge, challenge_audit = load_subset(data_dir, "challenge", file_type, dim)
    challenge = take(challenge, challenge.y == 1)
    log(f"challenge malicious rows {challenge.y.size:,}")

    train_weeks = np.unique(train_all.week)
    if train_weeks.size <= validation_weeks:
        raise ValueError(f"Only {train_weeks.size} training weeks; cannot hold out {validation_weeks}")
    first_validation_week = int(train_weeks[-validation_weeks])
    fit_mask = train_all.week < first_validation_week
    fit = take(train_all, fit_mask)
    validation = take(train_all, ~fit_mask)

    if fit.observed.max() >= validation.observed.min():
        raise ValueError("Fit and validation partitions overlap in time")
    if validation.observed.max() >= test.observed.min():
        raise ValueError("Validation and test partitions overlap in time")
    for name, part in (("fit", fit), ("validation", validation), ("test", test)):
        if set(np.unique(part.y).tolist()) != {0, 1}:
            raise ValueError(f"{name} partition does not contain both classes")

    leak_val = sha_overlap(fit.sha256, validation.sha256)
    leak_test = sha_overlap(np.concatenate([fit.sha256, validation.sha256]), test.sha256)
    leak_challenge = sha_overlap(np.concatenate([fit.sha256, validation.sha256]), challenge.sha256)
    validation = take(validation, ~leak_val)
    test = take(test, ~leak_test)
    challenge = take(challenge, ~leak_challenge)
    for name, part in (("validation", validation), ("test", test)):
        if set(np.unique(part.y).tolist()) != {0, 1}:
            raise ValueError(f"{name} partition lost a class after removing SHA-256 duplicates")
    log(f"SHA-256 duplicates removed: validation {int(leak_val.sum()):,}, test {int(leak_test.sum()):,}, challenge {int(leak_challenge.sum()):,}")

    log(f"training temporal model on {fit.y.size:,} rows")
    model = train(fit.source, fit.rows, fit.y, params)
    gc.collect()
    log("scoring validation, test and challenge")
    s_val = predict(model, validation)
    s_test = predict(model, test)
    s_chal = predict(model, challenge) if challenge.y.size else np.zeros(0)
    del model
    gc.collect()

    fit_families = set(fit.family[(fit.y == 1) & (fit.family != "")].tolist())
    unseen_test = (test.y == 1) & (test.family != "") & ~np.isin(test.family, list(fit_families))

    random_model_scores = None
    if random_baseline:
        rng = np.random.default_rng(seed)
        order = rng.permutation(train_all.y.size)
        cut = int(round(train_all.y.size * 0.85))
        r_fit, r_val = np.sort(order[:cut]), np.sort(order[cut:])
        log(f"training random-split baseline on {r_fit.size:,} rows")
        r_model = train(train_all.source, train_all.rows[r_fit], train_all.y[r_fit], params)
        gc.collect()
        log("scoring random-split baseline")
        random_model_scores = {
            "validation_y": train_all.y[r_val],
            "validation": predict(r_model, train_all, train_all.rows[r_val]),
            "test": predict(r_model, test),
            "challenge": predict(r_model, challenge) if challenge.y.size else np.zeros(0),
        }
        del r_model
        gc.collect()

    log("computing metrics")
    ece, reliability = E.expected_calibration_error(test.y, s_test)
    test_weeks = np.unique(test.week)

    operating = []
    for target in TARGET_FPRS:
        rows = []
        if random_model_scores is not None:
            vy, vs = random_model_scores["validation_y"], random_model_scores["validation"]
            theta = threshold_at_fpr(vs[vy == 0], target)
            rows.append({
                "protocol": "random 15% hold-out of training weeks",
                "threshold": theta,
                "validation": rate_block(vy, vs, theta),
                "test": rate_block(test.y, random_model_scores["test"], theta),
                "challenge": detection_block(random_model_scores["challenge"], theta),
                "unseen_family": detection_block(random_model_scores["test"][unseen_test], theta),
            })
        theta_val = threshold_at_fpr(s_val[validation.y == 0], target)
        rows.append({
            "protocol": f"temporal: last {validation_weeks} training weeks",
            "threshold": theta_val,
            "validation": rate_block(validation.y, s_val, theta_val),
            "test": rate_block(test.y, s_test, theta_val),
            "challenge": detection_block(s_chal, theta_val),
            "unseen_family": detection_block(s_test[unseen_test], theta_val),
        })
        theta_oracle = threshold_at_fpr(s_test[test.y == 0], target)
        rows.append({
            "protocol": "oracle: threshold read off the test set",
            "threshold": theta_oracle,
            "validation": rate_block(validation.y, s_val, theta_oracle),
            "test": rate_block(test.y, s_test, theta_oracle),
            "challenge": detection_block(s_chal, theta_oracle),
            "unseen_family": detection_block(s_test[unseen_test], theta_oracle),
        })
        weekly = []
        for w in test_weeks:
            mask = test.week == w
            start = EMBER_START + np.timedelta64(int(w) * 7, "D")
            block = rate_block(test.y[mask], s_test[mask], theta_val) if set(np.unique(test.y[mask]).tolist()) == {0, 1} else None
            weekly.append({"week": int(w), "starts": str(start)[:10], "metrics": block})
        operating.append({"target_fpr": target, "rows": rows, "weekly_temporal": weekly})

    return {
        "label": f"EMBER2024 {file_type}",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "quick_mode": quick,
        "seed": seed,
        "lightgbm_params": params,
        "versions": versions(),
        "audit": {
            "train": train_audit,
            "test": test_audit,
            "challenge": challenge_audit,
            "sha256_overlap_removed": {
                "fit_to_validation": int(leak_val.sum()),
                "train_to_test": int(leak_test.sum()),
                "train_to_challenge": int(leak_challenge.sum()),
            },
        },
        "split": {
            "fit": {"n": int(fit.y.size), "benign": int((fit.y == 0).sum()), "malicious": int((fit.y == 1).sum()), "first": str(fit.observed.min())[:10], "last": str(fit.observed.max())[:10]},
            "validation": {"n": int(validation.y.size), "benign": int((validation.y == 0).sum()), "malicious": int((validation.y == 1).sum()), "first": str(validation.observed.min())[:10], "last": str(validation.observed.max())[:10]},
            "test": {"n": int(test.y.size), "benign": int((test.y == 0).sum()), "malicious": int((test.y == 1).sum()), "first": str(test.observed.min())[:10], "last": str(test.observed.max())[:10]},
            "challenge_malicious": int(challenge.y.size),
            "unseen_family_test_malicious": int(unseen_test.sum()),
            "families_in_fit": len(fit_families),
        },
        "test_auc": float(roc_auc_score(test.y, s_test)),
        "test_pauc_fpr_0_01": float(roc_auc_score(test.y, s_test, max_fpr=0.01)),
        "test_brier": float(brier_score_loss(test.y, s_test)),
        "test_ece": ece,
        "reliability": reliability,
        "operating_points": operating,
    }


def pct(value, digits=2) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def to_markdown(r: dict) -> str:
    s = r["split"]
    lines = [
        f"# Results: {r['label']}",
        "",
        f"Generated {r['generated_at_utc']}, seed {r['seed']}" + (", QUICK MODE (not a result)" if r["quick_mode"] else "") + ".",
        "",
        "## Data",
        "",
        "| Partition | Weeks first seen | Benign | Malicious |",
        "|---|---|---|---|",
        f"| Fit | {s['fit']['first']} to {s['fit']['last']} | {s['fit']['benign']:,} | {s['fit']['malicious']:,} |",
        f"| Validation | {s['validation']['first']} to {s['validation']['last']} | {s['validation']['benign']:,} | {s['validation']['malicious']:,} |",
        f"| Test | {s['test']['first']} to {s['test']['last']} | {s['test']['benign']:,} | {s['test']['malicious']:,} |",
        f"| Challenge (evasive) | | | {s['challenge_malicious']:,} |",
        "",
        f"Malware families seen in fit: {s['families_in_fit']:,}. Test malware from families never seen in fit: {s['unseen_family_test_malicious']:,}.",
        f"SHA-256 duplicates removed: {r['audit']['sha256_overlap_removed']}.",
        "",
        f"Test ROC AUC {r['test_auc']:.4f}; standardised partial AUC up to 1% FPR {r['test_pauc_fpr_0_01']:.4f}; Brier {r['test_brier']:.4f}; ECE {r['test_ece']:.4f}.",
        "",
        "## Promised versus delivered false-positive rate",
        "",
        "Each row fixes a threshold for the target FPR on a different data set, then measures the later test weeks. Bounds are one-sided 95% Clopper-Pearson.",
        "",
    ]
    for op in r["operating_points"]:
        lines += [
            f"### Target FPR {pct(op['target_fpr'], 1)}",
            "",
            "| Threshold chosen on | FPR where chosen | Test FPR | FPR upper 95% | Test TPR | TPR lower 95% | Challenge TPR | Unseen-family TPR |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in op["rows"]:
            lines.append(
                f"| {row['protocol']} | {pct(row['validation']['fpr'], 3) if 'oracle' not in row['protocol'] else pct(row['test']['fpr'], 3)} | "
                f"{pct(row['test']['fpr'], 3)} | {pct(row['test']['fpr_upper_95'], 3)} | {pct(row['test']['tpr'])} | {pct(row['test']['tpr_lower_95'])} | "
                f"{pct(row['challenge']['tpr'])} | {pct(row['unseen_family']['tpr'])} |"
            )
        lines += ["", "Test weeks, temporal threshold:", "", "| Week starting | Benign | FPR | Malicious | TPR |", "|---|---|---|---|---|"]
        for wk in op["weekly_temporal"]:
            m = wk["metrics"]
            if m is None:
                lines.append(f"| {wk['starts']} | | n/a | | n/a |")
            else:
                lines.append(f"| {wk['starts']} | {m['benign']:,} | {pct(m['fpr'], 3)} | {m['malicious']:,} | {pct(m['tpr'])} |")
        lines.append("")
    lines += ["## Software", ""]
    for k, v in r["versions"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


def download(data_dir: str, file_type: str) -> None:
    import thrember

    os.makedirs(data_dir, exist_ok=True)
    here = os.getcwd()
    try:
        thrember.download_dataset(os.path.abspath(data_dir), split="all", file_type=file_type)
    finally:
        os.chdir(here)
    cache = os.environ.get("HF_HUB_CACHE") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    shutil.rmtree(cache, ignore_errors=True)
    root = os.path.abspath(data_dir)
    for directory, _, names in os.walk(root):
        if os.path.abspath(directory) == root:
            continue
        for name in names:
            if name.endswith(".jsonl"):
                target = os.path.join(root, name)
                if os.path.exists(target):
                    raise FileExistsError(f"Two downloaded files are both named {name}")
                shutil.move(os.path.join(directory, name), target)
    found = [n for n in os.listdir(root) if n.endswith(".jsonl")]
    for subset in ("train", "test", "challenge"):
        if not any(subset in n for n in found):
            raise FileNotFoundError(f"Download finished but no {subset} .jsonl file is present in {root}")


def vectorize(data_dir: str) -> None:
    import thrember

    thrember.create_vectorized_features(os.path.abspath(data_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="EMBER2024 benchmark under a strictly temporal protocol.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download")
    p.add_argument("--data", required=True)
    p.add_argument("--file-type", choices=FILE_TYPES, default="Dot_Net")

    p = sub.add_parser("vectorize")
    p.add_argument("--data", required=True)

    p = sub.add_parser("run")
    p.add_argument("--data", required=True)
    p.add_argument("--file-type", choices=FILE_TYPES, default="Dot_Net")
    p.add_argument("--out", required=True)
    p.add_argument("--validation-weeks", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-random-baseline", action="store_true")

    args = parser.parse_args()

    if args.command == "download":
        download(args.data, args.file_type)
    elif args.command == "vectorize":
        vectorize(args.data)
    else:
        report = run(args.data, args.file_type, args.validation_weeks, args.seed, args.quick, not args.no_random_baseline)
        log("writing report")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, default=float)
        markdown = to_markdown(report)
        with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as handle:
            handle.write(markdown)
        print(markdown)


if __name__ == "__main__":
    main()
