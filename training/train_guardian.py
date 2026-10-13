from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aegis_features as F


def iter_files(root: str):
    for directory, _, names in os.walk(root):
        for name in names:
            yield os.path.join(directory, name)


def extract(args) -> None:
    rows: list[list[float]] = []
    labels: list[int] = []

    for label, root in ((0, args.benign), (1, args.malicious)):
        if not root:
            continue

        count = 0
        skipped = 0

        for path in iter_files(root):
            vector = F.compute(path)
            if vector is None:
                skipped += 1
                continue

            rows.append(vector)
            labels.append(label)
            count += 1

            if count % 2000 == 0:
                print(f"  {root}: {count} files", flush=True)

        print(f"{'malicious' if label else 'benign':>9}: {count} extracted, {skipped} unreadable")

    if not rows:
        sys.exit("No features extracted. Check the paths.")

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    np.savez_compressed(args.out, X=X, y=y,
                        schema_version=F.VERSION,
                        feature_names=np.array(F.FEATURE_NAMES))

    print(f"\nWrote {args.out}: {X.shape[0]} samples x {X.shape[1]} features "
          f"({int(y.sum())} malicious, {int((1 - y).sum())} benign)")


def train(args) -> None:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve

    data = np.load(args.features, allow_pickle=True)
    X, y = data["X"], data["y"]

    schema = int(data.get("schema_version", -1))
    if schema != F.VERSION:
        sys.exit(f"Feature file is schema v{schema}, this code is v{F.VERSION}. "
                 "Re-extract; a mismatched schema trains a model that cannot be used.")

    if X.shape[1] != F.LENGTH:
        sys.exit(f"Feature file has {X.shape[1]} columns, schema expects {F.LENGTH}.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=64,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        scale_pos_weight=min(3.0, (y_train == 0).sum() / max(1, (y_train == 1).sum())),
        random_state=42,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              eval_metric="auc", callbacks=[lgb.log_evaluation(100)])

    scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, scores)
    fpr, tpr, thresholds = roc_curve(y_test, scores)

    print(f"\nAUC: {auc:.4f}")
    print("\nThe number that decides whether this ships is the false-positive")
    print("rate, not the AUC. One false positive per thousand clean files is")
    print("still hundreds of wrong alarms on a normal machine.\n")
    print(f"{'detection':>10}  {'false positives':>16}  {'threshold':>10}")

    for target in (0.90, 0.95, 0.98, 0.99):
        idx = int(np.searchsorted(tpr, target))
        if idx < len(fpr):
            print(f"{tpr[idx]:>9.1%}  {fpr[idx]:>15.4%}  {thresholds[idx]:>10.4f}")

    print("\nMost useful features:")
    order = np.argsort(model.feature_importances_)[::-1][:12]
    for i in order:
        print(f"  {F.FEATURE_NAMES[i]:<20} {model.feature_importances_[i]}")

    export_onnx(model, args.out)


def export_onnx(model, path: str) -> None:
    import onnxmltools

    from onnxmltools.convert.common.data_types import FloatTensorType

    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=[("features", FloatTensorType([None, F.LENGTH]))],
        target_opset=13,
    )

    with open(path, "wb") as handle:
        handle.write(onnx_model.SerializeToString())

    size = os.path.getsize(path)
    print(f"\nWrote {path} ({size:,} bytes)")

    if size < 4096:
        print("WARNING: GuardianOnnxModel rejects models under 4096 bytes as "
              "placeholders. Something went wrong.")

    verify(path)


def verify(path: str) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipping the load check.")
        return

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]

    print(f"Input:  {input_meta.name} {input_meta.shape}")
    print(f"Output: {[o.name for o in session.get_outputs()]}")

    width = input_meta.shape[-1]
    if isinstance(width, int) and width != F.LENGTH:
        print(f"WARNING: model expects {width} features, schema produces {F.LENGTH}. "
              "GuardianOnnxModel will refuse to load this.")
    else:
        print(f"Input width matches schema v{F.VERSION} ({F.LENGTH} features).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Guardian model.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_extract = sub.add_parser("extract", help="build a feature table")
    p_extract.add_argument("--benign", required=True)
    p_extract.add_argument("--malicious", required=True)
    p_extract.add_argument("--out", default="features.npz")
    p_extract.set_defaults(func=extract)

    p_train = sub.add_parser("train", help="train and export to ONNX")
    p_train.add_argument("--features", default="features.npz")
    p_train.add_argument("--out", default="guardian.onnx")
    p_train.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
