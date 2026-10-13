from __future__ import annotations

import argparse
import base64
import glob
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aegis_features as F
import dataset_contract as D

UNKNOWN = {"", "unknown", "none", "null", "n/a", "na", "-"}


def _scalar_int(data, key: str) -> int:
    return int(np.asarray(data[key]).reshape(-1)[0])


def load_part(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    missing = sorted(set(D.REQUIRED_KEYS).difference(data.files))
    if missing:
        raise ValueError(f"{path}: missing keys: {', '.join(missing)}")

    if _scalar_int(data, "schema_version") != F.VERSION:
        raise ValueError(f"{path}: feature schema mismatch")
    if _scalar_int(data, "dataset_contract_version") != D.DATASET_CONTRACT_VERSION:
        raise ValueError(f"{path}: dataset contract mismatch")

    X = np.asarray(data["X"], dtype=np.float32)
    y = np.asarray(data["y"], dtype=np.int64)
    if X.ndim != 2 or X.shape[1] != F.LENGTH:
        raise ValueError(f"{path}: invalid X shape {X.shape}")
    if y.shape != (X.shape[0],):
        raise ValueError(f"{path}: invalid y shape {y.shape}")

    result = {"X": X, "y": y}
    for key in (
        "sample_sha256", "observed_at_utc", "source", "family", "publisher"
    ):
        arr = np.asarray(data[key]).astype(str)
        if arr.shape != (X.shape[0],):
            raise ValueError(f"{path}: invalid {key} shape {arr.shape}")
        result[key] = arr
    return result


def merge_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        raise ValueError("No dataset parts")

    X = np.vstack([part["X"] for part in parts])
    y = np.concatenate([part["y"] for part in parts])
    sha = np.concatenate([part["sample_sha256"] for part in parts]).astype(str)
    observed = np.concatenate(
        [part["observed_at_utc"] for part in parts]
    ).astype(str)
    source = np.concatenate([part["source"] for part in parts]).astype(str)
    family = np.concatenate([part["family"] for part in parts]).astype(str)
    publisher = np.concatenate([part["publisher"] for part in parts]).astype(str)

    observed_text = [
        value[:-1] if value.endswith("Z")
        else value[:-6] if value.endswith("+00:00")
        else value
        for value in observed.astype(str).tolist()
    ]
    times = np.asarray(observed_text).astype("datetime64[s]")
    order = np.argsort(times, kind="stable")

    keep = []
    seen: dict[str, int] = {}

    for idx in order.tolist():
        digest = sha[idx].strip().lower()
        if len(digest) != 64:
            raise ValueError("Invalid full SHA-256 in merged input")

        previous = seen.get(digest)
        if previous is None:
            seen[digest] = idx
            keep.append(idx)
            continue

        if int(y[previous]) != int(y[idx]):
            raise ValueError(
                "Conflicting labels for SHA-256 " + digest
            )

        if family[previous].strip().lower() in UNKNOWN:
            if family[idx].strip().lower() not in UNKNOWN:
                family[previous] = family[idx]
        if publisher[previous].strip().lower() in UNKNOWN:
            if publisher[idx].strip().lower() not in UNKNOWN:
                publisher[previous] = publisher[idx]

    keep = np.asarray(keep, dtype=np.int64)

    return D.build_payload(
        X=X[keep],
        y=y[keep],
        schema_version=F.VERSION,
        feature_names=F.FEATURE_NAMES,
        sample_sha256=sha[keep],
        observed_at_utc=observed[keep],
        source=source[keep],
        family=family[keep],
        publisher=publisher[keep],
    )


def self_test() -> None:
    def part(start: int, label: int, count: int):
        X = np.zeros((count, F.LENGTH), dtype=np.float32)
        X[:, 0] = np.arange(start, start + count, dtype=np.float32)
        y = np.full(count, label, dtype=np.int64)
        sha = [f"{i:064x}" for i in range(start, start + count)]
        return {
            "X": X,
            "y": y,
            "sample_sha256": np.asarray(sha),
            "observed_at_utc": np.asarray(
                [f"2026-01-{(i % 28) + 1:02d}T00:00:00Z" for i in range(count)]
            ),
            "source": np.asarray(["synthetic"] * count),
            "family": np.asarray(["family-a" if label else ""] * count),
            "publisher": np.asarray(["" if label else "publisher-a"] * count),
        }

    a = part(0, 0, 5)
    b = part(5, 1, 5)
    merged = merge_parts([a, b])
    if merged["X"].shape != (10, F.LENGTH):
        raise AssertionError("Unexpected merged shape")

    c = part(0, 0, 1)
    merged2 = merge_parts([a, c])
    if merged2["X"].shape[0] != 5:
        raise AssertionError("SHA dedupe failed")

    bad = part(0, 1, 1)
    try:
        merge_parts([a, bad])
    except ValueError:
        pass
    else:
        raise AssertionError("Conflicting-label guard failed")

    raw = D.payload_to_bytes(merged)
    with np.load(io.BytesIO(raw), allow_pickle=False) as check:
        missing = set(D.REQUIRED_KEYS).difference(check.files)
        if missing:
            raise AssertionError("Serialized payload lost keys")

    print("MERGE_RELEASE_SELF_TEST=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge", nargs="*")
    parser.add_argument("--out", default="dataset-v2.npz")
    parser.add_argument("--emit-base64", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    paths = []
    for pattern in args.merge or []:
        paths.extend(sorted(glob.glob(pattern)))

    if not paths:
        sys.exit("No release-contract v2 files matched.")

    parts = [load_part(path) for path in paths]
    payload = merge_parts(parts)

    if args.emit_base64:
        print(D.payload_to_base64(payload))
    else:
        D.save_payload(args.out, payload)
        print(f"Wrote {args.out}: {payload['X'].shape[0]:,} unique samples")


if __name__ == "__main__":
    main()
