from __future__ import annotations

import base64
import hashlib
import io
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

DATASET_CONTRACT_VERSION = 2

REQUIRED_KEYS = (
    "X",
    "y",
    "schema_version",
    "dataset_contract_version",
    "feature_names",
    "sample_sha256",
    "observed_at_utc",
    "source",
    "family",
    "publisher",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _text_array(values: Iterable[str]) -> np.ndarray:
    values = ["" if value is None else str(value) for value in values]
    width = max(1, max((len(value) for value in values), default=1))
    return np.asarray(values, dtype=f"<U{width}")


def build_payload(
    *,
    X: np.ndarray,
    y: np.ndarray,
    schema_version: int,
    feature_names,
    sample_sha256,
    observed_at_utc,
    source,
    family,
    publisher,
) -> dict[str, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("y length does not match X")

    n = X.shape[0]
    fields = {
        "sample_sha256": list(sample_sha256),
        "observed_at_utc": list(observed_at_utc),
        "source": list(source),
        "family": list(family),
        "publisher": list(publisher),
    }
    for name, values in fields.items():
        if len(values) != n:
            raise ValueError(f"{name} length {len(values)} != {n}")

    return {
        "X": X,
        "y": y,
        "schema_version": np.asarray(int(schema_version), dtype=np.int64),
        "dataset_contract_version": np.asarray(DATASET_CONTRACT_VERSION, dtype=np.int64),
        "feature_names": _text_array(feature_names),
        "sample_sha256": _text_array(fields["sample_sha256"]),
        "observed_at_utc": _text_array(fields["observed_at_utc"]),
        "source": _text_array(fields["source"]),
        "family": _text_array(fields["family"]),
        "publisher": _text_array(fields["publisher"]),
    }


def payload_to_bytes(payload: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez_compressed(stream, **payload)
    return stream.getvalue()


def payload_to_base64(payload: dict[str, np.ndarray]) -> str:
    return base64.b64encode(payload_to_bytes(payload)).decode("ascii")


def save_payload(path: str, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)
