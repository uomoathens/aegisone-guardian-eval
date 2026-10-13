from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aegis_features as F
import dataset_contract as D

API = "https://mb-api.abuse.ch/api/v1/"
ZIP_PASSWORD = b"infected"
WANTED_TYPES = {"exe", "dll", "sys", "scr"}


def require(module: str):
    try:
        return __import__(module)
    except ImportError:
        sys.exit(f"Missing VM-only dependency: {module}")


def normalise_first_seen(value) -> str:
    if value:
        text = str(value).strip()
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
                return dt.isoformat().replace("+00:00", "Z")
            except ValueError:
                pass

        if "T" in text:
            return text.replace("+00:00", "Z")

    return D.utc_now_iso()


def recent_hashes(session, key: str, limit: int, selection: str) -> list[dict]:
    payload = {"query": "get_recent", "selector": selection}

    response = session.post(API, data=payload, headers={"Auth-Key": key}, timeout=60)
    response.raise_for_status()
    body = response.json()

    if body.get("query_status") != "ok":
        sys.exit(f"MalwareBazaar said: {body.get('query_status')}")

    samples = []
    for entry in body.get("data", []):
        if entry.get("file_type", "").lower() not in WANTED_TYPES:
            continue

        samples.append({
            "sha256": str(entry["sha256_hash"]).lower(),
            "family": entry.get("signature") or "unknown",
            "type": entry.get("file_type"),
            "observed_at_utc": normalise_first_seen(entry.get("first_seen")),
        })

        if len(samples) >= limit:
            break

    return samples


def fetch_and_featurise(session, key: str, sha256: str, workdir: str):
    pyzipper = require("pyzipper")

    response = session.post(
        API,
        data={"query": "get_file", "sha256_hash": sha256},
        headers={"Auth-Key": key},
        timeout=120,
    )

    if response.status_code != 200 or response.content[:2] != b"PK":
        return None

    path = os.path.join(workdir, sha256)

    try:
        with pyzipper.AESZipFile(io.BytesIO(response.content)) as archive:
            archive.setpassword(ZIP_PASSWORD)
            names = archive.namelist()
            if not names:
                return None

            with open(path, "wb") as handle:
                handle.write(archive.read(names[0]))

        return F.compute(path)

    except Exception:
        return None

    finally:
        try:
            if os.path.exists(path):
                with open(path, "wb") as handle:
                    handle.write(b"\x00" * 4096)
                os.remove(path)
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect malicious Guardian release-contract v2 vectors."
    )
    parser.add_argument("--key", required=True, help="Auth-Key from auth.abuse.ch")
    parser.add_argument("--out", default="malicious.npz")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--selection", default="time")
    parser.add_argument("--workdir", default=r"C:\malware-work")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--source",
        default="MalwareBazaar",
        help="Stable provenance label for this malware feed.",
    )
    args = parser.parse_args()

    requests = require("requests")
    require("pyzipper")

    print("=" * 68)
    print("THIS DOWNLOADS LIVE MALWARE")
    print("=" * 68)
    print()
    print("RUN ONLY IN A DISPOSABLE, ISOLATED VM.")
    print("Never run this collector on the AegisOne host/laptop.")
    print()

    if not args.yes:
        answer = input("Type: I am in a VM  ->  ")
        if answer.strip().lower() != "i am in a vm":
            sys.exit("Stopped.")

    os.makedirs(args.workdir, exist_ok=True)
    session = requests.Session()

    print("Asking MalwareBazaar what is recent...")
    samples = recent_hashes(session, args.key, args.limit, args.selection)
    print(f"  {len(samples)} Windows executables to fetch")
    print()

    rows: list[list[float]] = []
    sample_sha256: list[str] = []
    observed_at_utc: list[str] = []
    sources: list[str] = []
    families_out: list[str] = []
    publishers: list[str] = []

    family_counts: dict[str, int] = {}
    failed = 0
    started = time.time()

    try:
        for index, sample in enumerate(samples, 1):
            vector = fetch_and_featurise(
                session, args.key, sample["sha256"], args.workdir
            )

            if vector is None:
                failed += 1
            else:
                rows.append(vector)
                sample_sha256.append(sample["sha256"])
                observed_at_utc.append(sample["observed_at_utc"])
                sources.append(args.source)
                families_out.append(sample["family"])
                publishers.append("")
                family_counts[sample["family"]] = (
                    family_counts.get(sample["family"], 0) + 1
                )

            if index % 25 == 0:
                rate = index / max(1e-6, time.time() - started)
                print(
                    f"  {index}/{len(samples)}  kept {len(rows)}  ({rate:.1f}/s)"
                )

            time.sleep(0.4)

    except KeyboardInterrupt:
        print("\nStopped early. Writing what was collected.")

    finally:
        try:
            shutil.rmtree(args.workdir, ignore_errors=True)
        except Exception:
            pass

    if not rows:
        sys.exit("Nothing was collected. Check the key and the VM network.")

    X = np.asarray(rows, dtype=np.float32)
    y = np.ones(len(rows), dtype=np.int64)

    payload = D.build_payload(
        X=X,
        y=y,
        schema_version=F.VERSION,
        feature_names=F.FEATURE_NAMES,
        sample_sha256=sample_sha256,
        observed_at_utc=observed_at_utc,
        source=sources,
        family=families_out,
        publisher=publishers,
    )
    D.save_payload(args.out, payload)

    print()
    print(f"Wrote {args.out}")
    print(f"  contract v{D.DATASET_CONTRACT_VERSION}")
    print(f"  {X.shape[0]:,} samples, {failed} could not be read")
    print("Most common families:")

    for family, count in sorted(
        family_counts.items(), key=lambda kv: -kv[1]
    )[:12]:
        print(f"    {count:5}  {family}")

    print()
    print("Copy ONLY the .npz out of the VM.")
    print("Leave all executable material inside the VM and delete the VM.")

    if len(family_counts) < 15:
        print()
        print(
            f"WARNING: only {len(family_counts)} families. "
            "Collect on multiple later dates before release validation."
        )


if __name__ == "__main__":
    main()
