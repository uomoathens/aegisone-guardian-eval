from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aegis_features as F
import dataset_contract as D

EXTENSIONS = {".exe", ".dll", ".sys", ".ocx", ".cpl", ".scr", ".drv", ".efi"}

SKIP = (
    "\\windows\\winsxs\\",
    "\\windows\\servicing\\",
    "\\windows\\softwaredistribution\\",
    "\\$recycle.bin\\",
    "\\system volume information\\",
)


def looks_interesting(path: str) -> bool:
    lowered = path.lower()
    if any(marker in lowered for marker in SKIP):
        return False
    return os.path.splitext(lowered)[1] in EXTENSIONS


def windows_company_name(path: str) -> str:
    if os.name != "nt":
        return "unknown"

    try:
        version = ctypes.WinDLL("version", use_last_error=True)

        version.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
        version.GetFileVersionInfoSizeW.restype = wintypes.DWORD
        version.GetFileVersionInfoW.argtypes = [
            wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID
        ]
        version.GetFileVersionInfoW.restype = wintypes.BOOL
        version.VerQueryValueW.argtypes = [
            wintypes.LPCVOID,
            wintypes.LPCWSTR,
            ctypes.POINTER(wintypes.LPVOID),
            ctypes.POINTER(wintypes.UINT),
        ]
        version.VerQueryValueW.restype = wintypes.BOOL

        dummy = wintypes.DWORD(0)
        size = version.GetFileVersionInfoSizeW(path, ctypes.byref(dummy))
        if not size:
            return "unknown"

        buffer = ctypes.create_string_buffer(size)
        if not version.GetFileVersionInfoW(path, 0, size, buffer):
            return "unknown"

        ptr = wintypes.LPVOID()
        length = wintypes.UINT(0)

        translations: list[tuple[int, int]] = []
        if version.VerQueryValueW(
            buffer,
            r"\VarFileInfo\Translation",
            ctypes.byref(ptr),
            ctypes.byref(length),
        ) and length.value >= 4:
            count = length.value // 4
            pairs = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ushort))
            for i in range(count):
                translations.append((pairs[i * 2], pairs[i * 2 + 1]))

        if not translations:
            translations = [(0x0409, 0x04B0), (0x0409, 0x04E4)]

        for language, codepage in translations:
            query = (
                rf"\StringFileInfo\{language:04x}{codepage:04x}\CompanyName"
            )
            value_ptr = wintypes.LPVOID()
            value_len = wintypes.UINT(0)
            if version.VerQueryValueW(
                buffer,
                query,
                ctypes.byref(value_ptr),
                ctypes.byref(value_len),
            ) and value_len.value > 1:
                value = ctypes.wstring_at(value_ptr, value_len.value - 1).strip()
                if value:
                    return value[:256]

    except Exception:
        pass

    return "unknown"


def collect(
    roots: list[str],
    limit: int,
    quiet: bool,
    *,
    source_label: str,
    status_stream,
):
    rows: list[list[float]] = []
    sample_sha256: list[str] = []
    observed_at_utc: list[str] = []
    sources: list[str] = []
    families: list[str] = []
    publishers: list[str] = []

    hashes: set[str] = set()
    seen = 0
    skipped = 0
    started = time.time()

    run_observed_at = D.utc_now_iso()

    for root in roots:
        if not os.path.isdir(root):
            print(f"  skipping {root}: not a directory", file=status_stream)
            continue

        print(f"  walking {root}", file=status_stream)

        for directory, _, names in os.walk(root):
            for name in names:
                path = os.path.join(directory, name)

                if not looks_interesting(path):
                    continue

                seen += 1

                try:
                    digest = D.sha256_file(path)
                except OSError:
                    skipped += 1
                    continue

                if digest in hashes:
                    continue

                vector = F.compute(path)
                if vector is None:
                    skipped += 1
                    continue

                hashes.add(digest)
                rows.append(vector)
                sample_sha256.append(digest)
                observed_at_utc.append(run_observed_at)
                sources.append(source_label)
                families.append("")
                publishers.append(windows_company_name(path))

                if not quiet and len(rows) % 500 == 0:
                    rate = len(rows) / max(1e-6, time.time() - started)
                    print(
                        f"    {len(rows):,} collected  ({rate:.0f}/s)",
                        file=status_stream,
                    )

                if len(rows) >= limit:
                    print(
                        f"    reached the limit of {limit:,}",
                        file=status_stream,
                    )
                    return (
                        rows, sample_sha256, observed_at_utc,
                        sources, families, publishers, seen, skipped
                    )

    return (
        rows, sample_sha256, observed_at_utc,
        sources, families, publishers, seen, skipped
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect benign Guardian release-contract v2 feature vectors."
    )
    parser.add_argument("--out", default="benign.npz")
    parser.add_argument("--roots", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=200_000)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--source",
        default="trusted-windows",
        help="Stable provenance label for this clean collection source.",
    )
    parser.add_argument(
        "--emit-base64",
        action="store_true",
        help=(
            "Do not write a file. Serialize NPZ in memory and emit base64 to "
            "stdout so a CFA-compatible PowerShell wrapper can write it."
        ),
    )
    args = parser.parse_args()

    roots = args.roots or [
        os.environ.get("SystemRoot", r"C:\Windows"),
        os.environ.get("ProgramFiles", r"C:\Program Files"),
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    ]

    status_stream = sys.stderr if args.emit_base64 else sys.stdout

    print("Collecting benign samples", file=status_stream)
    print(
        f"  feature schema v{F.VERSION}, {F.LENGTH} features",
        file=status_stream,
    )
    print(
        f"  dataset contract v{D.DATASET_CONTRACT_VERSION}",
        file=status_stream,
    )
    print(file=status_stream)

    (
        rows, sample_sha256, observed_at_utc,
        sources, families, publishers, seen, skipped
    ) = collect(
        roots,
        args.limit,
        args.quiet,
        source_label=args.source,
        status_stream=status_stream,
    )

    if not rows:
        sys.exit("No executables were read. Are the paths right, and can you read them?")

    X = np.asarray(rows, dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int64)

    payload = D.build_payload(
        X=X,
        y=y,
        schema_version=F.VERSION,
        feature_names=F.FEATURE_NAMES,
        sample_sha256=sample_sha256,
        observed_at_utc=observed_at_utc,
        source=sources,
        family=families,
        publisher=publishers,
    )

    if args.emit_base64:
        print(D.payload_to_base64(payload))
    else:
        D.save_payload(args.out, payload)
        print(file=status_stream)
        print(f"Wrote {args.out}", file=status_stream)

    known_publishers = sum(
        1 for publisher in publishers
        if publisher.strip().lower() not in {"", "unknown"}
    )

    print(
        f"  {X.shape[0]:,} unique executables",
        file=status_stream,
    )
    print(
        f"  {seen:,} examined, {skipped:,} unreadable",
        file=status_stream,
    )
    print(
        f"  publisher labels present: {known_publishers:,}/{X.shape[0]:,}",
        file=status_stream,
    )

    if X.shape[0] < 5_000:
        print(
            "WARNING: under 5,000 samples. This is not enough for a shipping model.",
            file=status_stream,
        )


if __name__ == "__main__":
    main()
