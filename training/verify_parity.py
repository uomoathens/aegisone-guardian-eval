from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aegis_features as F

TOLERANCE = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csharp", required=True, help="CSV written by FeatureDump")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    checked = 0
    mismatched = 0
    missing = 0

    with open(args.csharp, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue

            path, values = row[0].replace("%2C", ","), [float(v) for v in row[1:]]

            if len(values) != F.LENGTH:
                print(f"MISMATCH  {path}: C# produced {len(values)} values, expected {F.LENGTH}")
                mismatched += 1
                continue

            mine = F.compute(path)
            if mine is None:
                print(f"UNREADABLE  {path}: Python could not open it")
                missing += 1
                continue

            checked += 1

            for index, (theirs, ours) in enumerate(zip(values, mine)):
                if abs(theirs - ours) > args.tolerance:
                    print(f"MISMATCH  {path}")
                    print(f"          feature {index} ({F.FEATURE_NAMES[index]}): "
                          f"C#={theirs:.9f}  Python={ours:.9f}  "
                          f"delta={abs(theirs - ours):.2e}")
                    mismatched += 1
                    break

    print(f"\nchecked {checked}, mismatched {mismatched}, unreadable {missing}")

    if mismatched:
        print("\nFAILED. Do not train or ship until these agree.")
        sys.exit(1)

    print("\nPASSED. The two implementations agree within tolerance.")


if __name__ == "__main__":
    main()
