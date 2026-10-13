import json
import subprocess
import sys
import os

import pytest

pytest.importorskip("lightgbm")

import release_experiment as E

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def report():
    dataset = E.synthetic_drift_dataset(days=60, per_day=40)
    result, _ = E.run(dataset, seed=1, train_fraction=0.7, validation_fraction=0.15, label="test")
    return result


def test_split_has_no_leakage(report):
    assert report["split"]["method"] == "temporal"
    assert report["split"]["sha256_cross_split_leakage"] == 0
    assert report["split"]["train"] > report["split"]["test"] > 0


def test_operating_points_are_bounded(report):
    for op in report["operating_points"]:
        t = op["test"]
        assert 0.0 <= t["fpr"] <= t["fpr_upper_95"] <= 1.0
        assert 0.0 <= t["tpr_lower_95"] <= t["tpr"] <= 1.0


def test_calibration_table_covers_test_set(report):
    assert sum(b["count"] for b in report["reliability"]) == report["split"]["test"]
    assert 0.0 <= report["test_ece"] <= 1.0


def test_cli_writes_reports(tmp_path):
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "training", "release_experiment.py"), "--synthetic", "--out", str(tmp_path)],
        check=True,
        capture_output=True,
    )
    data = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert "operating_points" in data
    assert (tmp_path / "report.md").read_text(encoding="utf-8").startswith("# Results")
