import json
import os

import numpy as np
import pytest

import ember2024 as M


def test_threshold_respects_target_fpr():
    rng = np.random.default_rng(0)
    benign = rng.random(10_000)
    for target in (0.1, 0.01, 0.001):
        theta = M.threshold_at_fpr(benign, target)
        assert (benign >= theta).mean() <= target
        k = int(np.floor(target * benign.size))
        kth = np.sort(benign)[::-1][k]
        assert (benign >= kth).sum() == k + 1


def test_threshold_with_ties_never_exceeds_target():
    benign = np.array([0.9] * 50 + [0.1] * 950)
    theta = M.threshold_at_fpr(benign, 0.01)
    assert (benign >= theta).mean() <= 0.01


def test_threshold_zero_target_excludes_every_benign():
    benign = np.array([0.2, 0.5, 0.99])
    theta = M.threshold_at_fpr(benign, 0.0)
    assert (benign >= theta).sum() == 0


@pytest.fixture(scope="module")
def fake_report(tmp_path_factory):
    pytest.importorskip("thrember")
    import make_fake_ember

    data = tmp_path_factory.mktemp("ember")
    make_fake_ember.build(str(data), rows_per_week=6)
    M.vectorize(str(data))
    return M.run(str(data), "Dot_Net", validation_weeks=8, seed=0, quick=True, random_baseline=True)


def test_file_type_and_label_filtering(fake_report):
    audit = fake_report["audit"]
    assert audit["train"]["rows_other_file_type"] == 52
    assert audit["test"]["rows_other_file_type"] == 12
    assert audit["train"]["rows_unlabelled"] == 52
    assert audit["challenge"]["rows_other_file_type"] == 1
    assert fake_report["split"]["challenge_malicious"] == 15


def test_partitions_are_chronological(fake_report):
    s = fake_report["split"]
    assert s["fit"]["last"] < s["validation"]["first"] <= s["validation"]["last"] < s["test"]["first"]


def test_thresholds_hold_where_chosen(fake_report):
    for op in fake_report["operating_points"]:
        rows = {r["protocol"].split(":")[0].split(" ")[0]: r for r in op["rows"]}
        assert rows["temporal"]["validation"]["fpr"] <= op["target_fpr"]
        assert rows["oracle"]["test"]["fpr"] <= op["target_fpr"]
        assert rows["random"]["validation"]["fpr"] <= op["target_fpr"]
        assert len(op["weekly_temporal"]) == 12


def test_report_serialises(fake_report, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(fake_report, default=float), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["label"] == "EMBER2024 Dot_Net"
    assert M.to_markdown(fake_report).startswith("# Results: EMBER2024 Dot_Net")


def fake_downloader(nested):
    import types

    def download_dataset(path, split="all", file_type="all"):
        os.chdir(path)
        folder = os.path.join(path, "inner") if nested else path
        os.makedirs(folder, exist_ok=True)
        for name in (f"{file_type}_train.jsonl", f"{file_type}_test.jsonl", "challenge.jsonl"):
            with open(os.path.join(folder, name), "w", encoding="utf-8") as handle:
                handle.write("{}\n")

    return types.SimpleNamespace(download_dataset=download_dataset)


@pytest.mark.parametrize("nested", [False, True])
def test_download_flattens_and_restores_cwd(tmp_path, monkeypatch, nested):
    monkeypatch.setitem(__import__("sys").modules, "thrember", fake_downloader(nested))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    before = os.getcwd()
    M.download(str(tmp_path / "data"), "Dot_Net")
    assert os.getcwd() == before
    names = sorted(n for n in os.listdir(tmp_path / "data") if n.endswith(".jsonl"))
    assert names == ["Dot_Net_test.jsonl", "Dot_Net_train.jsonl", "challenge.jsonl"]


def test_download_fails_loudly_when_a_subset_is_missing(tmp_path, monkeypatch):
    import types

    monkeypatch.setitem(__import__("sys").modules, "thrember", types.SimpleNamespace(download_dataset=lambda path, split, file_type: None))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    with pytest.raises(FileNotFoundError):
        M.download(str(tmp_path / "data"), "Dot_Net")
