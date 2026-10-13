import math

import numpy as np
import pytest

import release_validation as V


def test_self_test_passes(capsys):
    V.self_test()
    assert "RELEASE_VALIDATION_SELF_TEST=PASS" in capsys.readouterr().out


@pytest.mark.parametrize("n", [10, 300, 30_000])
def test_zero_failure_upper_bound_matches_closed_form(n):
    assert V.binomial_upper_95(0, n) == pytest.approx(1.0 - 0.05 ** (1.0 / n), rel=1e-9)


def test_rule_of_three_approximation():
    assert V.binomial_upper_95(0, 30_000) == pytest.approx(3.0 / 30_000, rel=0.01)


def test_bounds_bracket_point_estimate():
    for k, n in [(1, 100), (5, 1000), (50, 60)]:
        assert V.binomial_lower_95(k, n) <= k / n <= V.binomial_upper_95(k, n)


def test_temporal_split_is_strictly_chronological():
    dataset = V._synthetic_dataset()
    split = V.temporal_split(dataset)
    t = dataset.observed_at_utc
    assert t[split.train].max() < t[split.validation].min()
    assert t[split.validation].max() < t[split.test].min()
    assert not np.any(split.train & split.test)


def test_equal_timestamps_are_never_split():
    dataset = V._synthetic_dataset()
    split = V.temporal_split(dataset)
    for day in np.unique(dataset.observed_at_utc):
        mask = dataset.observed_at_utc == day
        assert len({bool(split.train[mask].all()), bool(split.validation[mask].all()), bool(split.test[mask].all())} - {False}) == 1


def test_sha_leakage_is_rejected():
    dataset = V._synthetic_dataset()
    split = V.temporal_split(dataset)
    sha = dataset.sample_sha256.copy()
    sha[np.flatnonzero(split.test)[0]] = sha[np.flatnonzero(split.train)[0]]
    tampered = V.Dataset(dataset.X, dataset.y, sha, dataset.observed_at_utc, dataset.source, dataset.family, dataset.publisher)
    with pytest.raises(ValueError):
        V.assert_no_sha_leakage(tampered, split)


def test_threshold_reaches_target_on_validation():
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99])
    threshold = V.choose_threshold(y, s, target_tpr=0.9)
    tpr = ((s >= threshold) & (y == 1)).sum() / (y == 1).sum()
    assert tpr >= 0.9
    assert math.isclose(threshold, 0.4)
