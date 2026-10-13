import os

import pytest

import aegis_features as F
import make_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("corpus")
    make_corpus.main(str(root))
    return root


def vec(corpus, name):
    v = F.compute(os.path.join(corpus, name))
    assert v is not None
    return v


def test_schema_shape():
    assert F.LENGTH == 48
    assert len(F.FEATURE_NAMES) == F.LENGTH
    assert len(set(F.FEATURE_NAMES)) == F.LENGTH


def test_every_value_in_unit_interval(corpus):
    for name in sorted(os.listdir(corpus)):
        v = vec(corpus, name)
        assert len(v) == F.LENGTH
        assert all(0.0 <= x <= 1.0 for x in v), name


def test_missing_file_returns_none(tmp_path):
    assert F.compute(str(tmp_path / "absent.bin")) is None


def test_empty_file_is_all_zero_except_gap(corpus):
    v = vec(corpus, "empty.bin")
    assert v[F.IDX_ENTROPY_GAP] == 0.5
    assert sum(v) == 0.5


def test_basic_pe_header_fields(corpus):
    v = vec(corpus, "pe_basic.exe")
    assert v[F.IDX_IS_PE] == 1.0
    assert v[F.IDX_SECTION_COUNT] == 2 / 16
    assert v[F.IDX_IS_64BIT] == 1.0
    assert v[F.IDX_IS_GUI_SUBSYSTEM] == 1.0
    assert v[F.IDX_IS_DLL] == 0.0
    assert v[F.IDX_TIMESTAMP_BAD] == 0.0


def test_dll_flags(corpus):
    v = vec(corpus, "pe_dll32.dll")
    assert v[F.IDX_IS_DLL] == 1.0
    assert v[F.IDX_IS_64BIT] == 0.0
    assert v[F.IDX_RELOCS_STRIPPED] == 1.0
    assert v[F.IDX_IS_GUI_SUBSYSTEM] == 0.0


@pytest.mark.parametrize("name,expected", [
    ("pe_ts_zero.exe", 1.0),
    ("pe_ts_1990.exe", 1.0),
    ("pe_ts_future_1_5d.exe", 0.0),
    ("pe_ts_future_3d.exe", 1.0),
    ("pe_ts_max.exe", 1.0),
])
def test_timestamp_rule(corpus, name, expected):
    assert vec(corpus, name)[F.IDX_TIMESTAMP_BAD] == expected


@pytest.mark.parametrize("name", [
    "pe_negative_lfanew.exe",
    "pe_lfanew_beyond.exe",
    "pe_truncated_coff.exe",
    "pe_bad_signature.exe",
    "mz_tiny.bin",
])
def test_malformed_headers_are_not_pe(corpus, name):
    assert vec(corpus, name)[F.IDX_IS_PE] == 0.0


def test_overlay_is_measured(corpus):
    assert vec(corpus, "pe_overlay.exe")[F.IDX_OVERLAY_RATIO] > 0.9
    assert vec(corpus, "pe_basic.exe")[F.IDX_OVERLAY_RATIO] == 0.0


def test_section_beyond_eof_is_ignored_for_overlay(corpus):
    assert vec(corpus, "pe_section_beyond_eof.exe")[F.IDX_OVERLAY_RATIO] == 0.0


def test_tail_window_sees_appended_payload(corpus):
    v = vec(corpus, "stub_plus_payload_600k.bin")
    assert v[F.IDX_HEADER_ENTROPY] == 0.0
    assert v[F.IDX_TAIL_ENTROPY] > 0.8
    assert v[F.IDX_ENTROPY_GAP] > 0.9


def test_small_optional_header_skips_subsystem(corpus):
    assert vec(corpus, "pe_small_optional.exe")[F.IDX_IS_GUI_SUBSYSTEM] == 0.0
