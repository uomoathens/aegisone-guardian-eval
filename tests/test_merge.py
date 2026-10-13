import merge_release_datasets as M


def test_merge_self_test(capsys):
    M.self_test()
    assert "MERGE_RELEASE_SELF_TEST=PASS" in capsys.readouterr().out
