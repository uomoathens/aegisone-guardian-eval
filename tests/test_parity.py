import os
import shutil
import subprocess
import sys

import pytest

import make_corpus

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(shutil.which("dotnet") is None, reason="dotnet SDK not installed")
def test_csharp_and_python_vectors_agree(tmp_path):
    corpus = tmp_path / "corpus"
    make_corpus.main(str(corpus))
    output = tmp_path / "csharp_vectors.csv"
    subprocess.run(
        ["dotnet", "run", "--project", os.path.join(ROOT, "tools", "FeatureDump"), "-c", "Release", "--", str(corpus), str(output)],
        check=True,
        capture_output=True,
    )
    result = subprocess.run(
        [sys.executable, os.path.join(ROOT, "training", "verify_parity.py"), "--csharp", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout
    assert "mismatched 0" in result.stdout
