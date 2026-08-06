from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_wrapper():
    path = ROOT / "scripts" / "kaggle_ctc_collapse_diagnostics.py"
    spec = importlib.util.spec_from_file_location("kaggle_ctc_wrapper_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_fallback_does_not_recurse(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    wrapper = _load_wrapper()
    expected = {"sample_id": "fallback-row"}
    monkeypatch.setattr(wrapper, "BASE_ITER_PAIRS", lambda root, split: iter([expected]))

    assert wrapper.BASE_ITER_PAIRS is not wrapper.iter_pairs
    assert list(wrapper.iter_pairs(tmp_path, "test")) == [expected]
