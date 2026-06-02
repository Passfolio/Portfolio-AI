"""Root conftest — mock heavy native/AI modules before any app code imports them."""
from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock


def _make_mock_module(name: str) -> ModuleType:
    mod = ModuleType(name)
    mod.__spec__ = MagicMock()
    return mod


# kiwipiepy: Kiwi() tries to load model files not available in test env
_kiwi_mock = MagicMock()
_kiwi_mock.Kiwi.return_value = MagicMock()

_kiwipiepy = _make_mock_module("kiwipiepy")
_kiwipiepy.Kiwi = _kiwi_mock.Kiwi
sys.modules.setdefault("kiwipiepy", _kiwipiepy)
