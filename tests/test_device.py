"""Проверки выбора CPU/MPS и диагностической информации PyTorch."""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from fatt_core import device
from fatt_core.errors import FattError


def _torch_module(*, built: bool, available: bool, version: str = "test") -> types.ModuleType:
    """Создаёт минимальную тестовую замену модуля torch."""
    module = types.ModuleType("torch")
    module.__version__ = version
    module.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(
            is_built=lambda: built,
            is_available=lambda: available,
        )
    )
    return module


class TestDevice(unittest.TestCase):
    """Проверяет все ветви выбора вычислительного устройства."""

    def test_get_device_info_reports_missing_torch(self) -> None:
        """Проверяет доменную ошибку при отсутствии PyTorch."""
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(FattError, "PyTorch не установлен"):
                device.get_device_info()

    def test_get_device_info_prefers_available_mps(self) -> None:
        """Проверяет выбор Apple GPU и FP16 при доступном MPS."""
        fake_torch = _torch_module(built=True, available=True, version="2.test")
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch("fatt_core.device.platform.python_version", return_value="3.test"):
                info = device.get_device_info()
        self.assertEqual(info.name, "mps")
        self.assertEqual(info.precision, "FP16")
        self.assertTrue(info.mps_built)
        self.assertTrue(info.mps_available)
        self.assertIn("будет использован", info.mps_status)
        self.assertEqual(info.torch_version, "2.test")
        self.assertEqual(info.python_version, "3.test")

    def test_get_device_info_falls_back_when_mps_built_but_unavailable(self) -> None:
        """Проверяет диагностику собранного, но недоступного MPS."""
        fake_torch = _torch_module(built=True, available=False)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            info = device.get_device_info()
        self.assertEqual(info.name, "cpu")
        self.assertEqual(info.mps_status, "собран, но недоступен")
        self.assertIn("используется CPU", info.reason)

    def test_get_device_info_handles_missing_or_unbuilt_mps_backend(self) -> None:
        """Проверяет CPU-ветку без MPS backend и без сборки MPS."""
        for backends in (types.SimpleNamespace(), types.SimpleNamespace(mps=None)):
            fake_torch = types.ModuleType("torch")
            fake_torch.__version__ = "test"
            fake_torch.backends = backends
            with patch.dict(sys.modules, {"torch": fake_torch}):
                info = device.get_device_info()
            self.assertEqual(info.name, "cpu")
            self.assertEqual(info.mps_status, "не собран")
            self.assertFalse(info.mps_built)
            self.assertFalse(info.mps_available)

    def test_get_torch_device_delegates_to_device_info(self) -> None:
        """Проверяет короткий публичный помощник выбора устройства."""
        fake_info = types.SimpleNamespace(name="mps")
        with patch("fatt_core.device.get_device_info", return_value=fake_info) as getter:
            self.assertEqual(device.get_torch_device(), "mps")
        getter.assert_called_once_with()
