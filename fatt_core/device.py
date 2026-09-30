"""Выбор устройства для локального вывода PyTorch."""

from __future__ import annotations

import platform
from dataclasses import dataclass

from .errors import FattError


@dataclass(frozen=True)
class DeviceInfo:
    """Хранит понятное описание вычислительного устройства и его состояния."""

    name: str
    label: str
    precision: str
    torch_version: str
    python_version: str
    mps_built: bool
    mps_available: bool
    mps_status: str
    reason: str


def get_device_info() -> DeviceInfo:
    """Определяет устройство и собирает диагностические сведения для консоли."""
    try:
        import torch
    except ImportError as exc:
        raise FattError(
            "PyTorch не установлен. Установите torch, torchvision и torchaudio."
        ) from exc

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend and mps_backend.is_built())
    mps_available = bool(mps_backend and mps_backend.is_available())
    if mps_available:
        return DeviceInfo(
            name="mps",
            label="Apple GPU через MPS/Metal",
            precision="FP16",
            torch_version=str(torch.__version__),
            python_version=platform.python_version(),
            mps_built=mps_built,
            mps_available=mps_available,
            mps_status="доступен и будет использован",
            reason="PyTorch видит GPU Apple через Metal",
        )

    if mps_built:
        reason = (
            "MPS собран в PyTorch, но недоступен в текущем окружении; используется CPU"
        )
        mps_status = "собран, но недоступен"
    else:
        reason = "PyTorch установлен без доступного MPS; используется CPU"
        mps_status = "не собран"
    return DeviceInfo(
        name="cpu",
        label="Процессор (CPU)",
        precision="FP32",
        torch_version=str(torch.__version__),
        python_version=platform.python_version(),
        mps_built=mps_built,
        mps_available=mps_available,
        mps_status=mps_status,
        reason=reason,
    )


def get_torch_device() -> str:
    """Возвращает MPS на Apple Silicon, если он доступен, иначе CPU."""
    return get_device_info().name
