"""Выбор устройства для локального вывода PyTorch."""

from __future__ import annotations

from .errors import FattError


def get_torch_device() -> str:
    """Возвращает MPS на Apple Silicon, если он доступен, иначе CPU."""
    try:
        import torch
    except ImportError as exc:
        raise FattError(
            "PyTorch не установлен. Установите torch, torchvision и torchaudio."
        ) from exc
    return "mps" if torch.backends.mps.is_available() else "cpu"
