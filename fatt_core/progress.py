"""Создание индикатора прогресса для конвейера FATT."""

from __future__ import annotations

from typing import Any

from .errors import FattError


def make_progress() -> Any:
    """Создаёт индикатор tqdm для трёх этапов командной утилиты."""
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise FattError(
            "tqdm не установлен. Установите зависимости из README.md."
        ) from exc
    return tqdm(total=3, desc="FATT", unit="этап", dynamic_ncols=True)
