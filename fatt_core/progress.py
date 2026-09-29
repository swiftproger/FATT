"""Создание индикатора прогресса для конвейера FATT."""

from __future__ import annotations

from typing import Any

from .errors import FattError


def make_progress(description: str = "FATT") -> Any:
    """Создаёт индикатор tqdm для трёх этапов одного обрабатываемого файла."""
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise FattError(
            "tqdm не установлен. Установите зависимости из README.md."
        ) from exc
    return tqdm(total=3, desc=description, unit="этап", dynamic_ncols=True)
