"""Специализированные исключения приложения FATT."""

from __future__ import annotations


class FattError(RuntimeError):
    """Описывает ожидаемую ошибку, которую можно показать пользователю."""
