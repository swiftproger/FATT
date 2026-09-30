"""Создание индикатора прогресса для конвейера FATT."""

from __future__ import annotations

import sys
import time


class ConsoleProgress:
    """Печатает этапы обработки отдельными строками без конфликтующих перерисовок."""

    def __init__(self, description: str, total: int = 3) -> None:
        """Создаёт последовательный индикатор обработки одного файла."""
        del description
        self._total = total
        self._current = 0
        self._stage_label: str | None = None
        self._stage_started_at: float | None = None

    def set_postfix_str(self, text: str) -> None:
        """Печатает начало нового этапа обработки."""
        self._stage_label = text.strip()
        self._stage_started_at = time.perf_counter()
        print(f"  {self._stage_label} — начато", file=sys.stderr, flush=True)

    def update(self, amount: int = 1) -> None:
        """Печатает завершение текущего этапа и обновляет его номер."""
        elapsed = 0.0
        if self._stage_started_at is not None:
            elapsed = time.perf_counter() - self._stage_started_at
        label = self._stage_label or "Этап обработки"
        self._current = min(self._total, self._current + amount)
        print(
            f"  {label} — готово за {_format_seconds(elapsed)} "
            f"({self._current}/{self._total})",
            file=sys.stderr,
            flush=True,
        )
        self._stage_label = None
        self._stage_started_at = None

    def close(self) -> None:
        """Завершает индикатор без управляющих символов в терминале."""
        if self._stage_label is not None:
            print(
                f"  {self._stage_label} — остановлено",
                file=sys.stderr,
                flush=True,
            )


def _format_seconds(seconds: float) -> str:
    """Форматирует длительность этапа в секундах или минутах."""
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)} мин {remainder:.1f} с"


def make_progress(description: str = "FATT") -> ConsoleProgress:
    """Создаёт читаемый текстовый индикатор трёх этапов одного файла."""
    return ConsoleProgress(description)
