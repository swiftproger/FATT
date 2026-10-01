"""Создание индикатора прогресса для конвейера FATT."""

from __future__ import annotations

import sys
import threading
import time

from .errors import FattError


class ConsoleProgress:
    """Показывает tqdm-бар этапов и indeterminate-активность текущего этапа."""

    def __init__(self, description: str, total: int = 3) -> None:
        """Создаёт progress bar обработки одного файла."""
        try:
            from tqdm import tqdm
        except ImportError as exc:
            raise FattError(
                "tqdm не установлен. Установите зависимости из README.md."
            ) from exc

        self._bar = tqdm(
            total=total,
            desc=description,
            unit="этап",
            dynamic_ncols=True,
            file=sys.stderr,
        )
        self._total = total
        self._position = 0
        self._stage_label: str | None = None
        self._stage_started_at: float | None = None
        self._stage_floor = 0.0
        self._animation_index = 0
        self._activity_stop: threading.Event | None = None
        self._activity_thread: threading.Thread | None = None
        self._bar_lock = threading.Lock()

    def set_postfix_str(self, text: str) -> None:
        """Показывает начало нового этапа и обновляет подпись progress bar."""
        self._stop_activity()
        self._stage_label = text.strip()
        self._stage_started_at = time.perf_counter()
        self._stage_floor = self._position
        self._animation_index = 0
        with self._bar_lock:
            self._bar.write(f"  {self._stage_label} — начато")
            self._bar.set_postfix_str(f"{self._stage_label} — начато")
        self._start_activity()

    def update(self, amount: int = 1) -> None:
        """Обновляет progress bar и показывает длительность этапа."""
        self._stop_activity()
        elapsed = 0.0
        if self._stage_started_at is not None:
            elapsed = time.perf_counter() - self._stage_started_at
        label = self._stage_label or "Этап обработки"
        target = min(self._total, self._stage_floor + amount)
        with self._bar_lock:
            self._bar.set_postfix_str(
                f"{label} — готово за {_format_seconds(elapsed)}"
            )
            self._bar.update(target - self._position)
        self._position = target
        self._stage_floor = target
        self._stage_limit = target
        self._stage_label = None
        self._stage_started_at = None

    def close(self) -> None:
        """Закрывает progress bar, не оставляя незавершённый этап без статуса."""
        self._stop_activity()
        if self._stage_label is not None:
            with self._bar_lock:
                self._bar.write(f"  {self._stage_label} — остановлено")
                self._bar.set_postfix_str(f"{self._stage_label} — остановлено")
        self._bar.close()

    def _start_activity(self) -> None:
        """Запускает indeterminate-анимацию подписи текущего этапа."""
        if self._total <= 0 or self._stage_label is None:
            return
        stop_event = threading.Event()
        self._activity_stop = stop_event
        self._activity_thread = threading.Thread(
            target=self._animate_activity,
            args=(stop_event,),
            name="fatt-progress",
            daemon=True,
        )
        self._activity_thread.start()

    def _stop_activity(self) -> None:
        """Останавливает поток анимации перед сменой состояния progress bar."""
        stop_event = self._activity_stop
        activity_thread = self._activity_thread
        if stop_event is not None:
            stop_event.set()
        if activity_thread is not None:
            activity_thread.join()
        self._activity_stop = None
        self._activity_thread = None

    def _animate_activity(self, stop_event: threading.Event) -> None:
        """Обновляет spinner в progress bar, пока этап ещё выполняется."""
        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        while not stop_event.wait(0.2):
            with self._bar_lock:
                self._animation_index = (self._animation_index + 1) % len(spinner)
                label = self._stage_label or "Этап обработки"
                self._bar.set_postfix_str(
                    f"{label} — {spinner[self._animation_index]} работает"
                )


def _format_seconds(seconds: float) -> str:
    """Форматирует длительность этапа в секундах или минутах."""
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)} мин {remainder:.1f} с"


def make_progress(description: str = "FATT") -> ConsoleProgress:
    """Создаёт tqdm-индикатор трёх этапов одного файла."""
    return ConsoleProgress(description)
