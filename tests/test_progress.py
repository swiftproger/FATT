"""Проверки progress bar, его статусов и анимации длительного этапа."""

from __future__ import annotations

import sys
import time
import types
import unittest
from unittest.mock import patch

from fatt_core.errors import FattError
from fatt_core.progress import ConsoleProgress, _format_seconds, make_progress


class FakeTqdm:
    """Минимальная тестовая замена класса tqdm."""

    instances: list["FakeTqdm"] = []

    def __init__(self, **options: object) -> None:
        """Сохраняет параметры создания и последующие действия."""
        self.options = options
        self.calls: list[tuple[object, ...]] = []
        self.n = 0.0
        self.closed = False
        type(self).instances.append(self)

    def write(self, text: str) -> None:
        """Запоминает строку, которую tqdm пишет над полосой."""
        self.calls.append(("write", text))

    def set_postfix_str(self, text: str) -> None:
        """Запоминает текущую подпись полосы."""
        self.calls.append(("postfix", text))

    def update(self, amount: float = 1.0) -> None:
        """Сдвигает тестовую позицию полосы."""
        self.n += amount
        self.calls.append(("update", amount))

    def close(self) -> None:
        """Запоминает закрытие полосы."""
        self.closed = True
        self.calls.append(("close",))


def fake_tqdm_module() -> types.ModuleType:
    """Создаёт модуль-заглушку для ленивого импорта tqdm."""
    module = types.ModuleType("tqdm")
    module.tqdm = FakeTqdm
    return module


class TestProgress(unittest.TestCase):
    """Проверяет поведение обёртки над tqdm."""

    def setUp(self) -> None:
        """Очищает список тестовых progress bar."""
        FakeTqdm.instances.clear()

    def test_format_seconds_and_missing_dependency(self) -> None:
        """Проверяет формат коротких и длинных этапов и ошибку импорта."""
        self.assertEqual(_format_seconds(12.34), "12.3 с")
        self.assertEqual(_format_seconds(65.5), "1 мин 5.5 с")
        with patch.dict(sys.modules, {"tqdm": None}):
            with self.assertRaisesRegex(FattError, "tqdm не установлен"):
                make_progress()

    def test_progress_starts_updates_animates_and_closes(self) -> None:
        """Проверяет этапы, движение полосы и фиксацию завершения."""
        with patch.dict(sys.modules, {"tqdm": fake_tqdm_module()}):
            progress = make_progress("FATT 1/1")
            progress.set_postfix_str("[3/3] Транскрибация Whisper")
            time.sleep(0.25)
            progress.update()
            progress.close()
        bar = FakeTqdm.instances[0]
        self.assertEqual(bar.options["total"], 3)
        self.assertEqual(bar.options["desc"], "FATT 1/1")
        self.assertEqual(bar.n, 1.0)
        self.assertTrue(bar.closed)
        self.assertIn(
            ("write", "  [3/3] Транскрибация Whisper — начато"),
            bar.calls,
        )
        completion_calls = [
            call
            for call in bar.calls
            if call[0] == "postfix" and "готово за" in str(call[1])
        ]
        self.assertEqual(len(completion_calls), 1)
        self.assertGreaterEqual(
            len([call for call in bar.calls if call[0] == "update"]),
            2,
        )

    def test_progress_closes_unfinished_stage_as_stopped(self) -> None:
        """Проверяет статус незавершённого этапа при ошибке или отмене."""
        with patch.dict(sys.modules, {"tqdm": fake_tqdm_module()}):
            progress = ConsoleProgress("FATT")
            progress.set_postfix_str("[2/3] Диаризация")
            progress.close()
        bar = FakeTqdm.instances[0]
        self.assertIn(("write", "  [2/3] Диаризация — остановлено"), bar.calls)
        self.assertIn(("postfix", "[2/3] Диаризация — остановлено"), bar.calls)

    def test_activity_animation_handles_both_directions_and_zero_total(self) -> None:
        """Проверяет разворот анимации у границ и нулевой progress bar."""
        class StepEvent:
            """Имитирует событие, завершающее поток после одного шага."""

            def __init__(self) -> None:
                """Создаёт счётчик ожиданий."""
                self.calls = 0

            def wait(self, _timeout: float) -> bool:
                """Возвращает False один раз, затем True."""
                self.calls += 1
                return self.calls > 1

        with patch.dict(sys.modules, {"tqdm": fake_tqdm_module()}):
            zero_progress = ConsoleProgress("FATT", total=0)
            zero_progress._start_activity()
            zero_progress.close()

            progress = ConsoleProgress("FATT")
            progress._stage_floor = 0.0
            progress._stage_limit = 0.99
            progress._position = 0.98
            progress._activity_direction = 1.0
            progress._animate_activity(StepEvent())
            self.assertEqual(progress._activity_direction, -1.0)

            progress._position = 0.01
            progress._activity_direction = -1.0
            progress._animate_activity(StepEvent())
            self.assertEqual(progress._activity_direction, 1.0)
            progress.close()
