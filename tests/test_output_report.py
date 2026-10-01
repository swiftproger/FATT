"""Проверки моделей данных, записи расшифровки и итогового отчёта."""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fatt_core.errors import FattError
from fatt_core.models import (
    ProcessingJob,
    ProcessingReport,
    SpeakerSegment,
    TranscriptSegment,
)
from fatt_core.output import format_timestamp, write_transcript
from fatt_core.report import format_elapsed, print_report


class TestModels(unittest.TestCase):
    """Проверяет неизменяемые модели данных проекта."""

    def test_models_store_values_and_are_frozen(self) -> None:
        """Проверяет поля dataclass и запрет случайного изменения результата."""
        speaker = SpeakerSegment(1.0, 2.0, "LOCAL_00")
        transcript = TranscriptSegment(1.0, 2.0, "Hello", "Спикер 1")
        job = ProcessingJob(Path("in.wav"), Path("out.txt"))
        report = ProcessingReport(Path("in.wav"), Path("out.txt"), 1.2, 4, True)
        self.assertEqual(speaker.label, "LOCAL_00")
        self.assertEqual(transcript.text, "Hello")
        self.assertEqual(job.output_path, Path("out.txt"))
        self.assertIsNone(report.error)
        with self.assertRaises(Exception):
            report.success = False  # type: ignore[misc]


class TestOutput(unittest.TestCase):
    """Проверяет форматирование и запись TXT/Markdown расшифровок."""

    def test_format_timestamp_clamps_negative_and_supports_hours(self) -> None:
        """Проверяет нулевую, обычную и многочасовую длительность."""
        self.assertEqual(format_timestamp(-1), "[00:00:00]")
        self.assertEqual(format_timestamp(1.9), "[00:00:01]")
        self.assertEqual(format_timestamp(3661), "[01:01:01]")

    def test_write_transcript_writes_txt_and_markdown(self) -> None:
        """Проверяет формат строк, заголовок Markdown и пустой результат."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            segments = [
                TranscriptSegment(0.2, 1.0, "Привет", "Спикер 1"),
                TranscriptSegment(61.0, 62.0, "Hello", "Спикер 2"),
            ]
            txt = root / "result.txt"
            md = root / "result.md"
            empty = root / "empty.txt"
            write_transcript(txt, segments)
            write_transcript(md, segments)
            write_transcript(empty, [])
            self.assertEqual(
                txt.read_text(encoding="utf-8"),
                "[00:00:00] Спикер 1: Привет\n[00:01:01] Спикер 2: Hello\n",
            )
            self.assertEqual(
                md.read_text(encoding="utf-8"),
                "# Расшифровка\n\n"
                "[00:00:00] Спикер 1: Привет\n"
                "[00:01:01] Спикер 2: Hello\n",
            )
            self.assertEqual(empty.read_text(encoding="utf-8"), "")

    def test_write_transcript_wraps_os_error(self) -> None:
        """Проверяет доменную ошибку при невозможности записи результата."""
        with patch.object(Path, "write_text", side_effect=OSError("read only")):
            with self.assertRaisesRegex(FattError, "Не удалось записать результат"):
                write_transcript(Path("result.txt"), [])


class TestReport(unittest.TestCase):
    """Проверяет форматирование времени и печать сводки."""

    def test_format_elapsed_supports_seconds_and_minutes(self) -> None:
        """Проверяет формат времени до и после минутной границы."""
        self.assertEqual(format_elapsed(12.345), "12.3 с")
        self.assertEqual(format_elapsed(60.5), "1 мин 0.5 с")
        self.assertEqual(format_elapsed(125.25), "2 мин 5.2 с")

    def test_print_report_contains_success_failure_and_totals(self) -> None:
        """Проверяет блоки успешного и ошибочного результата и общие итоги."""
        reports = [
            ProcessingReport(Path("in.wav"), Path("out.txt"), 1.2, 5, True),
            ProcessingReport(Path("bad.mp3"), None, 61.5, 0, False, "decode failed"),
        ]
        output = io.StringIO()
        with redirect_stdout(output):
            print_report(reports, 125.25)
        text = output.getvalue()
        self.assertIn("ОТЧЁТ О ВЫПОЛНЕНИИ", text)
        self.assertIn("Статус: УСПЕШНО", text)
        self.assertIn("Статус: ОШИБКА", text)
        self.assertIn("Результат: out.txt", text)
        self.assertIn("Причина: decode failed", text)
        self.assertIn("Файлов обработано: 2", text)
        self.assertIn("Успешно: 1", text)
        self.assertIn("С ошибками: 1", text)
        self.assertIn("Всего слов: 5", text)
        self.assertIn("Общее время: 2 мин 5.2 с", text)

    def test_print_report_handles_empty_report_list(self) -> None:
        """Проверяет сводку для пустого набора результатов."""
        output = io.StringIO()
        with redirect_stdout(output):
            print_report([], 0)
        text = output.getvalue()
        self.assertIn("Файлов обработано: 0", text)
        self.assertIn("Успешно: 0", text)
        self.assertIn("С ошибками: 0", text)
