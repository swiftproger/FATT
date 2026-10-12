"""Проверки цветов в терминале и чистого вывода в логах."""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from fatt_core.console import color
from fatt_core.models import ProcessingReport
from fatt_core.report import print_report


class TerminalBuffer(io.StringIO):
    """Текстовый буфер, имитирующий терминал."""

    def isatty(self) -> bool:
        """Сообщает, что поток поддерживает терминальный вывод."""
        return True


class TestConsole(unittest.TestCase):
    """Проверяет включение цветов и оформление результатов."""

    def test_color_respects_stream_and_environment(self) -> None:
        """Цвета включаются в TTY и отключаются для логов и NO_COLOR."""
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(color("готово", "success", TerminalBuffer()), "\033[32mготово\033[0m")
            self.assertEqual(color("готово", "success", io.StringIO()), "готово")
            with patch.dict("os.environ", {"NO_COLOR": ""}):
                self.assertEqual(color("готово", "success", TerminalBuffer()), "готово")
            with patch.dict("os.environ", {"TERM": "dumb"}):
                self.assertEqual(color("готово", "success", TerminalBuffer()), "готово")

    def test_report_colors_success_and_failure_only_in_terminal(self) -> None:
        """Отчёт различает статусы цветом, сохраняя чистый текст в логах."""
        reports = [
            ProcessingReport(Path("good.wav"), Path("good.txt"), 1, 3, True),
            ProcessingReport(Path("bad.wav"), None, 1, 0, False, "decode failed"),
        ]
        with patch.dict("os.environ", {}, clear=True):
            terminal = TerminalBuffer()
            with redirect_stdout(terminal):
                print_report(reports, 2)
            self.assertIn("\033[32mСтатус: УСПЕШНО\033[0m", terminal.getvalue())
            self.assertIn("\033[1;31mСтатус: ОШИБКА\033[0m", terminal.getvalue())
            log = io.StringIO()
            with redirect_stdout(log):
                print_report(reports, 2)
            self.assertNotIn("\033[", log.getvalue())
