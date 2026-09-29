"""Проверки оптимизации повторного использования модели Whisper."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fatt_core.cli import interactive_wizard
from fatt_core.transcription import WhisperTranscriber


class TestWhisperTranscriber(unittest.TestCase):
    """Проверяет, что модель Whisper не загружается заново для каждого файла."""

    def test_model_is_loaded_once_for_multiple_files(self) -> None:
        """Проверяет кэширование модели и передачу языка в транскрибацию."""
        load_calls: list[tuple[str, str]] = []
        transcribe_calls: list[dict[str, object]] = []

        class FakeModel:
            """Имитирует минимальный интерфейс модели Whisper."""

            def transcribe(self, path: str, **options: object) -> dict[str, list[object]]:
                """Запоминает параметры вызова и возвращает пустой результат."""
                transcribe_calls.append({"path": path, **options})
                return {"segments": []}

        def fake_load_model(model_name: str, device: str) -> FakeModel:
            """Запоминает загрузку модели и возвращает тестовую модель."""
            load_calls.append((model_name, device))
            return FakeModel()

        fake_whisper = types.SimpleNamespace(load_model=fake_load_model)
        with patch.dict(sys.modules, {"whisper": fake_whisper}):
            transcriber = WhisperTranscriber("small", "mps", "ru")
            transcriber.transcribe(Path("one.wav"))
            transcriber.transcribe(Path("two.wav"))

        self.assertEqual(load_calls, [("small", "mps")])
        self.assertEqual(len(transcribe_calls), 2)
        self.assertTrue(all(call["language"] == "ru" for call in transcribe_calls))
        self.assertTrue(all(call["fp16"] is True for call in transcribe_calls))


class TestInteractiveWizard(unittest.TestCase):
    """Проверяет стандартные ответы интерактивного мастера."""

    def test_wizard_uses_russian_defaults_and_parses_quoted_path(self) -> None:
        """Проверяет модель small, язык ru и путь с пробелами и решёткой."""
        answers = iter(
            [
                "",
                "",
                "'/Users/aleksandr/Desktop/#01 - Карты реальности - основы'",
                "/tmp/results",
            ]
        )
        with patch(
            "fatt_core.cli._prompt_text",
            side_effect=lambda _prompt: next(answers),
        ):
            source, output, model, language = interactive_wizard()

        self.assertEqual(model, "small")
        self.assertEqual(language, "ru")
        self.assertEqual(
            source.as_posix(),
            "/Users/aleksandr/Desktop/#01 - Карты реальности - основы",
        )
        self.assertEqual(output.as_posix(), "/tmp/results")
