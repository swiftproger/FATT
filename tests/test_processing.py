"""Проверки оптимизации повторного использования модели Whisper."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

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

