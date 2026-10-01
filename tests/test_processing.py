"""Проверки оптимизации повторного использования модели Whisper."""

from __future__ import annotations

import sys
import types
import unittest
import warnings
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from fatt_core.cli import interactive_wizard
from fatt_core.errors import FattError
from fatt_core.models import SpeakerSegment
from fatt_core.transcription import (
    WhisperTranscriber,
    _overlap,
    _print_whisper_warnings,
    _speaker_for_interval,
    combine_transcript_and_speakers,
    transcribe_audio,
)


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
        self.assertTrue(all(call["verbose"] is None for call in transcribe_calls))

    def test_cpu_transcription_omits_language_when_not_set(self) -> None:
        """Проверяет FP32 и отсутствие пустого параметра language для CPU."""
        calls: list[dict[str, object]] = []

        class FakeModel:
            """Имитирует модель, возвращающую один сегмент."""

            def transcribe(self, path: str, **options: object) -> dict[str, object]:
                """Запоминает путь и параметры вызова."""
                calls.append({"path": path, **options})
                return {"segments": [{"text": "hello"}]}

        fake_whisper = types.SimpleNamespace(load_model=lambda *_args, **_kwargs: FakeModel())
        with patch.dict(sys.modules, {"whisper": fake_whisper}):
            result = WhisperTranscriber("tiny", "cpu").transcribe(Path("audio.wav"))
        self.assertEqual(result, [{"text": "hello"}])
        self.assertEqual(calls[0]["fp16"], False)
        self.assertNotIn("language", calls[0])

    def test_transcriber_reports_missing_dependency_load_and_runtime_errors(self) -> None:
        """Проверяет ошибки импорта модели, загрузки и самого распознавания."""
        with patch.dict(sys.modules, {"whisper": None}):
            with self.assertRaisesRegex(FattError, "openai-whisper"):
                WhisperTranscriber("small", "cpu")

        def raise_load(*_args: object, **_kwargs: object) -> None:
            """Имитирует сбой загрузки модели."""
            raise RuntimeError("download failed")

        with patch.dict(sys.modules, {"whisper": types.SimpleNamespace(load_model=raise_load)}):
            with self.assertRaisesRegex(FattError, "Не удалось загрузить модель Whisper"):
                WhisperTranscriber("small", "cpu")

        class BrokenModel:
            """Имитирует ошибку во время распознавания."""

            def transcribe(self, *_args: object, **_kwargs: object) -> object:
                """Выбрасывает ошибку модели."""
                raise RuntimeError("decode failed")

        fake_whisper = types.SimpleNamespace(load_model=lambda *_args, **_kwargs: BrokenModel())
        with patch.dict(sys.modules, {"whisper": fake_whisper}):
            transcriber = WhisperTranscriber("small", "cpu")
            with self.assertRaisesRegex(FattError, "Ошибка транскрибации Whisper"):
                transcriber.transcribe(Path("audio.wav"))

    def test_whisper_warnings_are_simplified(self) -> None:
        """Проверяет специальное сообщение для повреждённого кэша модели."""
        captured: list[warnings.WarningMessage]
        with warnings.catch_warnings(record=True) as caught:
            warnings.warn("checksum does not match", UserWarning)
            warnings.warn("ordinary warning", UserWarning)
            captured = list(caught)
        stream = StringIO()
        with redirect_stderr(stream):
            _print_whisper_warnings(captured)
        text = stream.getvalue()
        self.assertIn("Кэш модели повреждён", text)
        self.assertIn("ordinary warning", text)

    def test_transcribe_audio_builds_transcriber(self) -> None:
        """Проверяет тонкую функцию транскрибации через новый экземпляр."""
        fake = types.SimpleNamespace(transcribe=lambda path: [{"path": str(path)}])
        with patch("fatt_core.transcription.WhisperTranscriber", return_value=fake) as factory:
            result = transcribe_audio(Path("audio.wav"), "base", "cpu", "en")
        self.assertEqual(result, [{"path": "audio.wav"}])
        factory.assert_called_once_with("base", "cpu", "en")

    def test_transcription_does_not_emit_legacy_heartbeat(self) -> None:
        """Проверяет, что длительная работа Whisper не печатает отдельный heartbeat."""
        fake_model = types.SimpleNamespace(
            transcribe=lambda _path, **_options: {"segments": []}
        )
        fake_whisper = types.SimpleNamespace(load_model=lambda *_args, **_kwargs: fake_model)
        stream = StringIO()
        with patch.dict(sys.modules, {"whisper": fake_whisper}):
            with redirect_stderr(stream):
                WhisperTranscriber("small", "cpu").transcribe(Path("audio.wav"))
        self.assertNotIn("Whisper всё ещё работает", stream.getvalue())


class TestSpeakerAlignment(unittest.TestCase):
    """Проверяет сопоставление сегментов Whisper и локальных спикеров."""

    def setUp(self) -> None:
        """Создаёт два неперекрывающихся интервала спикеров."""
        self.speakers = [
            SpeakerSegment(0.0, 1.0, "LOCAL_00"),
            SpeakerSegment(10.0, 11.0, "LOCAL_01"),
        ]

    def test_overlap_and_speaker_selection_use_maximum_intersection(self) -> None:
        """Проверяет пересечение и выбор спикера по максимальному overlap."""
        self.assertEqual(_overlap(0, 2, 1, 3), 1)
        self.assertEqual(_overlap(0, 1, 1, 2), 0)
        self.assertEqual(_speaker_for_interval(0.2, 0.8, self.speakers), "LOCAL_00")

    def test_speaker_selection_falls_back_to_nearest_interval(self) -> None:
        """Проверяет выбор ближайшего спикера для интервала без overlap."""
        self.assertEqual(_speaker_for_interval(2.0, 3.0, self.speakers), "LOCAL_00")
        self.assertEqual(_speaker_for_interval(8.0, 9.0, self.speakers), "LOCAL_01")

    def test_combine_skips_empty_text_normalizes_times_and_names_speakers(self) -> None:
        """Проверяет очистку текста, коррекцию времени и нумерацию спикеров."""
        raw = [
            {"text": "   ", "start": 0, "end": 1},
            {"text": "  first\nphrase ", "start": -2, "end": -1},
            {"text": "second", "start": 10.2},
            {"text": "third", "start": 8, "end": 9},
        ]
        result = combine_transcript_and_speakers(raw, self.speakers)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].text, "first phrase")
        self.assertEqual((result[0].start, result[0].end), (0.0, 0.0))
        self.assertEqual(result[0].speaker, "Спикер 1")
        self.assertEqual(result[1].speaker, "Спикер 2")
        self.assertEqual(result[2].speaker, "Спикер 2")


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
