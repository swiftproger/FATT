"""Проверки последовательного конвейера обработки одного файла."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from fatt_core import pipeline
from fatt_core.errors import FattError
from fatt_core.models import SpeakerSegment, TranscriptSegment


class FakeProgress:
    """Тестовый progress bar, записывающий переходы между этапами."""

    def __init__(self) -> None:
        """Создаёт пустой журнал действий."""
        self.events: list[tuple[str, object]] = []

    def set_postfix_str(self, text: str) -> None:
        """Записывает начало этапа."""
        self.events.append(("stage", text))

    def update(self, amount: int = 1) -> None:
        """Записывает завершение этапа."""
        self.events.append(("update", amount))

    def close(self) -> None:
        """Записывает закрытие progress bar."""
        self.events.append(("close", True))


class TestProcessFile(unittest.TestCase):
    """Проверяет успешную обработку и безопасное завершение конвейера."""

    def setUp(self) -> None:
        """Создаёт входной файл и папку результата."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "input.wav"
        self.output_path = self.root / "output.txt"
        self.input_path.write_bytes(b"audio")
        self.transcript = [
            TranscriptSegment(0.0, 1.0, "one two", "Спикер 1"),
            TranscriptSegment(1.0, 2.0, "three", "Спикер 2"),
        ]

    def tearDown(self) -> None:
        """Удаляет временную папку после теста."""
        self.temp_dir.cleanup()

    def test_process_file_runs_three_stages_and_writes_report(self) -> None:
        """Проверяет порядок этапов, объединение сегментов и подсчёт слов."""
        progress = FakeProgress()
        transcriber = MagicMock()
        transcriber.transcribe.return_value = [{"text": "raw"}]
        diarization = [SpeakerSegment(0.0, 2.0, "LOCAL_00")]
        with patch("fatt_core.pipeline.validate_input", return_value=self.input_path):
            with patch("fatt_core.pipeline.resolve_output_path", return_value=self.output_path):
                with patch(
                    "fatt_core.pipeline.make_progress", return_value=progress
                ) as make_progress:
                    with patch(
                        "fatt_core.pipeline.extract_audio",
                        return_value=self.root / "audio.wav",
                    ) as extract:
                        with patch(
                            "fatt_core.pipeline.diarize_audio", return_value=diarization
                        ) as diarize:
                            with patch(
                                "fatt_core.pipeline.combine_transcript_and_speakers",
                                return_value=self.transcript,
                            ) as combine:
                                with patch("fatt_core.pipeline.write_transcript") as write:
                                    report = pipeline.process_file(
                                        self.input_path,
                                        self.output_path,
                                        "small",
                                        "cpu",
                                        num_speakers=2,
                                        progress_description="FATT 1/1",
                                        transcriber=transcriber,
                                        language="ru",
                                    )
        self.assertTrue(report.success)
        self.assertEqual(report.word_count, 3)
        self.assertEqual(report.output_path, self.output_path)
        make_progress.assert_called_once_with("FATT 1/1")
        extract.assert_called_once()
        diarize.assert_called_once_with(
            self.root / "audio.wav", device="cpu", num_speakers=2
        )
        transcriber.transcribe.assert_called_once_with(self.root / "audio.wav")
        combine.assert_called_once_with(
            [{"text": "raw"}], diarization
        )
        write.assert_called_once_with(self.output_path, self.transcript)
        self.assertEqual(
            progress.events,
            [
                ("stage", "[1/3] Извлечение аудио"),
                ("update", 1),
                ("stage", "[2/3] Диаризация"),
                ("update", 1),
                ("stage", "[3/3] Транскрибация Whisper"),
                ("update", 1),
                ("close", True),
            ],
        )

    def test_process_file_integrates_transcript_alignment_and_real_output(self) -> None:
        """Проверяет полный pipeline с реальным объединением и записью TXT."""
        progress = FakeProgress()
        transcriber = MagicMock()
        transcriber.transcribe.return_value = [
            {"start": 0.1, "end": 0.9, "text": " first "},
            {"start": 1.1, "end": 1.9, "text": "second"},
        ]
        diarization = [
            SpeakerSegment(0.0, 1.0, "LOCAL_00"),
            SpeakerSegment(1.0, 2.0, "LOCAL_01"),
        ]
        with patch(
            "fatt_core.pipeline.make_progress", return_value=progress
        ):
            with patch(
                "fatt_core.pipeline.extract_audio",
                return_value=self.root / "audio.wav",
            ):
                with patch(
                    "fatt_core.pipeline.diarize_audio", return_value=diarization
                ):
                    report = pipeline.process_file(
                        self.input_path,
                        self.output_path,
                        "small",
                        "cpu",
                        transcriber=transcriber,
                    )
        self.assertTrue(report.success)
        self.assertEqual(report.word_count, 2)
        self.assertEqual(
            self.output_path.read_text(encoding="utf-8"),
            "[00:00:00] Спикер 1: first\n"
            "[00:00:01] Спикер 2: second\n",
        )

    def test_process_file_constructs_transcriber_when_not_supplied(self) -> None:
        """Проверяет создание WhisperTranscriber внутри конвейера."""
        progress = FakeProgress()
        fake_transcriber = MagicMock()
        fake_transcriber.transcribe.return_value = []
        with patch("fatt_core.pipeline.validate_input", return_value=self.input_path):
            with patch("fatt_core.pipeline.resolve_output_path", return_value=self.output_path):
                with patch("fatt_core.pipeline.make_progress", return_value=progress):
                    with patch(
                        "fatt_core.pipeline.extract_audio",
                        return_value=self.root / "audio.wav",
                    ):
                        with patch("fatt_core.pipeline.diarize_audio", return_value=[]):
                            with patch(
                                "fatt_core.pipeline.WhisperTranscriber",
                                return_value=fake_transcriber,
                            ) as transcriber_type:
                                with patch(
                                    "fatt_core.pipeline.combine_transcript_and_speakers",
                                    return_value=[],
                                ):
                                    with patch("fatt_core.pipeline.write_transcript"):
                                        report = pipeline.process_file(
                                            self.input_path,
                                            self.output_path,
                                            "base",
                                            "mps",
                                            language=None,
                                        )
        self.assertTrue(report.success)
        transcriber_type.assert_called_once_with("base", "mps", None)

    def test_process_file_returns_error_report_and_closes_progress(self) -> None:
        """Проверяет преобразование обычной ошибки в ProcessingReport."""
        progress = FakeProgress()
        with patch("fatt_core.pipeline.validate_input", return_value=self.input_path):
            with patch("fatt_core.pipeline.resolve_output_path", return_value=self.output_path):
                with patch("fatt_core.pipeline.make_progress", return_value=progress):
                    with patch(
                        "fatt_core.pipeline.extract_audio",
                        side_effect=RuntimeError("extract failed"),
                    ):
                        report = pipeline.process_file(
                            self.input_path,
                            self.output_path,
                            "small",
                            "cpu",
                        )
        self.assertFalse(report.success)
        self.assertEqual(report.error, "extract failed")
        self.assertEqual(report.word_count, 0)
        self.assertEqual(progress.events[-1], ("close", True))

    def test_process_file_preserves_keyboard_interrupt_and_closes_progress(self) -> None:
        """Проверяет, что Ctrl-C не превращается в обычную ошибку отчёта."""
        progress = FakeProgress()
        with patch("fatt_core.pipeline.validate_input", return_value=self.input_path):
            with patch("fatt_core.pipeline.resolve_output_path", return_value=self.output_path):
                with patch("fatt_core.pipeline.make_progress", return_value=progress):
                    with patch(
                        "fatt_core.pipeline.extract_audio",
                        side_effect=KeyboardInterrupt,
                    ):
                        with self.assertRaises(KeyboardInterrupt):
                            pipeline.process_file(
                                self.input_path,
                                self.output_path,
                                "small",
                                "cpu",
                            )
        self.assertEqual(progress.events[-1], ("close", True))

    def test_process_file_returns_error_when_output_write_fails(self) -> None:
        """Проверяет обработку ошибки записи после завершения этапов."""
        progress = FakeProgress()
        with patch("fatt_core.pipeline.validate_input", return_value=self.input_path):
            with patch("fatt_core.pipeline.resolve_output_path", return_value=self.output_path):
                with patch("fatt_core.pipeline.make_progress", return_value=progress):
                    with patch(
                        "fatt_core.pipeline.extract_audio",
                        return_value=self.root / "audio.wav",
                    ):
                        with patch("fatt_core.pipeline.diarize_audio", return_value=[]):
                            with patch(
                                "fatt_core.pipeline.WhisperTranscriber"
                            ) as transcriber_type:
                                transcriber_type.return_value.transcribe.return_value = []
                                with patch(
                                    "fatt_core.pipeline.combine_transcript_and_speakers",
                                    return_value=[],
                                ):
                                    with patch(
                                        "fatt_core.pipeline.write_transcript",
                                        side_effect=FattError("write failed"),
                                    ):
                                        report = pipeline.process_file(
                                            self.input_path,
                                            self.output_path,
                                            "small",
                                            "cpu",
                                        )
        self.assertFalse(report.success)
        self.assertEqual(report.error, "write failed")
