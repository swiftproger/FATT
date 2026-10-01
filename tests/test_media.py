"""Проверки валидации путей и интеграции с FFmpeg."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from fatt_core import media
from fatt_core.errors import FattError


class TestMediaPaths(unittest.TestCase):
    """Проверяет правила входных и выходных путей."""

    def setUp(self) -> None:
        """Создаёт временные файлы и папки для проверки путей."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "input.MP3"
        self.input_path.write_bytes(b"audio")

    def tearDown(self) -> None:
        """Удаляет временные файлы после теста."""
        self.temp_dir.cleanup()

    def test_validate_input_normalizes_valid_extension(self) -> None:
        """Проверяет успешную валидацию файла с регистронезависимым расширением."""
        self.assertEqual(media.validate_input(self.input_path), self.input_path.resolve())

    def test_validate_input_reports_missing_directory_and_extension(self) -> None:
        """Проверяет ошибки отсутствующего пути, каталога и формата."""
        with self.assertRaisesRegex(FattError, "Входной файл не найден"):
            media.validate_input(self.root / "missing.wav")
        directory = self.root / "folder"
        directory.mkdir()
        with self.assertRaisesRegex(FattError, "Путь не является файлом"):
            media.validate_input(directory)
        unsupported = self.root / "file.txt"
        unsupported.write_text("text", encoding="utf-8")
        with self.assertRaisesRegex(FattError, "Неподдерживаемый формат"):
            media.validate_input(unsupported)
        no_extension = self.root / "no-extension"
        no_extension.write_bytes(b"data")
        with self.assertRaisesRegex(FattError, "<без расширения>"):
            media.validate_input(no_extension)

    def test_resolve_output_path_uses_default_and_accepts_txt_md(self) -> None:
        """Проверяет расширения результата и путь TXT по умолчанию."""
        default = media.resolve_output_path(self.input_path, None)
        self.assertEqual(default, self.input_path.with_suffix(".txt").resolve())
        self.assertEqual(
            media.resolve_output_path(self.input_path, self.root / "result.MD"),
            (self.root / "result.MD").resolve(),
        )

    def test_resolve_output_path_rejects_invalid_or_colliding_paths(self) -> None:
        """Проверяет ошибки расширения, совпадения и родительского пути."""
        with self.assertRaisesRegex(FattError, "расширение"):
            media.resolve_output_path(self.input_path, self.root / "result.txt.bak")
        same_result = self.root / "input.txt"
        with self.assertRaisesRegex(FattError, "совпадать"):
            media.resolve_output_path(same_result.resolve(), same_result)
        missing_parent = self.root / "missing" / "result.txt"
        with self.assertRaisesRegex(FattError, "Папка для результата не найдена"):
            media.resolve_output_path(self.input_path, missing_parent)
        parent_file = self.root / "parent"
        parent_file.write_text("not a directory", encoding="utf-8")
        with self.assertRaisesRegex(FattError, "не является папкой"):
            media.resolve_output_path(self.input_path, parent_file / "result.txt")

    def test_check_ffmpeg_reports_presence_or_absence(self) -> None:
        """Проверяет реакцию на наличие и отсутствие бинарника FFmpeg."""
        with patch("fatt_core.media.shutil.which", return_value="/usr/bin/ffmpeg"):
            media.check_ffmpeg()
        with patch("fatt_core.media.shutil.which", return_value=None):
            with self.assertRaisesRegex(FattError, "FFmpeg не найден"):
                media.check_ffmpeg()


class _FakeFfmpegError(Exception):
    """Имитирует исключение ffmpeg.Error в тестах."""

    def __init__(self, stderr: bytes | str | None = None) -> None:
        """Сохраняет диагностический вывод FFmpeg."""
        super().__init__("ffmpeg failed")
        self.stderr = stderr


class _FakeOutput:
    """Имитирует цепочку построения команды ffmpeg-python."""

    def __init__(self, output_path: Path, error: Exception | None = None) -> None:
        """Создаёт цепочку, которая записывает WAV или выбрасывает ошибку."""
        self.output_path = output_path
        self.error = error
        self.calls: list[tuple[str, object]] = []

    def output(self, path: str, **options: object) -> "_FakeOutput":
        """Запоминает параметры output."""
        self.calls.append(("output", (path, options)))
        return self

    def global_args(self, *args: str) -> "_FakeOutput":
        """Запоминает глобальные аргументы FFmpeg."""
        self.calls.append(("global_args", args))
        return self

    def overwrite_output(self) -> "_FakeOutput":
        """Запоминает разрешение перезаписи."""
        self.calls.append(("overwrite_output", True))
        return self

    def run(self, **_options: object) -> None:
        """Создаёт результат либо возвращает ошибку FFmpeg."""
        if self.error is not None:
            raise self.error
        output_path = Path(self.calls[0][1][0])  # type: ignore[index]
        output_path.write_bytes(b"RIFF fake wav")


class TestExtractAudio(unittest.TestCase):
    """Проверяет преобразование медиафайла через ffmpeg-python."""

    def setUp(self) -> None:
        """Создаёт временные пути входа и рабочего каталога."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input_path = self.root / "input.mp3"
        self.input_path.write_bytes(b"audio")
        self.work_dir = self.root / "work"
        self.work_dir.mkdir()

    def tearDown(self) -> None:
        """Удаляет временные пути после теста."""
        self.temp_dir.cleanup()

    def _ffmpeg_module(self, error: Exception | None = None) -> types.ModuleType:
        """Создаёт модуль ffmpeg с тестовой цепочкой вызовов."""
        module = types.ModuleType("ffmpeg")
        module.Error = _FakeFfmpegError
        stream = _FakeOutput(self.work_dir / "audio_16khz_mono.wav", error)
        module.input = lambda _path: stream
        module.stream = stream
        return module

    def test_extract_audio_builds_expected_command_and_returns_wav(self) -> None:
        """Проверяет параметры нормализации и наличие созданного WAV."""
        fake_module = self._ffmpeg_module()
        with patch("fatt_core.media.check_ffmpeg"):
            with patch.dict(sys.modules, {"ffmpeg": fake_module}):
                result = media.extract_audio(self.input_path, self.work_dir)
        self.assertEqual(result, self.work_dir / "audio_16khz_mono.wav")
        calls = fake_module.stream.calls
        self.assertEqual(calls[0][0], "output")
        output_path, options = calls[0][1]  # type: ignore[misc]
        self.assertEqual(output_path, str(result))
        self.assertEqual(options["format"], "wav")
        self.assertEqual(options["ac"], 1)
        self.assertEqual(options["ar"], 16000)
        self.assertEqual(options["acodec"], "pcm_s16le")
        self.assertEqual(options["vn"], None)

    def test_extract_audio_reports_missing_dependency_ffmpeg_errors_and_empty_result(self) -> None:
        """Проверяет все основные ошибки запуска FFmpeg."""
        with patch("fatt_core.media.check_ffmpeg"):
            with patch.dict(sys.modules, {"ffmpeg": None}):
                with self.assertRaisesRegex(FattError, "ffmpeg-python"):
                    media.extract_audio(self.input_path, self.work_dir)

        for details in (b"bad bytes", "bad text", None):
            fake_module = self._ffmpeg_module(_FakeFfmpegError(details))
            with patch("fatt_core.media.check_ffmpeg"):
                with patch.dict(sys.modules, {"ffmpeg": fake_module}):
                    with self.assertRaisesRegex(FattError, "FFmpeg не смог"):
                        media.extract_audio(self.input_path, self.work_dir)

        class FailingOutput(_FakeOutput):
            """Имитирует сбой запуска дочернего процесса."""

            def run(self, **_options: object) -> None:
                """Выбрасывает системную ошибку."""
                raise OSError("no executable")

        fake_module = types.ModuleType("ffmpeg")
        fake_module.Error = _FakeFfmpegError
        stream = FailingOutput(self.work_dir / "audio_16khz_mono.wav")
        fake_module.input = lambda _path: stream
        with patch("fatt_core.media.check_ffmpeg"):
            with patch.dict(sys.modules, {"ffmpeg": fake_module}):
                with self.assertRaisesRegex(FattError, "Не удалось запустить FFmpeg"):
                    media.extract_audio(self.input_path, self.work_dir)

        class EmptyOutput(_FakeOutput):
            """Имитирует успешный FFmpeg без созданного результата."""

            def run(self, **_options: object) -> None:
                """Не создаёт выходной файл."""
                return None

        fake_module = types.ModuleType("ffmpeg")
        fake_module.Error = _FakeFfmpegError
        stream = EmptyOutput(self.work_dir / "audio_16khz_mono.wav")
        fake_module.input = lambda _path: stream
        with patch("fatt_core.media.check_ffmpeg"):
            with patch.dict(sys.modules, {"ffmpeg": fake_module}):
                with self.assertRaisesRegex(FattError, "не создал аудиофайл"):
                    media.extract_audio(self.input_path, self.work_dir)
