"""Проверки аргументов, мастера и построения заданий FATT."""

from __future__ import annotations

import argparse
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

from fatt_core import cli
from fatt_core.device import DeviceInfo
from fatt_core.errors import FattError
from fatt_core.models import ProcessingJob, ProcessingReport


class TestCliParsing(unittest.TestCase):
    """Проверяет разбор аргументов и пользовательских значений."""

    def test_positive_int_accepts_positive_value(self) -> None:
        """Проверяет разбор положительного количества спикеров."""
        self.assertEqual(cli.positive_int("3"), 3)

    def test_positive_int_rejects_non_integer_and_zero(self) -> None:
        """Проверяет понятные ошибки для недопустимого количества спикеров."""
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.positive_int("abc")
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.positive_int("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            cli.positive_int("-2")

    def test_build_parser_uses_defaults_and_parses_all_options(self) -> None:
        """Проверяет значения по умолчанию и все поддерживаемые опции CLI."""
        parser = cli.build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.model, "small")
        self.assertIsNone(defaults.input)
        self.assertIsNone(defaults.output)
        self.assertIsNone(defaults.speakers)

        parsed = parser.parse_args(
            [
                "-i",
                "recording.mp3",
                "-o",
                "result.md",
                "-m",
                "large-v3",
                "--speakers",
                "2",
                "--language",
                "en",
            ]
        )
        self.assertEqual(parsed.input, Path("recording.mp3"))
        self.assertEqual(parsed.output, Path("result.md"))
        self.assertEqual(parsed.model, "large-v3")
        self.assertEqual(parsed.speakers, 2)
        self.assertEqual(parsed.language, "en")

    def test_parse_user_path_removes_both_quote_styles_and_expands_home(self) -> None:
        """Проверяет обработку кавычек, пробелов и домашнего каталога."""
        self.assertEqual(
            cli.parse_user_path('  "/tmp/a b.mp3"  '),
            Path("/tmp/a b.mp3"),
        )
        with patch("fatt_core.cli.Path.expanduser", return_value=Path("/home/user/file")):
            self.assertEqual(cli.parse_user_path("'~/file.wav'"), Path("/home/user/file"))

    def test_prompt_text_reports_missing_dependency_and_input_interrupts(self) -> None:
        """Проверяет ошибки отсутствующей зависимости, EOF и Ctrl-C."""
        with patch.dict(sys.modules, {"prompt_toolkit": None}):
            with self.assertRaisesRegex(FattError, "prompt-toolkit"):
                cli._prompt_text("Путь: ")

        def raise_eof(_prompt: str) -> str:
            """Имитирует завершение ввода EOF."""
            raise EOFError

        def raise_keyboard_interrupt(_prompt: str) -> str:
            """Имитирует Ctrl-C в терминальном вводе."""
            raise KeyboardInterrupt

        prompt_module = types.ModuleType("prompt_toolkit")
        prompt_module.prompt = lambda _prompt: "  value  "
        with patch.dict(sys.modules, {"prompt_toolkit": prompt_module}):
            self.assertEqual(cli._prompt_text("Путь: "), "value")
            prompt_module.prompt = raise_eof
            with self.assertRaisesRegex(FattError, "прерван"):
                cli._prompt_text("Путь: ")
            prompt_module.prompt = raise_keyboard_interrupt
            with self.assertRaisesRegex(FattError, "прерван"):
                cli._prompt_text("Путь: ")

    def test_read_path_rejects_empty_value(self) -> None:
        """Проверяет запрет пустого пути в интерактивном мастере."""
        with patch("fatt_core.cli._prompt_text", return_value=""):
            with self.assertRaisesRegex(FattError, "Путь не может быть пустым"):
                cli._read_path("Путь: ")

    def test_read_choice_accepts_default_number_name_and_retries_invalid(self) -> None:
        """Проверяет все варианты выбора в интерактивном меню."""
        with patch("fatt_core.cli._prompt_text", side_effect=["", "2", "en"]):
            self.assertEqual(cli._read_choice("Меню", ("ru", "en"), "ru"), "ru")
            self.assertEqual(cli._read_choice("Меню", ("ru", "en"), "ru"), "en")
            self.assertEqual(cli._read_choice("Меню", ("ru", "en"), "ru"), "en")

        with patch("fatt_core.cli._prompt_text", side_effect=["0", "unknown", "1"]):
            output = io.StringIO()
            with redirect_stdout(output):
                result = cli._read_choice("Меню", ("ru", "en"), "ru")
        self.assertEqual(result, "ru")
        self.assertIn("Введите номер", output.getvalue())


class TestInteractiveWizard(unittest.TestCase):
    """Проверяет ветки интерактивного мастера запуска."""

    def test_wizard_converts_auto_language_and_reads_output(self) -> None:
        """Проверяет выбор auto и ввод отдельной папки результатов."""
        answers = iter(["5", "3", '"/tmp/source dir"', "~/results"])
        with patch("fatt_core.cli._prompt_text", side_effect=lambda _prompt: next(answers)):
            source, output, model, language = cli.interactive_wizard()
        self.assertEqual(model, "large-v3")
        self.assertIsNone(language)
        self.assertEqual(source, Path("/tmp/source dir"))
        self.assertEqual(output, Path("~/results").expanduser())

    def test_wizard_uses_supplied_output_without_reading_it(self) -> None:
        """Проверяет режим мастера с заранее заданной папкой результатов."""
        answers = iter(["1", "2", "/tmp/source"])
        with patch("fatt_core.cli._prompt_text", side_effect=lambda _prompt: next(answers)):
            source, output, model, language = cli.interactive_wizard(
                default_output=Path("/tmp/output"),
            )
        self.assertEqual(source, Path("/tmp/source"))
        self.assertEqual(output, Path("/tmp/output"))
        self.assertEqual(model, "tiny")
        self.assertEqual(language, "en")


class TestJobDiscovery(unittest.TestCase):
    """Проверяет поиск входных файлов и зеркальное создание заданий."""

    def setUp(self) -> None:
        """Создаёт временную иерархию медиафайлов для тестов."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "team" / "April").mkdir(parents=True)
        (self.root / "team" / "April" / "call.MP3").write_bytes(b"audio")
        (self.root / "team" / "notes.txt").write_text("ignore", encoding="utf-8")
        (self.root / "meeting.wav").write_bytes(b"audio")

    def tearDown(self) -> None:
        """Удаляет временную иерархию после теста."""
        self.temp_dir.cleanup()

    def test_discover_input_files_recurses_and_sorts(self) -> None:
        """Проверяет рекурсивный поиск и регистронезависимые расширения."""
        files = cli.discover_input_files(self.root)
        self.assertEqual(
            files,
            [
                (self.root / "meeting.wav").resolve(),
                (self.root / "team" / "April" / "call.MP3").resolve(),
            ],
        )

    def test_discover_input_files_validates_file_and_reports_source_errors(self) -> None:
        """Проверяет режим одного файла и ошибки источника."""
        media_file = self.root / "meeting.wav"
        self.assertEqual(cli.discover_input_files(media_file), [media_file.resolve()])
        with self.assertRaisesRegex(FattError, "Источник не найден"):
            cli.discover_input_files(self.root / "missing")
        with self.assertRaisesRegex(FattError, "не найдено подходящих"):
            empty = self.root / "empty"
            empty.mkdir()
            cli.discover_input_files(empty)
        with self.assertRaisesRegex(FattError, "Неподдерживаемый формат"):
            cli.discover_input_files(self.root / "team" / "notes.txt")
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_file", return_value=False):
                with patch.object(Path, "is_dir", return_value=False):
                    with self.assertRaisesRegex(FattError, "не является файлом или папкой"):
                        cli.discover_input_files(self.root / "special")

    def test_build_jobs_for_file_supports_default_file_and_output_directory(self) -> None:
        """Проверяет результаты для файла при разных формах -o."""
        source = self.root / "meeting.wav"
        default_job = cli.build_jobs(source, None)[0]
        self.assertEqual(default_job.output_path, source.with_suffix(".txt").resolve())

        explicit = self.root / "result.md"
        self.assertEqual(cli.build_jobs(source, explicit)[0].output_path, explicit)

        output_dir = self.root / "results"
        job = cli.build_jobs(source, output_dir)[0]
        self.assertEqual(job.output_path, (output_dir / "meeting.txt").resolve())
        self.assertTrue(output_dir.is_dir())

    def test_build_jobs_for_directory_preserves_relative_structure(self) -> None:
        """Проверяет зеркальную структуру результатов для папки."""
        output = self.root / "results"
        jobs = cli.build_jobs(self.root, output)
        self.assertEqual(
            [job.output_path for job in jobs],
            [
                (output / "meeting.txt").resolve(),
                (output / "team" / "April" / "call.txt").resolve(),
            ],
        )
        self.assertTrue((output / "team" / "April").is_dir())

        default_jobs = cli.build_jobs(self.root, None)
        self.assertTrue(default_jobs[0].output_path.parent.name.endswith("_transcripts"))

    def test_build_jobs_rejects_result_collisions(self) -> None:
        """Проверяет запрет перезаписи входного файла или папки источника."""
        source_file = self.root / "meeting.wav"
        with self.assertRaisesRegex(FattError, "совпадать с входным файлом"):
            cli.build_jobs(source_file, source_file, interactive=True)
        with self.assertRaisesRegex(FattError, "совпадать с папкой источника"):
            cli.build_jobs(self.root, self.root)

    def test_ensure_directory_wraps_filesystem_error(self) -> None:
        """Проверяет преобразование ошибки создания папки в FattError."""
        with patch.object(Path, "mkdir", side_effect=OSError("permission denied")):
            with self.assertRaisesRegex(FattError, "Не удалось создать папку результатов"):
                cli._ensure_directory(self.root / "blocked")

    def test_build_jobs_interactive_txt_output_is_treated_as_directory(self) -> None:
        """Проверяет интерактивное правило, где -o всегда означает папку."""
        source = self.root / "meeting.wav"
        output = self.root / "results.txt"
        job = cli.build_jobs(source, output, interactive=True)[0]
        self.assertEqual(job.output_path, (output / "meeting.txt").resolve())


class TestCliRun(unittest.TestCase):
    """Проверяет координацию запуска и коды завершения CLI."""

    def _device(self) -> DeviceInfo:
        """Возвращает тестовое описание устройства."""
        return DeviceInfo(
            name="cpu",
            label="CPU",
            precision="FP32",
            torch_version="test",
            python_version="3.test",
            mps_built=False,
            mps_available=False,
            mps_status="не собран",
            reason="test",
        )

    def test_run_processes_jobs_and_reuses_transcriber(self) -> None:
        """Проверяет запуск всех подготовительных шагов и обработку файла."""
        args = argparse.Namespace(
            input=Path("input.wav"),
            output=Path("output.txt"),
            model="small",
            language="ru",
            speakers=2,
        )
        job = ProcessingJob(Path("input.wav"), Path("output.txt"))
        report = ProcessingReport(Path("input.wav"), Path("output.txt"), 1.0, 3, True)
        with patch("fatt_core.cli.build_jobs", return_value=[job]) as build_jobs:
            with patch("fatt_core.cli.check_ffmpeg") as check_ffmpeg:
                with patch("fatt_core.cli.get_device_info", return_value=self._device()):
                    with patch("fatt_core.cli.WhisperTranscriber") as transcriber_type:
                        with patch("fatt_core.cli.process_file", return_value=report) as process:
                            with patch("fatt_core.cli.print_report") as print_report:
                                reports = cli.run(args)
        self.assertEqual(reports, [report])
        build_jobs.assert_called_once_with(Path("input.wav"), Path("output.txt"), interactive=False)
        check_ffmpeg.assert_called_once_with()
        transcriber_type.assert_called_once_with("small", "cpu", "ru")
        process.assert_called_once_with(
            input_path=job.input_path,
            output_path=job.output_path,
            model_name="small",
            device="cpu",
            num_speakers=2,
            progress_description="FATT 1/1",
            transcriber=transcriber_type.return_value,
            language="ru",
        )
        print_report.assert_called_once()

    def test_run_interactive_passes_wizard_values(self) -> None:
        """Проверяет передачу настроек интерактивного мастера в конвейер."""
        args = argparse.Namespace(
            input=None,
            output=Path("preset"),
            model="base",
            language=None,
            speakers=None,
        )
        job = ProcessingJob(Path("input.mp3"), Path("result.txt"))
        report = ProcessingReport(job.input_path, job.output_path, 0.1, 0, True)
        with patch(
            "fatt_core.cli.interactive_wizard",
            return_value=(Path("source"), Path("output"), "tiny", None),
        ):
            with patch("fatt_core.cli.build_jobs", return_value=[job]):
                with patch("fatt_core.cli.check_ffmpeg"):
                    with patch("fatt_core.cli.get_device_info", return_value=self._device()):
                        with patch("fatt_core.cli.WhisperTranscriber"):
                            with patch("fatt_core.cli.process_file", return_value=report):
                                with patch("fatt_core.cli.print_report"):
                                    reports = cli.run(args)
        self.assertEqual(reports, [report])

    def test_print_run_configuration_prints_auto_and_manual_speakers(self) -> None:
        """Проверяет отображение конфигурации с автоматическими спикерами."""
        job = ProcessingJob(Path("input.wav"), Path("result.txt"))
        stream = io.StringIO()
        with redirect_stderr(stream):
            cli._print_run_configuration(
                source=Path("input.wav"),
                output=None,
                jobs=[job],
                model_name="small",
                language=None,
                speakers=None,
                device_info=self._device(),
            )
        text = stream.getvalue()
        self.assertIn("ПАРАМЕТРЫ ЗАПУСКА", text)
        self.assertIn("Язык речи:         автоопределение (auto)", text)
        self.assertIn("Спикеры:           автоматически", text)
        self.assertIn("Результаты:        result.txt", text)

    def test_print_run_configuration_uses_default_directory_for_source_folder(self) -> None:
        """Проверяет имя папки результатов по умолчанию для пакетного источника."""
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "recordings"
            source.mkdir()
            job = ProcessingJob(source / "call.wav", source / "call.txt")
            stream = io.StringIO()
            with redirect_stderr(stream):
                cli._print_run_configuration(
                    source=source,
                    output=None,
                    jobs=[job],
                    model_name="small",
                    language="xx",
                    speakers=2,
                    device_info=self._device(),
                )
        text = stream.getvalue()
        self.assertIn(
            f"Результаты:        {(source.parent / 'recordings_transcripts').resolve()}",
            text,
        )
        self.assertIn("Язык речи:         xx (xx)", text)
        self.assertIn("Спикеры:           2 (задано вручную)", text)

    def test_main_returns_codes_for_domain_error_interrupt_and_reports(self) -> None:
        """Проверяет коды main для ошибки, отмены и смешанного результата."""
        with patch("fatt_core.cli.run", side_effect=FattError("bad")):
            error_stream = io.StringIO()
            with redirect_stderr(error_stream):
                self.assertEqual(cli.main([]), 1)
            self.assertIn("Ошибка: bad", error_stream.getvalue())

        with patch("fatt_core.cli.run", side_effect=KeyboardInterrupt):
            error_stream = io.StringIO()
            with redirect_stderr(error_stream):
                self.assertEqual(cli.main([]), 130)
            self.assertIn("прервана", error_stream.getvalue())

        failed = ProcessingReport(Path("in"), None, 0.0, 0, False, "bad")
        with patch("fatt_core.cli.run", return_value=[failed]):
            self.assertEqual(cli.main([]), 1)

        success = ProcessingReport(Path("in"), Path("out"), 0.0, 0, True)
        with patch("fatt_core.cli.run", return_value=[success]):
            self.assertEqual(cli.main([]), 0)
