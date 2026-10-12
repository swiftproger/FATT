"""Проверки восстановления пакетной задачи после остановки и сбоев."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fatt_core import cli
from fatt_core.errors import FattError
from fatt_core.journal import TaskJournal
from fatt_core.models import ProcessingReport


class TestBatchRecovery(unittest.TestCase):
    """Проверяет журнал через настоящий CLI с заменой тяжёлого конвейера."""

    def setUp(self) -> None:
        """Создаёт три исходника и подключает имитацию транскрибации."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "recordings"
        self.source.mkdir()
        for name in ("a", "b", "c"):
            (self.source / f"{name}.wav").write_bytes(b"audio")
        self.result_root = self.source / "transcripts"
        self.calls: list[str] = []
        self.failed: set[str] = set()
        self.interrupt: str | None = None
        self.stack = ExitStack()
        self.output = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.output))
        self.stack.enter_context(redirect_stderr(self.output))
        self.stack.enter_context(patch("fatt_core.cli._print_run_configuration"))
        self.check_ffmpeg = self.stack.enter_context(patch("fatt_core.cli.check_ffmpeg"))
        self.stack.enter_context(patch("fatt_core.cli.get_device_info", return_value=SimpleNamespace(name="cpu", label="CPU")))
        self.model = self.stack.enter_context(patch("fatt_core.cli.WhisperTranscriber"))
        self.process = self.stack.enter_context(patch("fatt_core.cli.process_file", side_effect=self.transcribe))
        self.prompt = self.stack.enter_context(patch("fatt_core.cli._prompt_text", return_value=""))

    def tearDown(self) -> None:
        """Освобождает имитации и временную папку."""
        self.stack.close()
        self.temporary.cleanup()

    def transcribe(self, **kwargs: object) -> ProcessingReport:
        """Имитирует реальную запись текста, ошибку или прерывание файла."""
        source = kwargs["input_path"]
        output = kwargs["output_path"]
        self.calls.append(source.stem)
        if source.stem == self.interrupt:
            output.write_text("partial", encoding="utf-8")
            raise KeyboardInterrupt
        if source.stem in self.failed:
            return ProcessingReport(source, output, 1.0, 0, False, "decode failed")
        output.write_text(f"Transcript for {source.name}", encoding="utf-8")
        return ProcessingReport(source, output, 1.0, 4, True)

    def run_cli(self, *extra: str) -> list[ProcessingReport]:
        """Запускает CLI с реальным парсером, поиском файлов и журналом."""
        args = cli.build_parser().parse_args(["-i", str(self.source), *extra])
        return cli.run(args)

    def read_journal(self) -> dict:
        """Читает единственный журнал, созданный для тестовой папки."""
        return json.loads(next(self.result_root.glob(".fatt-task-*.json")).read_text(encoding="utf-8"))

    def interrupt_batch(self) -> None:
        """Останавливает первый запуск во время второго файла."""
        self.interrupt = "b"
        with self.assertRaises(KeyboardInterrupt):
            self.run_cli()
        self.interrupt = None
        self.calls.clear()

    def test_interrupt_resume_skips_success_and_reprocesses_partial_file(self) -> None:
        """После Ctrl+C готовый файл пропускается, частичный обрабатывается заново."""
        self.interrupt_batch()
        state = self.read_journal()
        self.assertEqual(state["status"], "interrupted")
        self.assertEqual(state["files"][str(self.source / "a.wav")]["status"], "success")
        self.assertEqual(state["files"][str(self.source / "b.wav")]["status"], "processing")
        reports = self.run_cli()
        self.assertEqual(self.calls, ["b", "c"])
        self.prompt.assert_called_once()
        self.assertTrue(all(report.success for report in reports))
        self.assertEqual(self.read_journal()["status"], "completed")
        self.assertNotEqual((self.result_root / "b.txt").read_text(), "partial")
        self.assertIn("Найдена незавершённая задача", self.output.getvalue())

    def test_completed_batch_does_not_load_model_or_process_again(self) -> None:
        """Повторный запуск завершённой задачи не требует FFmpeg или Whisper."""
        first = self.run_cli()
        finished_at = self.read_journal()["finished_at"]
        self.calls.clear()
        self.model.reset_mock()
        self.check_ffmpeg.reset_mock()
        second = self.run_cli()
        self.assertEqual(second, first)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.read_journal()["finished_at"], finished_at)
        self.model.assert_not_called()
        self.check_ffmpeg.assert_not_called()
        self.prompt.assert_not_called()
        self.assertIn("Задача уже завершена", self.output.getvalue())

    def test_resume_flag_and_restart_flag(self) -> None:
        """Флаги позволяют продолжить без вопроса или явно обработать всё заново."""
        self.interrupt_batch()
        self.run_cli("--resume")
        self.assertEqual(self.calls, ["b", "c"])
        self.prompt.assert_not_called()
        self.calls.clear()
        self.run_cli("--restart")
        self.assertEqual(self.calls, ["a", "b", "c"])
        self.assertEqual(self.read_journal()["status"], "completed")

    def test_cancel_preserves_journal_and_does_not_load_model(self) -> None:
        """Отмена найденной задачи не меняет сохранённые данные."""
        self.interrupt_batch()
        path = next(self.result_root.glob(".fatt-task-*.json"))
        before = path.read_bytes()
        self.prompt.return_value = "отмена"
        self.model.reset_mock()
        with self.assertRaises(KeyboardInterrupt):
            self.run_cli()
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.calls, [])
        self.model.assert_not_called()

    def test_restart_choice_reprocesses_all_files(self) -> None:
        """Ответ «заново» сбрасывает пропуск готовых файлов."""
        self.interrupt_batch()
        self.prompt.return_value = "заново"
        self.run_cli()
        self.assertEqual(self.calls, ["a", "b", "c"])

    def test_failed_files_are_logged_and_only_errors_are_retried(self) -> None:
        """Задача с ошибками не считается полностью завершённой."""
        self.failed.add("b")
        reports = self.run_cli()
        self.assertEqual([r.success for r in reports], [True, False, True])
        state = self.read_journal()
        self.assertEqual(state["status"], "completed_with_errors")
        self.assertEqual(state["files"][str(self.source / "b.wav")]["error"], "decode failed")
        self.failed.clear()
        self.calls.clear()
        self.run_cli("--resume")
        self.assertEqual(self.calls, ["b"])
        self.assertEqual(self.read_journal()["status"], "completed")

    def test_changed_source_deleted_output_and_edited_output_are_reprocessed(self) -> None:
        """Готовность проверяется по исходнику и фактическому содержимому текста."""
        self.run_cli()
        (self.source / "a.wav").write_bytes(b"different audio")
        (self.result_root / "b.txt").unlink()
        (self.result_root / "c.txt").write_text("Edited output", encoding="utf-8")
        self.calls.clear()
        self.run_cli("--resume")
        self.assertEqual(self.calls, ["a", "b", "c"])

    def test_new_and_removed_inputs_update_task(self) -> None:
        """Добавленный исходник обрабатывается, удалённый исключается из задачи."""
        self.run_cli()
        (self.source / "d.wav").write_bytes(b"new audio")
        (self.source / "b.wav").unlink()
        self.calls.clear()
        reports = self.run_cli("--resume")
        self.assertEqual(self.calls, ["d"])
        self.assertEqual([r.input_path.stem for r in reports], ["a", "c", "d"])
        self.assertNotIn(str(self.source / "b.wav"), self.read_journal()["files"])

    def test_only_changed_file_is_reprocessed(self) -> None:
        """Изменение одного результата сохраняет готовность остальных файлов."""
        self.run_cli()
        for change in ("source", "missing", "edited"):
            with self.subTest(change=change):
                if change == "source":
                    (self.source / "b.wav").write_bytes(b"changed audio")
                elif change == "missing":
                    (self.result_root / "b.txt").unlink()
                else:
                    (self.result_root / "b.txt").write_text("Edited", encoding="utf-8")
                self.calls.clear()
                self.run_cli("--resume")
                self.assertEqual(self.calls, ["b"])

    def test_all_files_saved_before_final_status_is_recovered_as_complete(self) -> None:
        """Остановка после последнего успеха не вызывает повторную транскрибацию."""
        self.run_cli()
        path = next(self.result_root.glob(".fatt-task-*.json"))
        data = self.read_journal()
        data["status"] = "running"
        path.write_text(json.dumps(data))
        self.calls.clear()
        self.model.reset_mock()
        self.run_cli()
        self.assertEqual(self.calls, [])
        self.model.assert_not_called()
        self.prompt.assert_not_called()
        self.assertEqual(self.read_journal()["status"], "completed")

    def test_custom_output_folder_has_independent_journal(self) -> None:
        """Явно выбранная папка хранит свой журнал и поддерживает повторный запуск."""
        output = self.root / "results"
        self.run_cli("-o", str(output))
        self.assertEqual(len(list((output / "transcripts").glob(".fatt-task-*.json"))), 1)
        self.assertFalse(self.result_root.exists())
        self.calls.clear()
        self.run_cli("-o", str(output))
        self.assertEqual(self.calls, [])

    def test_main_returns_error_and_cancel_exit_codes(self) -> None:
        """Ошибки файлов и отмена восстановления дают корректные коды выхода."""
        self.failed.add("b")
        self.assertEqual(cli.main(["-i", str(self.source)]), 1)
        self.prompt.return_value = "отмена"
        self.assertEqual(cli.main(["-i", str(self.source)]), 130)

    def test_resume_and_restart_are_mutually_exclusive(self) -> None:
        """Парсер запрещает противоречивые флаги восстановления."""
        with self.assertRaises(SystemExit) as exit_result:
            self.run_cli("--resume", "--restart")
        self.assertEqual(exit_result.exception.code, 2)

    def test_changed_settings_require_restart_and_default_to_cancel(self) -> None:
        """Другая модель не смешивается с результатами сохранённой задачи."""
        self.run_cli()
        before = self.read_journal()
        with self.assertRaisesRegex(FattError, "Параметры задачи изменились"):
            self.run_cli("-m", "medium", "--resume")
        with self.assertRaises(KeyboardInterrupt):
            self.run_cli("-m", "medium")
        self.assertEqual(self.read_journal(), before)
        self.calls.clear()
        self.run_cli("-m", "medium", "--restart")
        self.assertEqual(self.calls, ["a", "b", "c"])
        self.assertEqual(self.read_journal()["settings"]["model"], "medium")

    def test_changed_settings_can_be_restarted_by_answer(self) -> None:
        """Подтверждённый новый запуск сохраняет новые параметры."""
        self.run_cli()
        self.calls.clear()
        self.prompt.return_value = "заново"
        self.run_cli("--language", "en", "--speakers", "2")
        self.assertEqual(self.calls, ["a", "b", "c"])
        self.assertEqual(self.read_journal()["settings"]["language"], "en")
        self.assertEqual(self.read_journal()["settings"]["speakers"], 2)

    def test_corrupt_journal_is_preserved_until_explicit_restart(self) -> None:
        """Повреждённый журнал не вызывает незаметную повторную обработку."""
        self.run_cli()
        path = next(self.result_root.glob(".fatt-task-*.json"))
        path.write_text("{broken", encoding="utf-8")
        self.calls.clear()
        with self.assertRaisesRegex(FattError, "Журнал повреждён"):
            self.run_cli("--resume")
        self.assertEqual(self.calls, [])
        self.assertEqual(path.read_text(), "{broken")
        self.run_cli("--restart")
        self.assertEqual(self.calls, ["a", "b", "c"])

    def test_model_load_failure_is_logged_and_recoverable(self) -> None:
        """Ошибка до первого файла сохраняет незавершённую задачу."""
        self.model.side_effect = FattError("model unavailable")
        with self.assertRaisesRegex(FattError, "model unavailable"):
            self.run_cli()
        self.assertEqual(self.read_journal()["status"], "interrupted")
        self.assertEqual(self.read_journal()["last_error"], "model unavailable")
        self.model.side_effect = None
        self.run_cli("--resume")
        self.assertEqual(self.calls, ["a", "b", "c"])

    def test_output_collision_is_rejected_before_processing(self) -> None:
        """Разные расширения с одним именем не могут перезаписывать общий текст."""
        (self.source / "a.mp3").write_bytes(b"audio")
        with self.assertRaisesRegex(FattError, "один путь результата"):
            self.run_cli()
        self.assertEqual(self.calls, [])


class TestJournalStorage(unittest.TestCase):
    """Проверяет атомарную запись, блокировку и завершение процесса без cleanup."""

    def setUp(self) -> None:
        """Создаёт исходники и реальные задания без загрузки моделей."""
        self.temporary = tempfile.TemporaryDirectory()
        self.source = Path(self.temporary.name).resolve()
        for name in ("a", "b"):
            (self.source / f"{name}.wav").write_bytes(b"audio")
        self.jobs = cli.build_jobs(self.source, None)
        self.output = self.source / "transcripts"
        self.settings = {"model": "small", "language": "ru", "speakers": None}

    def tearDown(self) -> None:
        """Удаляет временные файлы журнала и результаты."""
        self.temporary.cleanup()

    def journal(self, **kwargs: object) -> TaskJournal:
        """Создаёт журнал для тестовых заданий."""
        return TaskJournal(self.source, self.output, self.jobs, self.settings, **kwargs)

    def success(self, journal: TaskJournal, index: int, content: str = "text") -> None:
        """Фиксирует реально сохранённый результат одного файла."""
        job = self.jobs[index]
        journal.start_file(job)
        job.output_path.write_text(content, encoding="utf-8")
        journal.record(job, ProcessingReport(job.input_path, job.output_path, 1, 1, True))

    def test_atomic_failure_keeps_previous_journal_and_cleans_temporary_file(self) -> None:
        """Сбой замены журнала не портит предыдущую сохранённую версию."""
        with self.journal() as journal:
            journal.plan()
            journal.begin()
            before = journal.path.read_bytes()
            with patch("fatt_core.journal.os.replace", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(FattError, "Не удалось сохранить журнал"):
                    journal.start_file(self.jobs[0])
            self.assertEqual(journal.path.read_bytes(), before)
            self.assertEqual(list(self.output.glob("*.tmp")), [])

    def test_parallel_writer_is_blocked_and_lock_is_released(self) -> None:
        """Вторая задача не может одновременно писать в ту же папку результатов."""
        with self.journal():
            with self.assertRaisesRegex(FattError, "заблокировать"):
                with self.journal():
                    self.fail("Concurrent writer acquired lock")
        with self.journal() as journal:
            journal.plan()
            journal.begin()

    def test_abrupt_process_exit_keeps_success_and_processing_states(self) -> None:
        """После жёсткого завершения процесса успешный файл остаётся готовым."""
        script = '''
import os, sys
from pathlib import Path
from fatt_core import cli
from fatt_core.journal import TaskJournal
from fatt_core.models import ProcessingReport
source = Path(sys.argv[1])
jobs = cli.build_jobs(source, None)
settings = {"model": "small", "language": "ru", "speakers": None}
with TaskJournal(source, source / "transcripts", jobs, settings) as journal:
    journal.plan()
    journal.begin()
    job = jobs[0]
    journal.start_file(job)
    job.output_path.write_text("saved", encoding="utf-8")
    journal.record(job, ProcessingReport(job.input_path, job.output_path, 1, 1, True))
    journal.start_file(jobs[1])
    os._exit(7)
'''
        process = subprocess.run([sys.executable, "-c", script, str(self.source)], cwd=Path(__file__).resolve().parents[1], capture_output=True)
        self.assertEqual(process.returncode, 7, process.stderr.decode())
        with self.journal() as journal:
            self.assertEqual(journal.data["status"], "running")
            journal.plan()
            self.assertEqual(list(journal.cached), [self.jobs[0].input_path])
            self.assertEqual(journal.data["files"][str(self.jobs[1].input_path)]["status"], "pending")

    def test_empty_transcript_is_success_and_not_reprocessed(self) -> None:
        """Пустой текст без речи является корректным завершённым результатом."""
        with self.journal() as journal:
            journal.plan()
            journal.begin()
            self.success(journal, 0, "")
        with self.journal() as journal:
            journal.plan()
            self.assertIn(self.jobs[0].input_path, journal.cached)

    def test_input_changed_during_processing_is_not_success(self) -> None:
        """Результат для изменившегося в процессе исходника требует повторения."""
        with self.journal() as journal:
            journal.plan()
            journal.begin()
            (self.source / "a.wav").write_bytes(b"changed during run")
            self.success(journal, 0)
            entry = journal.data["files"][str(self.jobs[0].input_path)]
            self.assertEqual(entry["status"], "error")
            self.assertIn("изменился", entry["error"])

    def test_success_without_saved_output_is_recorded_as_error(self) -> None:
        """Успешный ответ конвейера без файла результата нельзя считать готовым."""
        with self.journal() as journal:
            journal.plan()
            journal.begin()
            job = self.jobs[0]
            journal.start_file(job)
            report = journal.record(job, ProcessingReport(job.input_path, job.output_path, 1, 1, True))
            self.assertFalse(report.success)
            self.assertEqual(journal.data["files"][str(job.input_path)]["status"], "error")

    def test_incomplete_success_record_is_rejected(self) -> None:
        """Синтаксически корректный JSON без данных успеха нельзя восстановить."""
        with self.journal() as journal:
            journal.plan()
            journal.begin()
            self.success(journal, 0)
            path = journal.path
        data = json.loads(path.read_text())
        del data["files"][str(self.jobs[0].input_path)]["word_count"]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(FattError, "Журнал повреждён"):
            with self.journal():
                pass
