"""Надёжный журнал пакетной задачи и восстановление завершённых файлов."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import FattError
from .models import ProcessingJob, ProcessingReport


def _timestamp() -> str:
    """Возвращает время события в формате UTC с указанием часового пояса."""
    return datetime.now(timezone.utc).isoformat()


def _input_signature(path: Path) -> dict[str, int]:
    """Фиксирует размер и время изменения исходника без чтения большого видео."""
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _output_signature(path: Path) -> str:
    """Проверяет содержимое текстового результата, включая пустую расшифровку."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TaskJournal:
    """Хранит состояние задачи и блокирует параллельную запись в её результаты."""

    def __init__(
        self, source: Path, output_root: Path, jobs: list[ProcessingJob],
        settings: dict[str, Any], restart: bool = False,
    ) -> None:
        """Задаёт источник, параметры и отдельное имя журнала для каждой папки."""
        self.source = source.resolve()
        self.output_root = output_root.resolve()
        task_id = hashlib.sha256(str(self.source).encode()).hexdigest()[:16]
        self.path = self.output_root / f".fatt-task-{task_id}.json"
        self.jobs = jobs
        self.settings = settings
        self.restart = restart
        self.existing = False
        self.compatible = True
        self.cached: dict[Path, ProcessingReport] = {}
        self.data: dict[str, Any] = {}
        self._lock: Any = None
        self._started = False

    def __enter__(self) -> TaskJournal:
        """Получает системную блокировку и читает сохранённое состояние задачи."""
        try:
            self._lock = (self.output_root / ".fatt.lock").open("a+b")
            if os.name == "nt":
                import msvcrt
                self._lock.seek(0)
                if not self._lock.read(1):
                    self._lock.write(b"0")
                    self._lock.flush()
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if self._lock is not None:
                self._lock.close()
            raise FattError(
                f"Не удалось заблокировать папку результатов {self.output_root}: "
                "возможно, её уже обрабатывает другой экземпляр FATT."
            ) from exc
        try:
            self._load()
        except BaseException:
            self._lock.close()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Фиксирует прерывание и всегда освобождает системную блокировку."""
        try:
            if exc_type is not None and self._started:
                self.data["status"] = "interrupted"
                self.data["last_error"] = str(exc) or exc_type.__name__
                self.save()
        finally:
            self._lock.close()

    def _load(self) -> None:
        """Проверяет журнал, не перезаписывая повреждённые данные без --restart."""
        self.existing = self.path.exists()
        if not self.existing or self.restart:
            return
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            self._validate()
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise FattError(
                f"Журнал повреждён или недоступен: {self.path}. "
                "Для нового запуска укажите --restart."
            ) from exc
        self.compatible = self.data["settings"] == self.settings

    def _validate(self) -> None:
        """Проверяет версию, источник и поля, используемые при восстановлении."""
        if (
            not isinstance(self.data, dict)
            or self.data.get("version") != 1
            or self.data.get("source") != str(self.source)
            or self.data.get("status") not in {
                "running", "interrupted", "completed", "completed_with_errors"
            }
            or not isinstance(self.data.get("settings"), dict)
            or not isinstance(self.data.get("files"), dict)
        ):
            raise ValueError("Неверный формат журнала")
        for entry in self.data["files"].values():
            if not isinstance(entry, dict) or entry.get("status") not in {
                "pending", "processing", "success", "error"
            }:
                raise ValueError("Неверный статус файла")
            if entry["status"] == "success":
                if (
                    not isinstance(entry.get("input_signature"), dict)
                    or not isinstance(entry.get("output_path"), str)
                    or not isinstance(entry.get("output_signature"), str)
                    or not isinstance(entry.get("elapsed_seconds"), (int, float))
                    or not isinstance(entry.get("word_count"), int)
                ):
                    raise ValueError("Неполные данные завершённого файла")

    def plan(self, restart: bool = False) -> None:
        """Восстанавливает только неизменённые исходники с целыми результатами."""
        previous = (
            self.data.get("files", {})
            if self.compatible and not (restart or self.restart) else {}
        )
        self.cached = {}
        entries = {}
        try:
            for job in self.jobs:
                signature = _input_signature(job.input_path)
                old = previous.get(str(job.input_path), {})
                reusable = (
                    old.get("status") == "success"
                    and old.get("input_signature") == signature
                    and old.get("output_path") == str(job.output_path)
                )
                if reusable:
                    try:
                        reusable = old["output_signature"] == _output_signature(job.output_path)
                    except OSError:
                        reusable = False
                if reusable:
                    entries[str(job.input_path)] = old
                    self.cached[job.input_path] = ProcessingReport(
                        job.input_path, job.output_path, old["elapsed_seconds"],
                        old["word_count"], True,
                    )
                else:
                    entries[str(job.input_path)] = {
                        "status": "pending", "input_signature": signature,
                        "output_path": str(job.output_path),
                    }
        except OSError as exc:
            raise FattError(f"Не удалось проверить файлы задачи: {exc}") from exc
        created_at = self.data.get("created_at", _timestamp())
        finished_at = (
            self.data.get("finished_at")
            if self.data.get("status") == "completed" and len(self.cached) == len(self.jobs)
            else None
        )
        self.data = {
            "version": 1, "source": str(self.source), "settings": self.settings,
            "created_at": _timestamp() if restart or self.restart else created_at,
            "status": "running", "files": entries,
        }
        if finished_at is not None:
            self.data["finished_at"] = finished_at

    def save(self) -> None:
        """Атомарно заменяет журнал после сброса данных временного файла на диск."""
        temporary = None
        self.data["updated_at"] = _timestamp()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.output_root,
                prefix=self.path.name + ".", suffix=".tmp", delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(self.data, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            raise FattError(f"Не удалось сохранить журнал {self.path}: {exc}") from exc
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def begin(self) -> None:
        """Сохраняет состояние до загрузки модели и начала обработки."""
        self.data["status"] = "running"
        self.save()
        self._started = True

    def start_file(self, job: ProcessingJob) -> None:
        """Помечает файл выполняющимся перед запуском конвейера."""
        entry = self.data["files"][str(job.input_path)]
        entry.update(status="processing", started_at=_timestamp())
        self.save()

    def record(self, job: ProcessingJob, report: ProcessingReport) -> ProcessingReport:
        """Сохраняет успех только после проверки записанного текста и исходника."""
        entry = self.data["files"][str(job.input_path)]
        if report.success:
            try:
                if entry["input_signature"] != _input_signature(job.input_path):
                    raise OSError("Исходный файл изменился во время обработки")
                entry["output_signature"] = _output_signature(job.output_path)
            except OSError as exc:
                report = ProcessingReport(
                    job.input_path, job.output_path, report.elapsed_seconds, 0, False, str(exc),
                )
        entry.update(
            status="success" if report.success else "error",
            elapsed_seconds=report.elapsed_seconds, word_count=report.word_count,
            error=report.error, finished_at=_timestamp(),
        )
        self.save()
        return report

    def finish(self) -> None:
        """Отмечает полностью завершённую задачу либо завершение с ошибками."""
        successful = all(
            entry["status"] == "success" for entry in self.data["files"].values()
        )
        self.data["status"] = "completed" if successful else "completed_with_errors"
        self.data.setdefault("finished_at", _timestamp())
        self.save()
        self._started = False
