"""Разбор командной строки и координация работы FATT."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

from .device import get_torch_device
from .errors import FattError
from .media import SUPPORTED_EXTENSIONS, check_ffmpeg, validate_input
from .models import ProcessingJob, ProcessingReport
from .pipeline import process_file
from .report import print_report


MODEL_CHOICES = ("tiny", "base", "small", "medium", "large-v3")
DEFAULT_MODEL = "small"


def positive_int(value: str) -> int:
    """Разбирает строго положительное целое число для параметра argparse."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидалось целое число") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("число должно быть больше нуля")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Создаёт и возвращает парсер аргументов командной строки FATT."""
    parser = argparse.ArgumentParser(
        prog="fatt.py",
        description=(
            "Транскрибация аудио/видео с распознаванием спикеров "
            "(Whisper + локальная кластеризация голосов)."
        ),
    )
    parser.add_argument(
        "-i", "--input", type=Path,
        help="путь к файлу или папке; без параметра запускается мастер",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="путь к файлу или папке результатов",
    )
    parser.add_argument(
        "-m",
        "--model",
        choices=MODEL_CHOICES,
        default=DEFAULT_MODEL,
        help="размер модели Whisper (по умолчанию: small)",
    )
    parser.add_argument(
        "--speakers",
        type=positive_int,
        help="известное количество спикеров (необязательно)",
    )
    return parser


def _read_path(prompt: str) -> Path:
    """Запрашивает путь с поддержкой стрелок и преобразует его в объект Path."""
    try:
        from prompt_toolkit import prompt as terminal_prompt
    except ImportError as exc:
        raise FattError(
            "prompt-toolkit не установлен. Установите зависимости из "
            "requirements.txt."
        ) from exc
    try:
        value = terminal_prompt(prompt).strip()
    except EOFError as exc:
        raise FattError("Ввод был прерван до указания пути") from exc
    except KeyboardInterrupt as exc:
        raise FattError("Ввод пути прерван пользователем") from exc
    if not value:
        raise FattError("Путь не может быть пустым")
    return parse_user_path(value)


def parse_user_path(value: str) -> Path:
    """Убирает внешние кавычки, сохраняя пробелы, решётки и другие символы."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return Path(value).expanduser()


def interactive_wizard(default_output: Path | None = None) -> tuple[Path, Path]:
    """Проводит интерактивный мастер выбора источника и папки результатов."""
    print("FATT — мастер запуска")
    print("Укажите аудиофайл, видеофайл или папку с ними.")
    source = _read_path("1. Файл или папка: ")
    if default_output is None:
        output = _read_path("2. Папка для сохранения результатов: ")
    else:
        output = default_output
        print(f"2. Папка для сохранения результатов: {output}")
    return source, output


def discover_input_files(source: Path) -> list[Path]:
    """Находит поддерживаемые файлы в источнике с рекурсией любой глубины."""
    source = source.expanduser().resolve()
    if source.is_file():
        return [validate_input(source)]
    if not source.exists():
        raise FattError(f"Источник не найден: {source}")
    if not source.is_dir():
        raise FattError(f"Источник не является файлом или папкой: {source}")
    files = sorted(
        path.resolve()
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FattError(f"В папке не найдено подходящих медиафайлов: {source}")
    return files


def _ensure_directory(path: Path) -> None:
    """Создаёт папку результатов и преобразует системную ошибку в FattError."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise FattError(f"Не удалось создать папку результатов '{path}': {exc}") from exc


def build_jobs(
    source: Path,
    output: Path | None,
    interactive: bool = False,
) -> list[ProcessingJob]:
    """Создаёт задания и зеркальную структуру текстовых результатов."""
    source = source.expanduser().resolve()
    input_files = discover_input_files(source)
    if source.is_file():
        input_path = input_files[0]
        if output is None:
            output_path = input_path.with_suffix(".txt")
        elif interactive or output.suffix.lower() not in {".txt", ".md"}:
            output_root = output.expanduser().resolve()
            if output_root == input_path:
                raise FattError("Папка результатов не должна совпадать с входным файлом")
            _ensure_directory(output_root)
            output_path = output_root / f"{input_path.stem}.txt"
        else:
            output_path = output
        _ensure_directory(output_path.parent)
        return [ProcessingJob(input_path, output_path)]

    output_root = (
        output.expanduser().resolve()
        if output is not None
        else source.parent / f"{source.name}_transcripts"
    )
    if output_root == source:
        raise FattError("Папка результатов не должна совпадать с папкой источника")
    _ensure_directory(output_root)
    jobs = []
    for input_path in input_files:
        relative_path = input_path.relative_to(source)
        output_path = (output_root / relative_path).with_suffix(".txt")
        _ensure_directory(output_path.parent)
        jobs.append(ProcessingJob(input_path, output_path))
    return jobs


def run(args: argparse.Namespace, interactive: bool = False) -> list[ProcessingReport]:
    """Запускает последовательную обработку всех файлов и печатает отчёт."""
    source = args.input
    output = args.output
    if source is None:
        source, output = interactive_wizard(output)
        interactive = True
    jobs = build_jobs(source, output, interactive=interactive)
    check_ffmpeg()
    device = get_torch_device()
    print(f"Устройство: {device}", file=sys.stderr)
    reports: list[ProcessingReport] = []
    started_at = time.perf_counter()
    for index, job in enumerate(jobs, start=1):
        reports.append(
            process_file(
                input_path=job.input_path,
                output_path=job.output_path,
                model_name=args.model,
                device=device,
                num_speakers=args.speakers,
                progress_description=f"FATT {index}/{len(jobs)}",
            )
        )
    print_report(reports, time.perf_counter() - started_at)
    return reports


def main(argv: Sequence[str] | None = None) -> int:
    """Разбирает аргументы, запускает FATT и возвращает код завершения оболочке."""
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    try:
        reports = run(args, interactive=args.input is None)
    except FattError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nОбработка прервана пользователем.", file=sys.stderr)
        return 130
    return 0 if all(report.success for report in reports) else 1
