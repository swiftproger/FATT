"""Разбор командной строки и координация работы FATT."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from .diarization import diarize_audio
from .device import get_torch_device
from .errors import FattError
from .media import check_ffmpeg, extract_audio, resolve_output_path, validate_input
from .output import write_transcript
from .progress import make_progress
from .transcription import combine_transcript_and_speakers, transcribe_audio


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
        "-i",
        "--input",
        required=True,
        type=Path,
        help="путь к исходному аудио- или видеофайлу",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="путь к .txt/.md результату (по умолчанию: имя входного файла.txt)",
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


def run(args: argparse.Namespace) -> Path:
    """Выполняет конвертацию медиа, локальную диаризацию, транскрибацию и запись."""
    input_path = validate_input(args.input)
    output_path = resolve_output_path(input_path, args.output)
    check_ffmpeg()
    device = get_torch_device()

    print(f"Устройство: {device}", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="fatt-") as temporary_dir:
        work_dir = Path(temporary_dir)
        progress = make_progress()
        try:
            progress.set_postfix_str("[1/3] Извлечение аудио")
            audio_path = extract_audio(input_path, work_dir)
            progress.update(1)

            progress.set_postfix_str("[2/3] Диаризация")
            diarization = diarize_audio(
                audio_path,
                device=device,
                num_speakers=args.speakers,
            )
            progress.update(1)

            progress.set_postfix_str("[3/3] Транскрибация Whisper")
            raw_transcript = transcribe_audio(audio_path, args.model, device)
            transcript = combine_transcript_and_speakers(raw_transcript, diarization)
            progress.update(1)
        finally:
            progress.close()

    write_transcript(output_path, transcript)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    """Разбирает аргументы, запускает FATT и возвращает код завершения оболочке."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path = run(args)
    except FattError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nОбработка прервана пользователем.", file=sys.stderr)
        return 130
    print(f"Готово: {output_path}")
    return 0
