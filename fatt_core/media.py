"""Проверка входных данных и нормализация аудио через FFmpeg."""

from __future__ import annotations

import shutil
from pathlib import Path

from .errors import FattError


SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".aac",
    ".aif",
    ".aiff",
    ".alac",
    ".amr",
    ".caf",
    ".mid",
    ".midi",
    ".ogg",
    ".oga",
    ".opus",
    ".wma",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v",
    ".3gp",
    ".ts",
    ".mts",
    ".m2ts",
    ".flv",
    ".wmv",
    ".vob",
    ".ogv",
    ".mxf",
    ".asf",
    ".rm",
    ".rmvb",
}


def validate_input(path: Path) -> Path:
    """Проверяет входной файл и возвращает его абсолютный нормализованный путь."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FattError(f"Входной файл не найден: {path}")
    if not path.is_file():
        raise FattError(f"Путь не является файлом: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise FattError(
            f"Неподдерживаемый формат '{path.suffix or '<без расширения>'}'. "
            f"Поддерживаются: {supported}"
        )
    return path


def resolve_output_path(input_path: Path, output: Path | None) -> Path:
    """Проверяет путь результата и при необходимости создаёт путь к TXT по умолчанию."""
    result = (output or input_path.with_suffix(".txt")).expanduser().resolve()
    if result.suffix.lower() not in {".txt", ".md"}:
        raise FattError("Файл результата должен иметь расширение .txt или .md")
    if result == input_path:
        raise FattError("Файл результата не должен совпадать с входным файлом")
    if not result.parent.exists():
        raise FattError(f"Папка для результата не найдена: {result.parent}")
    if not result.parent.is_dir():
        raise FattError(f"Путь для результата не является папкой: {result.parent}")
    return result


def check_ffmpeg() -> None:
    """Вызывает понятную ошибку, если исполняемый файл FFmpeg недоступен."""
    if shutil.which("ffmpeg") is None:
        raise FattError(
            "FFmpeg не найден в PATH. Установите его командой "
            "'brew install ffmpeg' и повторите запуск."
        )


def extract_audio(input_path: Path, work_dir: Path) -> Path:
    """Преобразует входной файл во временный моно WAV PCM 16 кГц."""
    check_ffmpeg()
    try:
        import ffmpeg
    except ImportError as exc:
        raise FattError(
            "ffmpeg-python не установлен. Установите зависимости из README.md."
        ) from exc

    audio_path = work_dir / "audio_16khz_mono.wav"
    try:
        (
            ffmpeg.input(str(input_path))
            .output(
                str(audio_path),
                format="wav",
                ac=1,
                ar=16000,
                acodec="pcm_s16le",
                vn=None,
            )
            .global_args("-hide_banner", "-loglevel", "error")
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        raw_details = exc.stderr or b""
        details = (
            raw_details.decode("utf-8", errors="replace")
            if isinstance(raw_details, bytes)
            else str(raw_details)
        ).strip() or "неизвестная ошибка FFmpeg"
        raise FattError(f"FFmpeg не смог обработать файл: {details}") from exc
    except OSError as exc:
        raise FattError(f"Не удалось запустить FFmpeg: {exc}") from exc
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise FattError("FFmpeg не создал аудиофайл для обработки")
    return audio_path
