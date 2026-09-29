"""Форматирование таймкодов и запись расшифровки в файл."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .errors import FattError
from .models import TranscriptSegment


def format_timestamp(seconds: float) -> str:
    """Форматирует секунды как ``[HH:MM:SS]`` для записей любой длительности."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def write_transcript(
    output_path: Path, segments: Sequence[TranscriptSegment]
) -> None:
    """Записывает сегменты расшифровки в обычный текстовый файл или Markdown."""
    lines = [
        f"{format_timestamp(segment.start)} {segment.speaker}: {segment.text}"
        for segment in segments
    ]
    if output_path.suffix.lower() == ".md":
        content = "# Расшифровка\n\n" + "\n".join(lines) + "\n"
    else:
        content = "\n".join(lines) + ("\n" if lines else "")
    try:
        output_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise FattError(
            f"Не удалось записать результат '{output_path}': {exc}"
        ) from exc
