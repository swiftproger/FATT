"""Небольшие модели данных, общие для модулей обработки FATT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SpeakerSegment:
    """Описывает временной интервал, отнесённый к локальному кластеру спикера."""

    start: float
    end: float
    label: str


@dataclass(frozen=True)
class TranscriptSegment:
    """Описывает фрагмент расшифровки с именем отображаемого спикера."""

    start: float
    end: float
    text: str
    speaker: str


@dataclass(frozen=True)
class ProcessingJob:
    """Описывает пару входного файла и соответствующего текстового результата."""

    input_path: Path
    output_path: Path


@dataclass(frozen=True)
class ProcessingReport:
    """Хранит итог обработки одного файла для финального отчёта."""

    input_path: Path
    output_path: Path | None
    elapsed_seconds: float
    word_count: int
    success: bool
    error: str | None = None
