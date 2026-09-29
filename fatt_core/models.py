"""Небольшие модели данных, общие для модулей обработки FATT."""

from __future__ import annotations

from dataclasses import dataclass


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
