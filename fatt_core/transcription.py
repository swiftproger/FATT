"""Транскрибация Whisper и согласование временных шкал спикеров."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import FattError
from .models import SpeakerSegment, TranscriptSegment


class WhisperTranscriber:
    """Загружает модель Whisper один раз и повторно использует её для файлов."""

    def __init__(
        self,
        model_name: str,
        device: str,
        language: str | None = None,
    ) -> None:
        """Загружает выбранную модель Whisper на указанное устройство."""
        try:
            import whisper
        except ImportError as exc:
            raise FattError(
                "openai-whisper не установлен. Установите зависимости из README.md."
            ) from exc
        captured_warnings: list[warnings.WarningMessage] = []
        try:
            with warnings.catch_warnings(record=True) as captured_warnings:
                warnings.simplefilter("always")
                self._model = whisper.load_model(model_name, device=device)
        except Exception as exc:
            raise FattError(f"Не удалось загрузить модель Whisper: {exc}") from exc
        _print_whisper_warnings(captured_warnings)
        self._device = device
        self._language = language

    def transcribe(self, audio_path: Path) -> list[dict[str, Any]]:
        """Транскрибирует один WAV-файл и возвращает сегменты с таймкодами."""
        options: dict[str, Any] = {
            "fp16": self._device == "mps",
            "verbose": False,
            "temperature": 0,
        }
        if self._language:
            options["language"] = self._language
        try:
            result = self._model.transcribe(str(audio_path), **options)
        except Exception as exc:
            raise FattError(f"Ошибка транскрибации Whisper: {exc}") from exc
        return list(result.get("segments", []))


def _print_whisper_warnings(
    captured_warnings: Sequence[warnings.WarningMessage],
) -> None:
    """Печатает предупреждения Whisper в коротком понятном виде."""
    for warning in captured_warnings:
        message = str(warning.message)
        if "checksum does not match" in message.lower():
            message = (
                "Кэш модели повреждён или загружен не полностью. "
                "Whisper автоматически скачал модель заново."
            )
        print(f"Предупреждение Whisper: {message}", file=sys.stderr)


def transcribe_audio(
    audio_path: Path,
    model_name: str,
    device: str,
    language: str | None = None,
) -> list[dict[str, Any]]:
    """Транскрибирует один файл через временный экземпляр Whisper."""
    return WhisperTranscriber(model_name, device, language).transcribe(audio_path)


def _overlap(
    left_start: float, left_end: float, right_start: float, right_end: float
) -> float:
    """Возвращает количество секунд пересечения двух временных интервалов."""
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _speaker_for_interval(
    start: float, end: float, diarization: Sequence[SpeakerSegment]
) -> str:
    """Выбирает спикера с максимальным пересечением или ближайшей серединой интервала."""
    best = max(
        diarization,
        key=lambda item: _overlap(start, end, item.start, item.end),
    )
    if _overlap(start, end, best.start, best.end) > 0:
        return best.label
    midpoint = (start + end) / 2
    nearest = min(
        diarization,
        key=lambda item: (
            min(abs(midpoint - item.start), abs(midpoint - item.end)),
            item.start,
        ),
    )
    return nearest.label


def combine_transcript_and_speakers(
    raw_segments: Iterable[dict[str, Any]],
    diarization: Sequence[SpeakerSegment],
) -> list[TranscriptSegment]:
    """Назначает сегментам Whisper стабильные понятные номера спикеров."""
    speaker_names: dict[str, str] = {}
    combined: list[TranscriptSegment] = []
    for raw in raw_segments:
        text = " ".join(str(raw.get("text", "")).split())
        if not text:
            continue
        start = max(0.0, float(raw.get("start", 0.0)))
        end = max(start, float(raw.get("end", start)))
        raw_speaker = _speaker_for_interval(start, end, diarization)
        if raw_speaker not in speaker_names:
            speaker_names[raw_speaker] = f"Спикер {len(speaker_names) + 1}"
        combined.append(
            TranscriptSegment(
                start=start,
                end=end,
                text=text,
                speaker=speaker_names[raw_speaker],
            )
        )
    return combined
