"""Обработка одного медиафайла от извлечения аудио до записи результата."""

from __future__ import annotations

import time
import tempfile
from pathlib import Path

from .diarization import diarize_audio
from .errors import FattError
from .media import extract_audio, resolve_output_path, validate_input
from .models import ProcessingReport
from .output import write_transcript
from .progress import make_progress
from .transcription import combine_transcript_and_speakers, transcribe_audio


def process_file(
    input_path: Path,
    output_path: Path,
    model_name: str,
    device: str,
    num_speakers: int | None = None,
    progress_description: str = "FATT",
) -> ProcessingReport:
    """Обрабатывает один файл и возвращает успешный или ошибочный отчёт."""
    started_at = time.perf_counter()
    progress = None
    try:
        input_path = validate_input(input_path)
        output_path = resolve_output_path(input_path, output_path)
        progress = make_progress(progress_description)
        with tempfile.TemporaryDirectory(prefix="fatt-") as temporary_dir:
            work_dir = Path(temporary_dir)
            progress.set_postfix_str("[1/3] Извлечение аудио")
            audio_path = extract_audio(input_path, work_dir)
            progress.update(1)

            progress.set_postfix_str("[2/3] Диаризация")
            diarization = diarize_audio(
                audio_path,
                device=device,
                num_speakers=num_speakers,
            )
            progress.update(1)

            progress.set_postfix_str("[3/3] Транскрибация Whisper")
            raw_transcript = transcribe_audio(audio_path, model_name, device)
            transcript = combine_transcript_and_speakers(raw_transcript, diarization)
            progress.update(1)
        write_transcript(output_path, transcript)
        word_count = sum(len(segment.text.split()) for segment in transcript)
        return ProcessingReport(
            input_path=input_path,
            output_path=output_path,
            elapsed_seconds=time.perf_counter() - started_at,
            word_count=word_count,
            success=True,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        return ProcessingReport(
            input_path=input_path,
            output_path=output_path,
            elapsed_seconds=time.perf_counter() - started_at,
            word_count=0,
            success=False,
            error=str(exc),
        )
    finally:
        if progress is not None:
            progress.close()
