#!/usr/bin/env python3
"""FATT (FromAudioToText): transcribe audio/video with speaker diarization."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".mp4", ".mkv", ".avi", ".mov"
}
MODEL_CHOICES = ("tiny", "base", "small", "medium", "large-v3")
DEFAULT_MODEL = "small"


class FattError(RuntimeError):
    """An expected, user-facing FATT error."""


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    label: str


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ожидалось целое число") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("число должно быть больше нуля")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fatt.py",
        description=(
            "Транскрибация аудио/видео с распознаванием спикеров "
            "(Whisper + локальная кластеризация голосов)."
        ),
    )
    parser.add_argument(
        "-i", "--input", required=True, type=Path,
        help="путь к исходному аудио- или видеофайлу",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="путь к .txt/.md результату (по умолчанию: имя входного файла.txt)",
    )
    parser.add_argument(
        "-m", "--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL,
        help="размер модели Whisper (по умолчанию: small)",
    )
    parser.add_argument(
        "--speakers", type=positive_int,
        help="известное количество спикеров (необязательно)",
    )
    return parser


def validate_input(path: Path) -> Path:
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
    if shutil.which("ffmpeg") is None:
        raise FattError(
            "FFmpeg не найден в PATH. Установите его командой "
            "'brew install ffmpeg' и повторите запуск."
        )


def extract_audio(input_path: Path, work_dir: Path) -> Path:
    """Convert any supported input to a temporary 16 kHz mono WAV file."""
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
        details = (exc.stderr or b"").decode("utf-8", errors="replace").strip()
        details = details or "неизвестная ошибка FFmpeg"
        raise FattError(f"FFmpeg не смог обработать файл: {details}") from exc
    except OSError as exc:
        raise FattError(f"Не удалось запустить FFmpeg: {exc}") from exc
    if not audio_path.exists() or audio_path.stat().st_size == 0:
        raise FattError("FFmpeg не создал аудиофайл для обработки")
    return audio_path


def get_torch_device() -> str:
    """Select MPS on Apple Silicon when available, otherwise CPU."""
    try:
        import torch
    except ImportError as exc:
        raise FattError(
            "PyTorch не установлен. Установите torch, torchvision и torchaudio."
        ) from exc
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return device


def _read_wav_samples(audio_path: Path) -> tuple[Any, int]:
    """Read the normalized PCM WAV produced by FFmpeg."""
    try:
        import numpy as np
    except ImportError as exc:
        raise FattError(
            "numpy не установлен. Установите зависимости из requirements.txt."
        ) from exc
    import wave

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
    except (OSError, wave.Error) as exc:
        raise FattError(f"Не удалось прочитать временный WAV-файл: {exc}") from exc

    if sample_rate != 16000 or channels != 1 or sample_width != 2:
        raise FattError(
            "Временный WAV-файл имеет неожиданные параметры: ожидались "
            "16 кГц, mono, PCM 16-bit"
        )
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def _audio_feature(samples: Any, sample_rate: int) -> Any:
    """Build a compact local voice feature without downloading a model."""
    import numpy as np

    frame_length = int(sample_rate * 0.025)
    hop_length = int(sample_rate * 0.010)
    if len(samples) < frame_length:
        samples = np.pad(samples, (0, frame_length - len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_length)[::hop_length]
    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)

    # Log-spaced spectral bands describe voice timbre and work without a
    # pretrained speaker model or an online service.
    edges = np.geomspace(80.0, 7600.0, 25)
    band_means = []
    for low, high in zip(edges[:-1], edges[1:]):
        band = spectrum[:, (frequencies >= low) & (frequencies < high)]
        if band.size:
            band_means.append(np.log1p(band.mean(axis=1)))
        else:
            # At low frequencies a narrow log band can contain no FFT bins.
            # Keep the feature matrix rectangular for np.stack().
            band_means.append(np.zeros(spectrum.shape[0], dtype=np.float32))
    band_features = np.stack(band_means, axis=1)
    spectral_centroid = (
        (spectrum * frequencies).sum(axis=1) / (spectrum.sum(axis=1) + 1e-8)
    )
    zero_crossing_rate = (frames[:, 1:] * frames[:, :-1] < 0).mean(axis=1)
    feature = np.concatenate(
        [
            band_features.mean(axis=0),
            band_features.std(axis=0),
            [spectral_centroid.mean() / sample_rate, spectral_centroid.std() / sample_rate],
            [zero_crossing_rate.mean(), zero_crossing_rate.std()],
        ]
    ).astype(np.float32)
    norm = np.linalg.norm(feature)
    return feature / norm if norm > 0 else feature


def _cluster_features(features: Sequence[Any], num_speakers: int | None) -> list[int]:
    """Cluster local voice features using cosine similarity/k-means."""
    import numpy as np

    if not features:
        return []
    matrix = np.asarray(features, dtype=np.float32)
    if len(matrix) == 1:
        return [0]

    if num_speakers is not None:
        cluster_count = min(num_speakers, len(matrix))
        initial_indices = np.linspace(
            0, len(matrix) - 1, cluster_count, dtype=int
        )
        centers = matrix[initial_indices].copy()
        labels = np.zeros(len(matrix), dtype=int)
        for _ in range(30):
            labels = np.argmax(matrix @ centers.T, axis=1)
            updated = np.array(
                [
                    matrix[labels == cluster].mean(axis=0)
                    if np.any(labels == cluster)
                    else centers[cluster]
                    for cluster in range(cluster_count)
                ],
                dtype=np.float32,
            )
            norms = np.linalg.norm(updated, axis=1, keepdims=True)
            updated = updated / np.maximum(norms, 1e-8)
            if np.array_equal(updated, centers):
                break
            centers = updated
        return labels.tolist()

    # When the count is unknown, add a new cluster only when the current
    # window is sufficiently unlike all speaker centroids. This is intentionally
    # conservative: over-splitting a single speaker is worse for a transcript.
    centroids: list[Any] = []
    counts: list[int] = []
    labels = []
    similarity_threshold = 0.82
    for feature in matrix:
        if not centroids:
            centroids.append(feature.copy())
            counts.append(1)
            labels.append(0)
            continue
        similarities = [float(feature @ centroid) for centroid in centroids]
        best_cluster = int(np.argmax(similarities))
        if similarities[best_cluster] < similarity_threshold and len(centroids) < 8:
            centroids.append(feature.copy())
            counts.append(1)
            labels.append(len(centroids) - 1)
            continue
        labels.append(best_cluster)
        counts[best_cluster] += 1
        centroids[best_cluster] = (
            centroids[best_cluster] * (counts[best_cluster] - 1) + feature
        ) / counts[best_cluster]
        centroid_norm = np.linalg.norm(centroids[best_cluster])
        if centroid_norm > 0:
            centroids[best_cluster] /= centroid_norm
    return labels


def diarize_audio(
    audio_path: Path,
    device: str,
    num_speakers: int | None = None,
) -> list[SpeakerSegment]:
    """Perform token-free local diarization using acoustic feature clustering.

    This is lighter and more private than pyannote, but less accurate on
    overlapping speech and very short turns. ``device`` is kept in the API so
    the main pipeline can share the same MPS/CPU device selection as Whisper.
    """
    del device
    import numpy as np

    samples, sample_rate = _read_wav_samples(audio_path)
    window_length = int(sample_rate * 1.5)
    windows: list[tuple[float, float, Any]] = []
    energies: list[float] = []
    chunks: list[tuple[int, Any]] = []
    for start in range(0, len(samples), window_length):
        chunk = samples[start : start + window_length]
        if len(chunk) < sample_rate * 0.5:
            continue
        padded = np.pad(chunk, (0, max(0, window_length - len(chunk))))
        chunks.append((start, padded))
        energies.append(float(np.sqrt(np.mean(padded**2))))
    if not chunks:
        raise FattError("Аудиофайл слишком короткий для локальной диаризации")

    noise_floor = float(np.percentile(energies, 20))
    peak_energy = max(energies)
    # Keep the threshold below the signal peak for recordings that contain
    # speech throughout; otherwise a short all-speech file looks silent.
    threshold = max(0.001, min(noise_floor * 1.8, peak_energy * 0.65))
    for (start, chunk), energy in zip(chunks, energies):
        if energy < threshold:
            continue
        features = _audio_feature(chunk, sample_rate)
        windows.append((start / sample_rate, min((start + window_length) / sample_rate, len(samples) / sample_rate), features))
    if not windows:
        raise FattError("В аудио не обнаружена речь для локальной диаризации")

    labels = _cluster_features([item[2] for item in windows], num_speakers)
    segments: list[SpeakerSegment] = []
    for (start, end, _), label in zip(windows, labels):
        speaker = f"LOCAL_{label:02d}"
        if segments and segments[-1].label == speaker and start <= segments[-1].end + 0.1:
            previous = segments[-1]
            segments[-1] = SpeakerSegment(previous.start, max(previous.end, end), speaker)
        else:
            segments.append(SpeakerSegment(start, end, speaker))
    return segments


def transcribe_audio(
    audio_path: Path, model_name: str, device: str
) -> list[dict[str, Any]]:
    """Run OpenAI Whisper and return its raw timed segments."""
    try:
        import whisper
    except ImportError as exc:
        raise FattError(
            "openai-whisper не установлен. Установите зависимости из README.md."
        ) from exc
    try:
        model = whisper.load_model(model_name, device=device)
        result = model.transcribe(
            str(audio_path), fp16=device == "mps", verbose=False
        )
    except Exception as exc:
        raise FattError(f"Ошибка транскрибации Whisper: {exc}") from exc
    return list(result.get("segments", []))


def _overlap(
    left_start: float, left_end: float, right_start: float, right_end: float
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))


def _speaker_for_interval(
    start: float, end: float, diarization: Sequence[SpeakerSegment]
) -> str:
    """Choose the speaker with the largest overlap, or the nearest turn."""
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
    """Assign a stable human-readable speaker number to each Whisper segment."""
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


def format_timestamp(seconds: float) -> str:
    """Format seconds as [HH:MM:SS], allowing durations longer than one day."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def write_transcript(
    output_path: Path, segments: Sequence[TranscriptSegment]
) -> None:
    """Write a plain-text or Markdown transcript."""
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


def _make_progress() -> Any:
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise FattError(
            "tqdm не установлен. Установите зависимости из README.md."
        ) from exc
    return tqdm(total=3, desc="FATT", unit="этап", dynamic_ncols=True)


def run(args: argparse.Namespace) -> Path:
    """Execute the complete three-stage processing pipeline."""
    input_path = validate_input(args.input)
    output_path = resolve_output_path(input_path, args.output)
    check_ffmpeg()
    device = get_torch_device()

    print(f"Устройство: {device}", file=sys.stderr)
    with tempfile.TemporaryDirectory(prefix="fatt-") as temporary_dir:
        work_dir = Path(temporary_dir)
        progress = _make_progress()
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


if __name__ == "__main__":
    raise SystemExit(main())
