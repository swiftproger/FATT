"""Локальная диаризация без токенов на основе кластеризации акустических признаков."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Sequence

from .errors import FattError
from .models import SpeakerSegment


def _read_wav_samples(audio_path: Path) -> tuple[Any, int]:
    """Читает нормализованный моно WAV PCM 16 кГц, созданный модулем media."""
    try:
        import numpy as np
    except ImportError as exc:
        raise FattError(
            "numpy не установлен. Установите зависимости из requirements.txt."
        ) from exc

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
    """Строит нормализованный спектральный вектор признаков для аудиофрагмента."""
    import numpy as np

    frame_length = int(sample_rate * 0.025)
    hop_length = int(sample_rate * 0.010)
    if len(samples) < frame_length:
        samples = np.pad(samples, (0, frame_length - len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_length)[::hop_length]
    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(frame_length, 1.0 / sample_rate)

    edges = np.geomspace(80.0, 7600.0, 25)
    band_means = []
    for low, high in zip(edges[:-1], edges[1:]):
        band = spectrum[:, (frequencies >= low) & (frequencies < high)]
        if band.size:
            band_means.append(np.log1p(band.mean(axis=1)))
        else:
            # Узкая низкочастотная полоса может не содержать ни одного FFT-бина.
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
    """Группирует векторы признаков косинусным k-means или автоматическим порогом."""
    import numpy as np

    if not features:
        return []
    matrix = np.asarray(features, dtype=np.float32)
    if len(matrix) == 1:
        return [0]

    if num_speakers is not None:
        cluster_count = min(num_speakers, len(matrix))
        initial_indices = np.linspace(0, len(matrix) - 1, cluster_count, dtype=int)
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
    """Возвращает локальные сегменты спикеров без Hugging Face и удалённых сервисов."""
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
    threshold = max(0.001, min(noise_floor * 1.8, peak_energy * 0.65))
    for (start, chunk), energy in zip(chunks, energies):
        if energy < threshold:
            continue
        features = _audio_feature(chunk, sample_rate)
        windows.append(
            (
                start / sample_rate,
                min((start + window_length) / sample_rate, len(samples) / sample_rate),
                features,
            )
        )
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
