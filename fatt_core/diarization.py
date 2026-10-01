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
    except (EOFError, OSError, wave.Error) as exc:
        raise FattError(f"Не удалось прочитать временный WAV-файл: {exc}") from exc

    if sample_rate != 16000 or channels != 1 or sample_width != 2:
        raise FattError(
            "Временный WAV-файл имеет неожиданные параметры: ожидались "
            "16 кГц, mono, PCM 16-bit"
        )
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


class AcousticFeatureExtractor:
    """Кэширует параметры FFT и быстро строит признаки голоса для окон."""

    def __init__(self, sample_rate: int) -> None:
        """Подготавливает FFT-окно и полосы частот для заданной частоты дискретизации."""
        import numpy as np

        self._np = np
        self._sample_rate = sample_rate
        self._frame_length = int(sample_rate * 0.025)
        self._hop_length = int(sample_rate * 0.010)
        self._window = np.hanning(self._frame_length).astype(np.float32)
        self._frequencies = np.fft.rfftfreq(
            self._frame_length, 1.0 / sample_rate
        )
        edges = np.geomspace(80.0, 7600.0, 25)
        self._band_masks = [
            (self._frequencies >= low) & (self._frequencies < high)
            for low, high in zip(edges[:-1], edges[1:])
        ]

    def extract(self, samples: Any) -> Any:
        """Строит нормализованный спектральный вектор для одного аудиоокна."""
        np = self._np
        if len(samples) < self._frame_length:
            samples = np.pad(samples, (0, self._frame_length - len(samples)))
        frames = np.lib.stride_tricks.sliding_window_view(
            samples, self._frame_length
        )[:: self._hop_length]
        spectrum = np.abs(np.fft.rfft(frames * self._window, axis=1)) ** 2
        band_means = []
        for mask in self._band_masks:
            band = spectrum[:, mask]
            if band.size:
                band_means.append(np.log1p(band.mean(axis=1)))
            else:
                # Узкая низкочастотная полоса может не содержать FFT-бинов.
                band_means.append(np.zeros(spectrum.shape[0], dtype=np.float32))
        band_features = np.stack(band_means, axis=1)
        spectral_centroid = (
            (spectrum * self._frequencies).sum(axis=1)
            / (spectrum.sum(axis=1) + 1e-8)
        )
        zero_crossing_rate = (frames[:, 1:] * frames[:, :-1] < 0).mean(axis=1)
        feature = np.concatenate(
            [
                band_features.mean(axis=0),
                band_features.std(axis=0),
                [
                    spectral_centroid.mean() / self._sample_rate,
                    spectral_centroid.std() / self._sample_rate,
                ],
                [zero_crossing_rate.mean(), zero_crossing_rate.std()],
            ]
        ).astype(np.float32)
        norm = np.linalg.norm(feature)
        return feature / norm if norm > 0 else feature


def _audio_feature(samples: Any, sample_rate: int) -> Any:
    """Строит признаки одного окна через оптимизированный экстрактор."""
    return AcousticFeatureExtractor(sample_rate).extract(samples)


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
            if np.allclose(updated, centers, rtol=1e-4, atol=1e-5):
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
    hop_length = int(sample_rate * 1.0)
    feature_extractor = AcousticFeatureExtractor(sample_rate)
    windows: list[tuple[float, float, Any]] = []
    energies: list[float] = []
    chunks: list[tuple[int, Any]] = []
    for start in range(0, len(samples), hop_length):
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
        features = feature_extractor.extract(chunk)
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
    labels = _smooth_labels(labels)
    segments: list[SpeakerSegment] = []
    for index, ((start, end, _), label) in enumerate(zip(windows, labels)):
        speaker = f"LOCAL_{label:02d}"
        segment_start = start if index == 0 else (windows[index - 1][0] + start) / 2
        segment_end = (
            end
            if index == len(windows) - 1
            else (start + windows[index + 1][0]) / 2
        )
        if segments and segments[-1].label == speaker:
            previous = segments[-1]
            segments[-1] = SpeakerSegment(previous.start, segment_end, speaker)
        else:
            segments.append(SpeakerSegment(segment_start, segment_end, speaker))
    return segments


def _smooth_labels(labels: Sequence[int]) -> list[int]:
    """Убирает одиночные скачки метки между двумя одинаковыми спикерами."""
    if len(labels) < 3:
        return list(labels)
    smoothed = list(labels)
    for index in range(1, len(labels) - 1):
        if labels[index - 1] == labels[index + 1] != labels[index]:
            smoothed[index] = labels[index - 1]
    return smoothed
