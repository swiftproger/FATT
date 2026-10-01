"""Проверки чтения WAV, признаков и локальной диаризации."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from fatt_core import diarization
from fatt_core.errors import FattError


class _FakeSampleArray:
    """Минимальная замена массива numpy для чтения PCM в тесте."""

    def __init__(self, values: list[int]) -> None:
        """Сохраняет значения PCM."""
        self.values = values

    def astype(self, _dtype: object) -> "_FakeSampleArray":
        """Имитирует преобразование типа массива."""
        return self

    def __truediv__(self, divisor: float) -> list[float]:
        """Имитирует нормализацию PCM в диапазон float."""
        return [value / divisor for value in self.values]


class _FakeNumpyForWav(types.ModuleType):
    """Модуль numpy только для проверки нормализации WAV."""

    int16 = object()
    float32 = object()

    def frombuffer(self, _data: bytes, dtype: object) -> _FakeSampleArray:
        """Возвращает заранее заданные PCM-сэмплы."""
        del dtype
        return _FakeSampleArray([0, 16384, -16384])


def _write_wav(path: Path, *, sample_rate: int = 16000, channels: int = 1) -> None:
    """Создаёт небольшой PCM WAV с заданными параметрами."""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * max(1, channels))


class TestWavReading(unittest.TestCase):
    """Проверяет чтение нормализованных WAV-файлов."""

    def setUp(self) -> None:
        """Создаёт временную папку для WAV-тестов."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        """Удаляет временную папку после теста."""
        self.temp_dir.cleanup()

    def test_read_wav_samples_validates_format_and_normalizes_pcm(self) -> None:
        """Проверяет параметры WAV и преобразование int16 в float."""
        path = self.root / "valid.wav"
        _write_wav(path)
        fake_numpy = _FakeNumpyForWav("numpy")
        with patch.dict(sys.modules, {"numpy": fake_numpy}):
            samples, rate = diarization._read_wav_samples(path)
        self.assertEqual(rate, 16000)
        self.assertEqual(samples, [0.0, 0.5, -0.5])

    def test_read_wav_samples_reports_dependency_file_and_format_errors(self) -> None:
        """Проверяет ошибки импорта numpy, чтения и параметров WAV."""
        with patch.dict(sys.modules, {"numpy": None}):
            with self.assertRaisesRegex(FattError, "numpy не установлен"):
                diarization._read_wav_samples(self.root / "missing.wav")

        bad_file = self.root / "bad.wav"
        bad_file.write_bytes(b"not wav")
        with patch.dict(sys.modules, {"numpy": _FakeNumpyForWav("numpy")}):
            with self.assertRaisesRegex(FattError, "Не удалось прочитать"):
                diarization._read_wav_samples(bad_file)

        for name, kwargs in (
            ("rate.wav", {"sample_rate": 8000}),
            ("channels.wav", {"channels": 2}),
        ):
            path = self.root / name
            _write_wav(path, **kwargs)
            with patch.dict(sys.modules, {"numpy": _FakeNumpyForWav("numpy")}):
                with self.assertRaisesRegex(FattError, "неожиданные параметры"):
                    diarization._read_wav_samples(path)


class TestDiarizationMath(unittest.TestCase):
    """Проверяет чистые функции кластеризации и сглаживания меток."""

    def test_cluster_features_handles_empty_and_singleton_without_numpy_math(self) -> None:
        """Проверяет быстрые ветви пустого и единственного признака."""
        fake_numpy = types.ModuleType("numpy")
        fake_numpy.float32 = object()
        fake_numpy.asarray = lambda features, dtype=None: features
        with patch.dict(sys.modules, {"numpy": fake_numpy}):
            self.assertEqual(diarization._cluster_features([], None), [])
            self.assertEqual(diarization._cluster_features([[1.0, 0.0]], None), [0])

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "требуется numpy из requirements.txt")
    def test_cluster_features_supports_fixed_and_automatic_clusters(self) -> None:
        """Проверяет k-means с заданным числом и автоматический порог."""
        features = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        self.assertEqual(diarization._cluster_features(features, 2), [0, 1, 0])
        self.assertEqual(diarization._cluster_features(features, None), [0, 1, 0])

    def test_smooth_labels_removes_only_isolated_spike(self) -> None:
        """Проверяет короткие последовательности и одиночный скачок метки."""
        self.assertEqual(diarization._smooth_labels([]), [])
        self.assertEqual(diarization._smooth_labels([1]), [1])
        self.assertEqual(diarization._smooth_labels([1, 2]), [1, 2])
        self.assertEqual(diarization._smooth_labels([0, 1, 0, 2]), [0, 0, 0, 2])
        self.assertEqual(diarization._smooth_labels([0, 1, 2]), [0, 1, 2])

    def test_audio_feature_delegates_to_feature_extractor(self) -> None:
        """Проверяет оптимизированную обёртку построения признака."""
        fake_extractor = types.SimpleNamespace(extract=lambda samples: ("feature", samples))
        with patch(
            "fatt_core.diarization.AcousticFeatureExtractor",
            return_value=fake_extractor,
        ) as extractor:
            result = diarization._audio_feature([1, 2], 16000)
        self.assertEqual(result, ("feature", [1, 2]))
        extractor.assert_called_once_with(16000)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "требуется numpy из requirements.txt")
    def test_feature_extractor_returns_normalized_vector(self) -> None:
        """Проверяет построение спектрального вектора для короткого сигнала."""
        import numpy as np

        extractor = diarization.AcousticFeatureExtractor(16000)
        feature = extractor.extract(np.zeros(100, dtype=np.float32))
        self.assertEqual(feature.shape, (52,))
        self.assertTrue(np.all(np.isfinite(feature)))


class TestDiarizeAudio(unittest.TestCase):
    """Проверяет граничные ошибки и сборку сегментов диаризации."""

    def test_diarize_audio_rejects_too_short_and_silent_audio(self) -> None:
        """Проверяет отсутствие пригодных окон и отсутствие речи."""
        fake_numpy = _FakeDiarizationNumpy(energy=0.0)
        with patch(
            "fatt_core.diarization._read_wav_samples",
            return_value=(_FakeSignal(100), 16000),
        ):
            with patch("fatt_core.diarization.AcousticFeatureExtractor"):
                with patch.dict(sys.modules, {"numpy": fake_numpy}):
                    with self.assertRaisesRegex(FattError, "слишком короткий"):
                        diarization.diarize_audio(Path("audio.wav"), "cpu")

        silent = _FakeSignal(16000 * 2)
        with patch(
            "fatt_core.diarization._read_wav_samples",
            return_value=(silent, 16000),
        ):
            with patch("fatt_core.diarization.AcousticFeatureExtractor"):
                with patch.dict(sys.modules, {"numpy": fake_numpy}):
                    with self.assertRaisesRegex(FattError, "не обнаружена речь"):
                        diarization.diarize_audio(Path("audio.wav"), "cpu")

    def test_diarize_audio_merges_adjacent_same_speaker_windows(self) -> None:
        """Проверяет формирование таймлайна и объединение соседних меток."""
        fake_numpy = _FakeDiarizationNumpy(energy=0.1)
        samples = _FakeSignal(16000 * 2)
        fake_extractor = MagicMock()
        fake_extractor.extract.side_effect = lambda _chunk: [1.0, 0.0]
        with patch(
            "fatt_core.diarization._read_wav_samples",
            return_value=(samples, 16000),
        ):
            with patch.dict(sys.modules, {"numpy": fake_numpy}):
                with patch(
                    "fatt_core.diarization.AcousticFeatureExtractor",
                    return_value=fake_extractor,
                ):
                    with patch(
                        "fatt_core.diarization._cluster_features",
                        return_value=[0, 0],
                    ):
                        segments = diarization.diarize_audio(Path("audio.wav"), "cpu")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].label, "LOCAL_00")
        self.assertGreater(segments[0].end, segments[0].start)


class _FakeSignal:
    """Минимальный вектор для теста окон диаризации без numpy."""

    def __init__(self, length: int) -> None:
        """Создаёт сигнал заданной длины."""
        self.length = length

    def __len__(self) -> int:
        """Возвращает число сэмплов."""
        return self.length

    def __getitem__(self, item: object) -> "_FakeSignal":
        """Возвращает срез сигнала с корректной длиной."""
        if isinstance(item, slice):
            start, stop, step = item.indices(self.length)
            return _FakeSignal(max(0, (stop - start + max(step, 1) - 1) // max(step, 1)))
        return _FakeSignal(1)

    def __pow__(self, _power: int) -> "_FakeSignal":
        """Имитирует возведение сэмплов в квадрат."""
        return self


class _FakeDiarizationNumpy(types.ModuleType):
    """Небольшая замена numpy для управления энергией тестового сигнала."""

    def __init__(self, energy: float) -> None:
        """Сохраняет энергию, которую должны увидеть окна."""
        super().__init__("numpy")
        self.energy = energy

    def pad(self, samples: _FakeSignal, padding: tuple[int, int]) -> _FakeSignal:
        """Имитирует дополнение сигнала нулевыми сэмплами."""
        return _FakeSignal(len(samples) + padding[0] + padding[1])

    def mean(self, _samples: _FakeSignal) -> float:
        """Возвращает тестовую энергию сигнала."""
        return self.energy

    def sqrt(self, value: float) -> float:
        """Имитирует квадратный корень для энергии."""
        return value

    def percentile(self, values: list[float], _percentile: int) -> float:
        """Возвращает нижний уровень энергии окон."""
        return min(values)
