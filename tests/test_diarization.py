"""Проверки чтения WAV, признаков и локальной диаризации."""

from __future__ import annotations

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

    def test_cluster_features_supports_fixed_and_automatic_clusters(self) -> None:
        """Проверяет k-means с заданным числом и автоматический порог."""
        fake_numpy = _FakeClusterNumpy()
        features = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        with patch.dict(sys.modules, {"numpy": fake_numpy}):
            self.assertEqual(diarization._cluster_features(features, 2), [0, 1, 1])
            self.assertEqual(diarization._cluster_features(features, None), [0, 1, 1])

        with patch.dict(
            sys.modules,
            {"numpy": _FakeClusterNumpy(allclose_result=False)},
        ):
            self.assertEqual(
                diarization._cluster_features([[1.0, 0.0], [0.0, 1.0]], 2),
                [0, 1],
            )
            self.assertEqual(
                diarization._cluster_features([[1.0, 0.0], [1.0, 0.0]], 3),
                [0, 0],
            )

        one_hot = [
            [1.0 if column == row else 0.0 for column in range(9)]
            for row in range(9)
        ]
        with patch.dict(sys.modules, {"numpy": _FakeClusterNumpy()}):
            labels = diarization._cluster_features(one_hot, None)
        self.assertEqual(labels[:8], list(range(8)))
        self.assertEqual(labels[8], 0)

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

    def test_feature_extractor_returns_normalized_vector(self) -> None:
        """Проверяет построение спектрального вектора для короткого сигнала."""
        fake_numpy = _FakeFeatureNumpy()
        with patch.dict(sys.modules, {"numpy": fake_numpy}):
            extractor = diarization.AcousticFeatureExtractor(16000)
            feature = extractor.extract([0.0] * 100)
        self.assertEqual(feature.shape, (52,))
        self.assertEqual(feature.value, "normalized")

        with patch.dict(sys.modules, {"numpy": _FakeFeatureNumpy(empty_bands=True)}):
            extractor = diarization.AcousticFeatureExtractor(16000)
            feature = extractor.extract([0.0] * 100)
        self.assertEqual(feature.shape, (52,))


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


class _MiniArray:
    """Небольшой двумерный массив для тестирования k-means без numpy."""

    def __init__(self, data: object) -> None:
        """Копирует скаляры, строки и матрицы в простой список."""
        if isinstance(data, _MiniArray):
            data = data.data
        if isinstance(data, list):
            self.data = [
                list(item.data) if isinstance(item, _MiniArray) else item
                for item in data
            ]
        else:
            self.data = data

    @property
    def shape(self) -> tuple[int, ...]:
        """Возвращает размерность тестового массива."""
        if not isinstance(self.data, list):
            return ()
        if self.data and isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    @property
    def T(self) -> "_MiniArray":
        """Возвращает транспонированную матрицу."""
        return _MiniArray([list(column) for column in zip(*self.data)])

    def __len__(self) -> int:
        """Возвращает длину первого измерения."""
        return len(self.data)

    def __iter__(self):
        """Итерирует строки матрицы как одномерные массивы."""
        if self.data and isinstance(self.data[0], list):
            return iter([_MiniArray(item) for item in self.data])
        return iter(self.data)

    def __getitem__(self, key: object) -> object:
        """Поддерживает индексы строк, маски и срезы столбцов."""
        if isinstance(key, tuple):
            rows, columns = key
            selected_rows = self[rows]
            if not isinstance(selected_rows, _MiniArray):
                selected_rows = _MiniArray([selected_rows])
            row_data = selected_rows.data
            if not row_data or not isinstance(row_data[0], list):
                row_data = [row_data]
            if isinstance(columns, slice):
                result = [row[columns] for row in row_data]
            else:
                mask = columns.data if isinstance(columns, _MiniArray) else columns
                result = [[value for value, keep in zip(row, mask) if keep] for row in row_data]
            return _MiniArray(result)
        if isinstance(key, _MiniArray):
            key = key.data
        if isinstance(key, list):
            if key and all(isinstance(value, bool) for value in key):
                return _MiniArray([row for row, keep in zip(self.data, key) if keep])
            return _MiniArray([self.data[index] for index in key])
        value = self.data[key]  # type: ignore[index]
        return _MiniArray(value) if isinstance(value, list) else value

    def copy(self) -> "_MiniArray":
        """Возвращает независимую копию массива."""
        return _MiniArray(self.data)

    def _binary(self, other: object, operation) -> "_MiniArray":
        """Применяет бинарную операцию со скаляром или массивом."""
        right = other.data if isinstance(other, _MiniArray) else other

        def apply(left_value: object, right_value: object) -> object:
            if isinstance(left_value, list):
                if isinstance(right_value, list):
                    if right_value and isinstance(right_value[0], list):
                        return [
                            apply(left_item, right_item)
                            for left_item, right_item in zip(left_value, right_value)
                        ]
                    if len(right_value) == 1:
                        return [apply(left_item, right_value[0]) for left_item in left_value]
                    return [
                        apply(left_item, right_item)
                        for left_item, right_item in zip(left_value, right_value)
                    ]
                return [apply(left_item, right_value) for left_item in left_value]
            return operation(left_value, right_value)

        return _MiniArray(apply(self.data, right))

    def __matmul__(self, other: "_MiniArray") -> object:
        """Вычисляет скалярное или матричное произведение."""
        if self.shape and len(self.shape) == 1 and len(other.shape) == 1:
            return sum(left * right for left, right in zip(self.data, other.data))
        return _MiniArray(
            [
                [sum(left * right for left, right in zip(row, column)) for column in zip(*other.data)]
                for row in self.data
            ]
        )

    def __mul__(self, other: object) -> "_MiniArray":
        """Умножает массив на скаляр или массив."""
        return self._binary(other, lambda left, right: left * right)

    __rmul__ = __mul__

    def __add__(self, other: object) -> "_MiniArray":
        """Складывает массив со скаляром или массивом."""
        return self._binary(other, lambda left, right: left + right)

    def __truediv__(self, other: object) -> "_MiniArray":
        """Делит массив на скаляр или массив с broadcasting по строкам."""
        return self._binary(other, lambda left, right: left / right)

    def __eq__(self, other: object) -> "_MiniArray":
        """Сравнивает элементы массива со значением."""
        return self._binary(other, lambda left, right: left == right)

    def mean(self, axis: int | None = None) -> "_MiniArray | float":
        """Вычисляет среднее значение для нужной оси."""
        if axis is None:
            values = self.data if self.data and not isinstance(self.data[0], list) else sum(self.data, [])
            return sum(values) / len(values)
        if axis == 0:
            return _MiniArray(
                [sum(row[index] for row in self.data) / len(self.data) for index in range(len(self.data[0]))]
            )
        return _MiniArray([sum(row) / len(row) for row in self.data])

    def std(self, axis: int | None = None) -> "_MiniArray | float":
        """Возвращает нулевое стандартное отклонение для stub-массива."""
        del axis
        return _MiniArray([0.0] * (len(self.data[0]) if self.data and isinstance(self.data[0], list) else len(self.data)))

    def sum(self, axis: int | None = None) -> "_MiniArray | float":
        """Вычисляет сумму по оси."""
        if axis == 1:
            return _MiniArray([sum(row) for row in self.data])
        return sum(self.data)

    def astype(self, _dtype: object) -> "_MiniArray":
        """Имитирует преобразование типа."""
        return self

    def tolist(self) -> list[object]:
        """Возвращает обычный список значений."""
        return list(self.data)


class _FakeClusterNumpy(types.ModuleType):
    """Реализует только операции numpy, нужные для проверки кластеризации."""

    float32 = object()
    int = int

    def __init__(self, *, allclose_result: bool = True) -> None:
        """Создаёт заглушку numpy."""
        super().__init__("numpy")
        self.allclose_result = allclose_result
        self.linalg = types.SimpleNamespace(norm=self._norm)

    def asarray(self, values: object, dtype: object = None) -> _MiniArray:
        """Преобразует значения в тестовый массив."""
        del dtype
        return _MiniArray(values)

    def array(self, values: object, dtype: object = None) -> _MiniArray:
        """Создаёт тестовый массив."""
        del dtype
        return _MiniArray(values)

    def linspace(self, start: int, end: int, count: int, dtype: object = None) -> list[int]:
        """Создаёт равномерные индексы начальных центров."""
        del dtype
        if count == 1:
            return [int(start)]
        return [round(start + index * (end - start) / (count - 1)) for index in range(count)]

    def zeros(self, count: int, dtype: object = None) -> _MiniArray:
        """Создаёт нулевой вектор меток."""
        del dtype
        return _MiniArray([0] * count)

    def argmax(self, values: object, axis: int | None = None) -> object:
        """Возвращает индекс максимального значения."""
        if axis == 1:
            matrix = values.data if isinstance(values, _MiniArray) else values
            return _MiniArray([max(range(len(row)), key=lambda index: row[index]) for row in matrix])
        return max(range(len(values)), key=lambda index: values[index])

    def any(self, values: _MiniArray) -> bool:
        """Проверяет наличие истинного элемента."""
        return any(values.data)

    def maximum(self, left: _MiniArray, right: float) -> _MiniArray:
        """Ограничивает нормы снизу."""
        return _MiniArray([[max(value[0], right)] for value in left.data])

    def allclose(self, left: _MiniArray, right: _MiniArray, **_kwargs: object) -> bool:
        """Сравнивает массивы тестовых значений."""
        del left, right
        return self.allclose_result

    def _norm(self, values: _MiniArray, axis: int | None = None, keepdims: bool = False) -> object:
        """Вычисляет евклидову норму вектора или строк матрицы."""
        if axis == 1:
            result = [[sum(value * value for value in row) ** 0.5] for row in values.data]
            return _MiniArray(result) if keepdims else _MiniArray([row[0] for row in result])
        return sum(value * value for value in values.data) ** 0.5


class _FakeFeatureValue:
    """Значение-заглушка для прохождения FFT-конвейера экстрактора."""

    shape = (1, 1)

    def __init__(self, value: str = "value", *, size: int = 1) -> None:
        """Сохраняет метку операции."""
        self.value = value
        self.size = size

    def __getitem__(self, _key: object) -> "_FakeFeatureValue":
        """Возвращает значение для любого среза."""
        return self

    def __mul__(self, _other: object) -> "_FakeFeatureValue":
        """Имитирует умножение массивов."""
        return self

    __rmul__ = __mul__

    def __pow__(self, _power: int) -> "_FakeFeatureValue":
        """Имитирует возведение в степень."""
        return self

    def __truediv__(self, _other: object) -> "_FakeFeatureValue":
        """Имитирует деление признаков."""
        return self

    def __add__(self, _other: object) -> "_FakeFeatureValue":
        """Имитирует сложение признаков."""
        return self

    def __radd__(self, _other: object) -> "_FakeFeatureValue":
        """Имитирует сложение с числовым нулём."""
        return self

    def __lt__(self, _other: object) -> "_FakeFeatureValue":
        """Имитирует сравнение матриц."""
        return self

    def __ge__(self, _other: object) -> "_FakeFeatureValue":
        """Имитирует сравнение частот."""
        return self

    def __and__(self, _other: object) -> "_FakeFeatureValue":
        """Имитирует объединение масок частот."""
        return self

    def __len__(self) -> int:
        """Возвращает размер первой оси."""
        return 1

    def astype(self, _dtype: object) -> "_FakeFeatureValue":
        """Имитирует преобразование типа."""
        return self

    def mean(self, axis: int | None = None) -> "_FakeFeatureValue":
        """Имитирует среднее значение."""
        del axis
        return self

    def std(self, axis: int | None = None) -> "_FakeFeatureValue":
        """Имитирует стандартное отклонение."""
        del axis
        return self

    def sum(self, axis: int | None = None) -> "_FakeFeatureValue":
        """Имитирует сумму."""
        del axis
        return self


class _FakeFeatureNumpy(types.ModuleType):
    """Реализует формы и операции, нужные FFT-экстрактору."""

    float32 = object()

    def __init__(self, *, empty_bands: bool = False) -> None:
        """Создаёт заглушки пространств numpy."""
        super().__init__("numpy")
        value = _FakeFeatureValue(size=0 if empty_bands else 1)
        self.fft = types.SimpleNamespace(
            rfftfreq=lambda *_args: value,
            rfft=lambda *_args, **_kwargs: value,
        )
        self.lib = types.SimpleNamespace(
            stride_tricks=types.SimpleNamespace(
                sliding_window_view=lambda *_args, **_kwargs: value,
            )
        )
        self.linalg = types.SimpleNamespace(norm=lambda *_args, **_kwargs: 1.0)

    def hanning(self, _length: int) -> _FakeFeatureValue:
        """Возвращает тестовое оконное значение."""
        return _FakeFeatureValue()

    def geomspace(self, start: float, end: float, count: int) -> list[float]:
        """Возвращает монотонные границы частот."""
        return [start + (end - start) * index / (count - 1) for index in range(count)]

    def pad(self, _samples: object, _padding: object) -> _FakeFeatureValue:
        """Имитирует дополнение короткого сигнала."""
        return _FakeFeatureValue()

    def abs(self, value: _FakeFeatureValue) -> _FakeFeatureValue:
        """Имитирует модуль спектра."""
        return value

    def log1p(self, value: _FakeFeatureValue) -> _FakeFeatureValue:
        """Имитирует логарифм мощности."""
        return value

    def zeros(self, _shape: object, dtype: object = None) -> _FakeFeatureValue:
        """Возвращает пустую полосу частот."""
        del dtype
        return _FakeFeatureValue()

    def stack(self, _values: object, axis: int = 0) -> _FakeFeatureValue:
        """Имитирует сборку матрицы полос."""
        del axis
        return _FakeFeatureValue()

    def concatenate(self, _values: object) -> _FakeFeatureValue:
        """Возвращает итоговый вектор признака заданной длины."""
        value = _FakeFeatureValue("normalized")
        value.shape = (52,)
        return value
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
