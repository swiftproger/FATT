"""Архитектурные тесты структуры исходного кода и документации FATT."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = PROJECT_ROOT / "fatt_core"


class TestArchitecture(unittest.TestCase):
    """Проверяет правила, поддерживающие модульность и документированность FATT."""

    def test_core_package_contains_expected_modules(self) -> None:
        """Проверяет наличие отдельных модулей для основных обязанностей приложения."""
        expected_modules = {
            "cli.py",
            "device.py",
            "diarization.py",
            "errors.py",
            "media.py",
            "models.py",
            "output.py",
            "pipeline.py",
            "progress.py",
            "report.py",
            "transcription.py",
        }
        actual_modules = {path.name for path in CORE_ROOT.glob("*.py")}
        self.assertTrue(
            expected_modules.issubset(actual_modules),
            f"Не хватает модулей: {sorted(expected_modules - actual_modules)}",
        )

    def test_each_production_area_has_a_dedicated_test_module(self) -> None:
        """Проверяет, что каждый слой проекта закреплён за тестовым модулем."""
        required_tests = {
            "cli": "test_cli.py",
            "device": "test_device.py",
            "diarization": "test_diarization.py",
            "errors": "test_errors.py",
            "media": "test_media.py",
            "models": "test_output_report.py",
            "output": "test_output_report.py",
            "pipeline": "test_pipeline.py",
            "progress": "test_progress.py",
            "report": "test_output_report.py",
            "transcription": "test_processing.py",
        }
        test_files = {path.name for path in (PROJECT_ROOT / "tests").glob("test_*.py")}
        missing = {
            module: test_file
            for module, test_file in required_tests.items()
            if test_file not in test_files
        }
        self.assertEqual(missing, {}, f"Для модулей нет тестов: {missing}")

    def test_optional_runtime_dependencies_are_lazy_imported(self) -> None:
        """Проверяет, что тяжёлые зависимости не загружаются при импорте пакета."""
        optional_modules = {
            "torch",
            "numpy",
            "ffmpeg",
            "whisper",
            "tqdm",
            "prompt_toolkit",
        }
        violations: list[str] = []
        for source_file in sorted(CORE_ROOT.glob("*.py")):
            tree = ast.parse(source_file.read_text(encoding="utf-8"), str(source_file))
            for node in tree.body:
                imported_names: list[str] = []
                if isinstance(node, ast.Import):
                    imported_names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_names = [node.module.split(".")[0]]
                for imported_name in imported_names:
                    if imported_name in optional_modules:
                        violations.append(f"{source_file.name}: {imported_name}")
        self.assertEqual(
            violations,
            [],
            "Опциональные зависимости должны импортироваться внутри функций: "
            + ", ".join(violations),
        )

    def test_dependency_manifest_contains_runtime_packages(self) -> None:
        """Проверяет наличие библиотек, требуемых основными слоями приложения."""
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
        for package in ("torch", "numpy", "openai-whisper", "ffmpeg-python", "tqdm"):
            self.assertIn(package, requirements)

    def test_every_production_class_and_function_has_docstring(self) -> None:
        """Проверяет наличие docstring у каждого класса и функции в рабочем коде."""
        source_files = [PROJECT_ROOT / "fatt.py", *sorted(CORE_ROOT.glob("*.py"))]
        missing: list[str] = []
        for source_file in source_files:
            tree = ast.parse(source_file.read_text(encoding="utf-8"), str(source_file))
            for node in ast.walk(tree):
                if not isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                if ast.get_docstring(node) is None:
                    missing.append(f"{source_file.relative_to(PROJECT_ROOT)}:{node.name}")
        self.assertEqual(
            missing,
            [],
            "У классов/функций отсутствует docstring: " + ", ".join(missing),
        )

    def test_production_docstrings_are_in_russian(self) -> None:
        """Проверяет наличие кириллицы во всех docstring производственного кода."""
        missing: list[str] = []
        for source_file in [PROJECT_ROOT / "fatt.py", *sorted(CORE_ROOT.glob("*.py"))]:
            tree = ast.parse(source_file.read_text(encoding="utf-8"), str(source_file))
            nodes = [tree, *ast.walk(tree)]
            for node in nodes:
                if not isinstance(
                    node,
                    (
                        ast.Module,
                        ast.ClassDef,
                        ast.FunctionDef,
                        ast.AsyncFunctionDef,
                    ),
                ):
                    continue
                documentation = ast.get_docstring(node) or ""
                if not re.search(r"[А-Яа-яЁё]", documentation):
                    name = getattr(node, "name", "модуль")
                    missing.append(f"{source_file.relative_to(PROJECT_ROOT)}:{name}")
        self.assertEqual(
            missing,
            [],
            "В docstring отсутствует русский текст: " + ", ".join(missing),
        )

    def test_root_entrypoint_delegates_to_core_package(self) -> None:
        """Проверяет, что fatt.py остаётся тонким запускателем без бизнес-логики."""
        source = (PROJECT_ROOT / "fatt.py").read_text(encoding="utf-8")
        self.assertIn("from fatt_core.cli import main", source)
        self.assertNotIn("def ", source)
