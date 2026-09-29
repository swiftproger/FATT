"""Формирование итогового отчёта по пакетной обработке файлов."""

from __future__ import annotations

from collections.abc import Sequence

from .models import ProcessingReport


def format_elapsed(seconds: float) -> str:
    """Форматирует длительность обработки в удобный вид."""
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)} мин {remainder:.1f} с"


def print_report(reports: Sequence[ProcessingReport], total_seconds: float) -> None:
    """Печатает отчёт по каждому файлу и общие итоги пакетного запуска."""
    print("\nОтчёт о выполнении")
    print("=" * 72)
    for report in reports:
        status = "УСПЕШНО" if report.success else "ОШИБКА"
        print(f"{status}: {report.input_path}")
        print(f"  Время обработки: {format_elapsed(report.elapsed_seconds)}")
        print(f"  Слов: {report.word_count}")
        if report.output_path is not None:
            print(f"  Результат: {report.output_path}")
        if report.error:
            print(f"  Причина: {report.error}")
    successful = sum(report.success for report in reports)
    failed = len(reports) - successful
    total_words = sum(report.word_count for report in reports)
    print("-" * 72)
    print(f"Файлов обработано: {len(reports)}")
    print(f"Успешно: {successful}; с ошибками: {failed}")
    print(f"Всего слов: {total_words}")
    print(f"Общее время: {format_elapsed(total_seconds)}")
