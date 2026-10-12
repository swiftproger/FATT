"""Формирование итогового отчёта по пакетной обработке файлов."""

from __future__ import annotations

from collections.abc import Sequence

from .console import color
from .models import ProcessingReport


def format_elapsed(seconds: float) -> str:
    """Форматирует длительность обработки в удобный вид."""
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes, remainder = divmod(seconds, 60)
    return f"{int(minutes)} мин {remainder:.1f} с"


def print_report(reports: Sequence[ProcessingReport], total_seconds: float) -> None:
    """Печатает отчёт по каждому файлу и общие итоги пакетного запуска."""
    separator = "=" * 72
    print(color(f"\n\n{separator}", "muted"))
    print(color("ОТЧЁТ О ВЫПОЛНЕНИИ", "heading"))
    print(color(separator, "muted"))
    for index, report in enumerate(reports, start=1):
        print(color(f"\nФАЙЛ {index} ИЗ {len(reports)}", "heading"))
        print(color("-" * 72, "muted"))
        print(color(f"Статус: {'УСПЕШНО' if report.success else 'ОШИБКА'}", "success" if report.success else "error"))
        print(f"Вход: {report.input_path}")
        print(f"Время обработки: {format_elapsed(report.elapsed_seconds)}")
        print(f"Количество слов: {report.word_count}")
        if report.output_path is not None:
            print(color(f"Результат: {report.output_path}", "path"))
        if report.error:
            print(color(f"Причина: {report.error}", "error"))
    successful = sum(report.success for report in reports)
    failed = len(reports) - successful
    total_words = sum(report.word_count for report in reports)
    print(color(f"\n{separator}", "muted"))
    print(color("ИТОГИ", "heading"))
    print(color(separator, "muted"))
    print(f"Файлов обработано: {len(reports)}")
    print(color(f"Успешно: {successful}", "success"))
    print(color(f"С ошибками: {failed}", "error" if failed else "muted"))
    print(f"Всего слов: {total_words}")
    print(f"Общее время: {format_elapsed(total_seconds)}")
