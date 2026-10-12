"""Цветовое оформление терминала без ANSI-кодов в перенаправленных логах."""

from __future__ import annotations

import os
import sys
from typing import TextIO


STYLES = {
    "heading": "1;36",
    "active": "33",
    "success": "32",
    "error": "1;31",
    "warning": "33",
    "path": "36",
    "muted": "90",
}


def supports_color(stream: TextIO) -> bool:
    """Включает цвета для терминала, учитывая NO_COLOR и TERM=dumb."""
    return (
        stream.isatty()
        and "NO_COLOR" not in os.environ
        and os.environ.get("TERM") != "dumb"
    )


def color(text: str, style: str, stream: TextIO | None = None) -> str:
    """Оформляет строку и сразу сбрасывает цвет следующих сообщений."""
    if not supports_color(sys.stdout if stream is None else stream):
        return text
    return f"\033[{STYLES[style]}m{text}\033[0m"
