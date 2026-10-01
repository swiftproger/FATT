"""Проверки доменного исключения FATT."""

from __future__ import annotations

import unittest

from fatt_core.errors import FattError


class TestFattError(unittest.TestCase):
    """Проверяет тип исключения, используемый для ожидаемых ошибок."""

    def test_fatt_error_is_runtime_error_with_message(self) -> None:
        """Проверяет наследование и сохранение сообщения пользователя."""
        error = FattError("ошибка теста")
        self.assertIsInstance(error, RuntimeError)
        self.assertEqual(str(error), "ошибка теста")
