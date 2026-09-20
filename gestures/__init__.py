#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib
import inspect
import logging
import os
import pkgutil
from typing import List

from .base import BaseGesture, GestureContext

logger = logging.getLogger("GestureLoader")


def load_gestures(ctx: GestureContext) -> List[BaseGesture]:
    """
    Автоматически находит, импортирует и инстанцирует все модули жестов из папки gestures/.
    """
    gestures: List[BaseGesture] = []
    pkg_dir = os.path.dirname(__file__)

    for _, module_name, is_pkg in pkgutil.iter_modules([pkg_dir]):
        if module_name in ("base", "__main__") or is_pkg:
            continue

        try:
            full_module_name = f"gestures.{module_name}"
            module = importlib.import_module(full_module_name)

            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    inspect.isclass(attr) and
                    issubclass(attr, BaseGesture) and
                    attr is not BaseGesture and
                    not inspect.isabstract(attr)
                ):
                    instance = attr(ctx)
                    gestures.append(instance)
                    logger.info("Загружен плагин жеста: %s (Priority: %d) из %s",
                                instance.name, instance.priority, module_name)

        except Exception as e:
            logger.error("Ошибка при динамической загрузке жеста из '%s': %s", module_name, e, exc_info=True)

    # Сортировка по приоритету (от большего к меньшему)
    gestures.sort(key=lambda g: g.priority, reverse=True)
    logger.info("Всего успешно подключено плагинов жестов: %d", len(gestures))
    return gestures