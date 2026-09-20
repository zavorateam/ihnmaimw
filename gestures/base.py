#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations   # ← ВАЖНО: аннотации становятся ленивыми

from abc import ABC, abstractmethod
import logging
import math
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

# Pylance видит эти импорты в фазе type-check.
# В runtime при отсутствии модуля они будут None, но аннотации
# благодаря __future__.annotations не вычисляются.
if TYPE_CHECKING:
    from kinect_tracker import HandData, Joint, TrackingFrame
else:
    try:
        from kinect_tracker import HandData, Joint, TrackingFrame
    except ImportError:
        Joint = None            # type: ignore[assignment]
        HandData = None         # type: ignore[assignment]
        TrackingFrame = None    # type: ignore[assignment]

try:
    from kinect_tracker import HandData, Joint, TrackingFrame
except ImportError:
    Joint = None
    HandData = None
    TrackingFrame = None


# Конфигурация геометрии
SCREEN_W = 2560
SCREEN_H = 1600
WORKSPACE_WIDTH = 0.9
WORKSPACE_HEIGHT = 0.72
WORKSPACE_DEPTH_MIN_M = 0.35
WORKSPACE_DEPTH_MAX_M = 1.00


class GestureContext:
    """Контекст, передаваемый во все жесты для взаимодействия с системой."""
    def __init__(self, mouse, system_bridge, one_euro_filter, on_emergency_stop):
        self.mouse = mouse
        self.system_bridge = system_bridge
        self.filter = one_euro_filter
        self.on_emergency_stop = on_emergency_stop
        self.logger = logging.getLogger("GestureContext")

        # Общие переменные состояния HUD — с явными типами
        self.current_gesture: str = "CLUTCH (IDLE)"
        self.current_color: Tuple[int, int, int] = (130, 130, 130)
        self.cursor_pos: Tuple[int, int] = (SCREEN_W // 2, SCREEN_H // 2)
        self.pinch_val: float = 1.0
        self.overview_charge: float = 0.0
        self.fist_hold_ratio: float = 0.0

    @staticmethod
    def dist2d(j1: Joint, j2: Joint) -> float:
        return math.hypot(j1.nx - j2.nx, j1.ny - j2.ny)

    def get_palm_scale(self, joints: List[Joint]) -> float:
        return max(self.dist2d(joints[0], joints[9]), 0.04)

    def map_to_screen(self, nx: float, ny: float) -> Tuple[int, int]:
        x_min = (1.0 - WORKSPACE_WIDTH) / 2.0
        x_max = x_min + WORKSPACE_WIDTH
        y_min = (1.0 - WORKSPACE_HEIGHT) / 2.0
        y_max = y_min + WORKSPACE_HEIGHT
        clamped_x = max(x_min, min(x_max, nx))
        clamped_y = max(y_min, min(y_max, ny))

        norm_x = (clamped_x - x_min) / WORKSPACE_WIDTH
        norm_y = (clamped_y - y_min) / WORKSPACE_HEIGHT
        norm_x = 1.0 - norm_x

        return int(norm_x * SCREEN_W), int(norm_y * SCREEN_H)

    def is_in_box(self, hand: HandData) -> bool:
        if hand.depth_m > WORKSPACE_DEPTH_MAX_M:
            return False
        w = hand.joints[0]
        x_min = (1.0 - WORKSPACE_WIDTH) / 2.0
        x_max = x_min + WORKSPACE_WIDTH
        y_min = (1.0 - WORKSPACE_HEIGHT) / 2.0
        y_max = y_min + WORKSPACE_HEIGHT
        return (
            x_min <= w.nx <= x_max and
            y_min <= w.ny <= y_max and
            WORKSPACE_DEPTH_MIN_M <= hand.depth_m <= WORKSPACE_DEPTH_MAX_M
        )


class BaseGesture(ABC):
    """
    Базовый класс для всех плагинов жестов.
    """
    name: str = "BaseGesture"
    priority: int = 50  # Чем выше число, тем раньше жест обрабатывается в кадре

    def __init__(self, ctx: GestureContext):
        self.ctx = ctx
        self.logger = logging.getLogger(f"Gesture.{self.__class__.__name__}")

    @abstractmethod
    def process(self, frame: TrackingFrame, active_hands: List[HandData], dt: float) -> bool:
        """
        Обработка жеста.
        Возвращает True, если жест поглотил событие (остальные жесты с меньшим приоритетом пропускаются).
        """
        pass

    def reset(self):
        """Сброс внутреннего состояния жеста при потере рук."""
        pass