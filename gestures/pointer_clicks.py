#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, Tuple
import numpy as np

from .base import BaseGesture, GestureContext

PINCH_LOCK_ENTER = 0.36
PINCH_CLICK_DOWN = 0.25
PINCH_CLICK_UP = 0.42


class PointerClicksGesture(BaseGesture):
    name = "Pointer & Clicks"
    priority = 10  # Базовый жест движения и клика

    def __init__(self, ctx: GestureContext):
        super().__init__(ctx)
        self.is_left_down = False
        self.is_right_down = False
        self.is_middle_down = False
        self.is_pinch_locked = False
        self.locked_pos = (0, 0)
        self.corner_fail_start_ts: Optional[float] = None

    def reset(self):
        self.ctx.mouse.release_all()
        self.ctx.filter.reset()
        self.is_left_down = False
        self.is_right_down = False
        self.is_middle_down = False
        self.is_pinch_locked = False
        self.corner_fail_start_ts = None

    def process(self, frame, active_hands, dt: float) -> bool:
        if not active_hands:
            self.reset()
            return False

        hand = active_hands[0]
        joints = hand.joints
        scale = self.ctx.get_palm_scale(joints)
        now = frame.timestamp

        thumb_tip = joints[4]
        index_tip = joints[8]
        middle_tip = joints[12]
        ring_tip = joints[16]

        index_thumb_dist = self.ctx.dist2d(thumb_tip, index_tip) / scale
        middle_thumb_dist = self.ctx.dist2d(middle_tip, thumb_tip) / scale
        ring_thumb_dist = self.ctx.dist2d(ring_tip, thumb_tip) / scale
        self.ctx.pinch_val = index_thumb_dist

        raw_x, raw_y = self.ctx.map_to_screen(index_tip.nx, index_tip.ny)
        filtered_pt = self.ctx.filter.filter(np.array([raw_x, raw_y], dtype=np.float32), now)
        smooth_x, smooth_y = int(filtered_pt[0]), int(filtered_pt[1])

        # Corner Fail-Safe
        if smooth_x < 15 and smooth_y < 15:
            if self.corner_fail_start_ts is None:
                self.corner_fail_start_ts = now
            elif now - self.corner_fail_start_ts > 1.2:
                self.logger.critical("KILLER SWITCH: КУРСОР В УГЛУ ЭКРАНА!")
                self.ctx.on_emergency_stop()
                return True
        else:
            self.corner_fail_start_ts = None

        # Pinch-Lock прицеливания
        if index_thumb_dist < PINCH_LOCK_ENTER:
            if not self.is_pinch_locked:
                self.is_pinch_locked = True
                self.locked_pos = (smooth_x, smooth_y)
            final_x, final_y = self.locked_pos
        else:
            self.is_pinch_locked = False
            final_x, final_y = smooth_x, smooth_y

        self.ctx.cursor_pos = (final_x, final_y)
        self.ctx.mouse.move_abs(final_x, final_y)

        # Обработка ЛКМ
        if not self.is_left_down:
            if index_thumb_dist < PINCH_CLICK_DOWN:
                self.is_left_down = True
                self.ctx.mouse.set_left_btn(True)
        else:
            if index_thumb_dist > PINCH_CLICK_UP:
                self.is_left_down = False
                self.ctx.mouse.set_left_btn(False)
                self.is_pinch_locked = False

        # Обработка СКМ
        if not self.is_middle_down:
            if ring_thumb_dist < PINCH_CLICK_DOWN:
                self.is_middle_down = True
                self.ctx.mouse.set_middle_btn(True)
        else:
            if ring_thumb_dist > PINCH_CLICK_UP:
                self.is_middle_down = False
                self.ctx.mouse.set_middle_btn(False)

        # Обработка ПКМ
        if not self.is_right_down:
            if middle_thumb_dist < PINCH_CLICK_DOWN:
                self.is_right_down = True
                self.ctx.mouse.set_right_btn(True)
        else:
            if middle_thumb_dist > PINCH_CLICK_UP:
                self.is_right_down = False
                self.ctx.mouse.set_right_btn(False)

        # Обновление HUD
        if self.is_left_down:
            self.ctx.current_gesture = "🔴 LKM: CLICK / DRAG"
            self.ctx.current_color = (0, 0, 255)
        elif self.is_middle_down:
            self.ctx.current_gesture = "🟢 SKM: MIDDLE CLICK"
            self.ctx.current_color = (0, 255, 100)
        elif self.is_right_down:
            self.ctx.current_gesture = "🔵 PKM: CONTEXT MENU"
            self.ctx.current_color = (255, 120, 0)
        elif self.is_pinch_locked:
            self.ctx.current_gesture = "🎯 PINCH LOCK (AIM FREEZE)"
            self.ctx.current_color = (0, 255, 255)
        else:
            self.ctx.current_gesture = "🟢 POINTER: ACTIVE"
            self.ctx.current_color = (0, 255, 0)

        return True