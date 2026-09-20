#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional, Tuple
from .base import BaseGesture, GestureContext

SCROLL_FINGER_SPLIT_THRESH = 0.50
SCROLL_STEP_Y = 0.011
SCROLL_STEP_X = 0.014
PINCH_LOCK_ENTER = 0.36


class KineticScrollGesture(BaseGesture):
    name = "Kinetic Scroll (V-Sign)"
    priority = 70

    def __init__(self, ctx: GestureContext):
        super().__init__(ctx)
        self.is_in_scroll_mode = False
        self.scroll_anchor: Optional[Tuple[float, float]] = None

    def reset(self):
        self.is_in_scroll_mode = False
        self.scroll_anchor = None

    def _is_peace_v_sign(self, joints, scale: float) -> bool:
        pc_x = (joints[0].nx + joints[9].nx) / 2.0
        pc_y = (joints[0].ny + joints[9].ny) / 2.0

        d_index_pc = math.hypot(joints[8].nx - pc_x, joints[8].ny - pc_y) / scale
        d_middle_pc = math.hypot(joints[12].nx - pc_x, joints[12].ny - pc_y) / scale
        if d_index_pc < 0.85 or d_middle_pc < 0.85:
            return False

        d_ring_pc = math.hypot(joints[16].nx - pc_x, joints[16].ny - pc_y) / scale
        d_pinky_pc = math.hypot(joints[20].nx - pc_x, joints[20].ny - pc_y) / scale
        if d_ring_pc > 0.60 or d_pinky_pc > 0.60:
            return False

        middle_ring_split = self.ctx.dist2d(joints[12], joints[16]) / scale
        if middle_ring_split < SCROLL_FINGER_SPLIT_THRESH:
            return False

        if (self.ctx.dist2d(joints[4], joints[8]) / scale < 0.35 or
                self.ctx.dist2d(joints[4], joints[12]) / scale < 0.35):
            return False

        return True

    def process(self, frame, active_hands, dt: float) -> bool:
        if not active_hands:
            self.reset()
            return False

        joints = active_hands[0].joints
        scale = self.ctx.get_palm_scale(joints)
        is_v_sign = self._is_peace_v_sign(joints, scale)

        index_tip = joints[8]
        middle_tip = joints[12]

        if not self.is_in_scroll_mode:
            if is_v_sign:
                self.is_in_scroll_mode = True
                self.scroll_anchor = ((index_tip.nx + middle_tip.nx) / 2.0,
                                      (index_tip.ny + middle_tip.ny) / 2.0)
        else:
            wrist = joints[0]
            index_open = self.ctx.dist2d(joints[8], wrist) > self.ctx.dist2d(joints[6], wrist) * 1.05
            middle_open = self.ctx.dist2d(joints[12], wrist) > self.ctx.dist2d(joints[10], wrist) * 1.05
            has_pinch = (self.ctx.dist2d(joints[4], joints[8]) / scale < PINCH_LOCK_ENTER)
            middle_ring_split = self.ctx.dist2d(joints[12], joints[16]) / scale

            if (not (index_open and middle_open)) or has_pinch or (middle_ring_split < (SCROLL_FINGER_SPLIT_THRESH - 0.08)):
                self.reset()
                return False

        if self.is_in_scroll_mode:
            self.ctx.mouse.release_all()
            self.ctx.current_gesture = "📜 SCROLLING (V-SIGN LOCKED)"
            self.ctx.current_color = (0, 220, 255)

            v_x = (index_tip.nx + middle_tip.nx) / 2.0
            v_y = (index_tip.ny + middle_tip.ny) / 2.0

            if self.scroll_anchor is None:
                self.scroll_anchor = (v_x, v_y)
            else:
                dy = v_y - self.scroll_anchor[1]
                dx = -(v_x - self.scroll_anchor[0])

                if abs(dy) >= SCROLL_STEP_Y:
                    steps_v = int(dy / SCROLL_STEP_Y)
                    self.ctx.mouse.scroll_v(-steps_v)
                    self.scroll_anchor = (self.scroll_anchor[0], self.scroll_anchor[1] + steps_v * SCROLL_STEP_Y)

                if abs(dx) >= SCROLL_STEP_X:
                    steps_h = int(dx / SCROLL_STEP_X)
                    self.ctx.mouse.scroll_h(steps_h)
                    self.scroll_anchor = (self.scroll_anchor[0] - steps_h * SCROLL_STEP_X, self.scroll_anchor[1])

            return True

        return False