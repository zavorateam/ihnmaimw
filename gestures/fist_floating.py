#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import Optional
from .base import BaseGesture, GestureContext


class FistFloatingGesture(BaseGesture):
    name = "Fist Floating Toggle"
    priority = 80

    def __init__(self, ctx: GestureContext):
        super().__init__(ctx)
        self.fist_start_ts: Optional[float] = None
        self.fist_latched = False

    def reset(self):
        self.fist_start_ts = None
        self.fist_latched = False
        self.ctx.fist_hold_ratio = 0.0

    def _is_fist_anatomical(self, joints, scale: float) -> bool:
        wrist = joints[0]
        if self.ctx.dist2d(joints[8], wrist) > self.ctx.dist2d(joints[6], wrist) * 1.08:
            return False

        pc_x = (joints[0].nx + joints[9].nx) / 2.0
        pc_y = (joints[0].ny + joints[9].ny) / 2.0

        folded = 0
        for tip_idx in (8, 12, 16, 20):
            d = math.hypot(joints[tip_idx].nx - pc_x, joints[tip_idx].ny - pc_y) / scale
            if d < 0.48:
                folded += 1

        thumb_d = math.hypot(joints[4].nx - pc_x, joints[4].ny - pc_y) / scale
        return (folded == 4) and (thumb_d < 0.65)

    def process(self, frame, active_hands, dt: float) -> bool:
        if not active_hands:
            self.reset()
            return False

        hand = active_hands[0]
        scale = self.ctx.get_palm_scale(hand.joints)

        if self._is_fist_anatomical(hand.joints, scale):
            self.ctx.mouse.release_all()
            now = frame.timestamp

            if not self.fist_latched:
                if self.fist_start_ts is None:
                    self.fist_start_ts = now
                elapsed = now - self.fist_start_ts
                self.ctx.fist_hold_ratio = min(1.0, elapsed / 0.35)

                if elapsed >= 0.35:
                    self.ctx.system_bridge.toggle_floating()
                    self.fist_latched = True
                    self.fist_start_ts = None
                    self.ctx.current_gesture = "📌 TOGGLED FLOATING WINDOW!"
                    self.ctx.current_color = (255, 255, 0)
                    return True
                else:
                    self.ctx.current_gesture = f"✊ FIST HOLDING [{int(self.ctx.fist_hold_ratio * 100)}%]"
                    self.ctx.current_color = (200, 200, 0)
                    return True
            else:
                self.ctx.current_gesture = "✊ FIST ACTIVE"
                self.ctx.current_color = (180, 180, 0)
                return True
        else:
            self.reset()
            return False