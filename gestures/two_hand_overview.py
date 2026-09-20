#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import List, Optional, Tuple

from .base import BaseGesture, GestureContext

PINCH_CLICK_DOWN = 0.25


class TwoHandOverviewGesture(BaseGesture):
    name = "Two-Hand Overview"
    priority = 100

    def __init__(self, ctx: GestureContext):
        super().__init__(ctx)
        self.is_in_overview = False
        self.overview_charge = 0.0
        self.overview_latched = False
        self.overview_cooldown = 0.0
        self.steer_origin: Optional[Tuple[float, float]] = None
        self.steer_step_cooldown = 0.0

    def _is_open_palm(self, joints) -> bool:
        wrist = joints[0]
        straight = 0
        for tip_i, pip_i in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if self.ctx.dist2d(joints[tip_i], wrist) > self.ctx.dist2d(joints[pip_i], wrist) * 1.10:
                straight += 1
        if self.ctx.dist2d(joints[4], joints[17]) > self.ctx.dist2d(joints[2], joints[17]):
            straight += 1
        return straight >= 4

    def reset(self):
        if self.is_in_overview:
            self.ctx.system_bridge.toggle_overview()
        self.is_in_overview = False
        self.overview_charge = 0.0
        self.overview_latched = False
        self.steer_origin = None
        self.ctx.overview_charge = 0.0

    def process(self, frame, active_hands, dt: float) -> bool:
        now = frame.timestamp

        if len(active_hands) < 2:
            self.steer_origin = None
            self.overview_charge = max(0.0, self.overview_charge - dt * 2.5)
            self.ctx.overview_charge = self.overview_charge
            if self.is_in_overview:
                self.ctx.system_bridge.toggle_overview()
                self.is_in_overview = False
            return False

        sorted_h = sorted(active_hands, key=lambda h: h.joints[0].nx)
        h_l, h_r = sorted_h[0], sorted_h[1]

        sz_l = self.ctx.get_palm_scale(h_l.joints)
        sz_r = self.ctx.get_palm_scale(h_r.joints)
        p_l = (self.ctx.dist2d(h_l.joints[4], h_l.joints[8]) / sz_l) < PINCH_CLICK_DOWN
        p_r = (self.ctx.dist2d(h_r.joints[4], h_r.joints[8]) / sz_r) < PINCH_CLICK_DOWN

        is_l_open = self._is_open_palm(h_l.joints)
        is_r_open = self._is_open_palm(h_r.joints)
        hands_span = abs(h_l.joints[0].nx - h_r.joints[0].nx)

        # 1. Вход в Overview
        if not self.is_in_overview:
            if is_l_open and is_r_open and hands_span > 0.38 and not p_l and not p_r:
                if not self.overview_latched:
                    self.overview_charge = min(1.0, self.overview_charge + dt * 3.3)
                    self.ctx.overview_charge = self.overview_charge
                    if self.overview_charge >= 1.0 and (now - self.overview_cooldown) > 0.60:
                        self.ctx.system_bridge.toggle_overview()
                        self.is_in_overview = True
                        self.overview_cooldown = now
                        self.overview_latched = True
                        self.overview_charge = 0.0
                        self.ctx.overview_charge = 0.0
                        self.steer_origin = None
                        self.ctx.current_gesture = "👁️ ENTERED OVERVIEW"
                        self.ctx.current_color = (255, 255, 0)
                        return True
                    else:
                        self.ctx.current_gesture = f"🖐️ OVERVIEW CHARGE [{int(self.overview_charge * 100)}%]"
                        self.ctx.current_color = (255, 220, 0)
                        return True
            else:
                self.overview_charge = max(0.0, self.overview_charge - dt * 2.5)
                self.ctx.overview_charge = self.overview_charge
                if self.overview_charge < 0.1:
                    self.overview_latched = False

        # 2. Навигация внутри
        if self.is_in_overview:
            self.ctx.mouse.release_all()
            if is_l_open and not p_l:
                steer_pt = (h_r.joints[8].nx, h_r.joints[8].ny)
                if self.steer_origin is None:
                    self.steer_origin = steer_pt
                else:
                    dx = -(steer_pt[0] - self.steer_origin[0])
                    dy = steer_pt[1] - self.steer_origin[1]
                    if (now - self.steer_step_cooldown) > 0.16:
                        if abs(dx) > 0.042 and abs(dx) > abs(dy):
                            direction = "right" if dx > 0 else "left"
                            self.ctx.system_bridge.focus_column(direction)
                            self.ctx.current_gesture = f"{'➡️' if dx > 0 else '⬅️'} OVERVIEW: {direction.upper()}"
                            self.ctx.current_color = (0, 255, 255)
                            self.steer_origin = steer_pt
                            self.steer_step_cooldown = now
                            return True
                        elif abs(dy) > 0.042:
                            if dy < 0:
                                self.ctx.system_bridge.workspace_up()
                                self.ctx.current_gesture = "⬆️ OVERVIEW: WS UP"
                            else:
                                self.ctx.system_bridge.workspace_down()
                                self.ctx.current_gesture = "⬇️ OVERVIEW: WS DOWN"
                            self.ctx.current_color = (0, 255, 255)
                            self.steer_origin = steer_pt
                            self.steer_step_cooldown = now
                            return True

                if p_r:
                    self.ctx.system_bridge.toggle_overview()
                    self.is_in_overview = False
                    self.steer_origin = None
                    self.ctx.current_gesture = "✅ WINDOW SELECTED"
                    self.ctx.current_color = (0, 255, 0)
                    return True

                self.ctx.current_gesture = "🧭 STEER: R-HAND | DROP L-HAND TO SELECT"
                self.ctx.current_color = (0, 220, 255)
                return True
            else:
                self.ctx.system_bridge.toggle_overview()
                self.is_in_overview = False
                self.steer_origin = None
                self.ctx.current_gesture = "✅ AUTO-FOCUS (OVERVIEW CLOSED)"
                self.ctx.current_color = (0, 255, 0)
                return True

        return False