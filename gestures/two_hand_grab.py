#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List
from .base import BaseGesture, GestureContext

PINCH_CLICK_DOWN = 0.25


class TwoHandGrabGesture(BaseGesture):
    name = "Two-Hand Niri Grab"
    priority = 90

    def __init__(self, ctx: GestureContext):
        super().__init__(ctx)
        self.two_hand_active = False
        self.smoothed_2h_dist = None
        self.smoothed_2h_cx = None
        self.smoothed_2h_cy = None
        self.two_hand_cooldown = 0.0

    def reset(self):
        self.two_hand_active = False
        self.smoothed_2h_dist = None
        self.smoothed_2h_cx = None
        self.smoothed_2h_cy = None

    def process(self, frame, active_hands, dt: float) -> bool:
        if len(active_hands) < 2:
            self.reset()
            return False

        sorted_h = sorted(active_hands, key=lambda h: h.joints[0].nx)
        h_l, h_r = sorted_h[0], sorted_h[1]

        sz_l = self.ctx.get_palm_scale(h_l.joints)
        sz_r = self.ctx.get_palm_scale(h_r.joints)
        p_l = (self.ctx.dist2d(h_l.joints[4], h_l.joints[8]) / sz_l) < PINCH_CLICK_DOWN
        p_r = (self.ctx.dist2d(h_r.joints[4], h_r.joints[8]) / sz_r) < PINCH_CLICK_DOWN

        if p_l and p_r:
            self.ctx.mouse.release_all()
            cur_dist = self.ctx.dist2d(h_l.joints[0], h_r.joints[0])
            cur_cx = (h_l.joints[0].nx + h_r.joints[0].nx) / 2.0
            cur_cy = (h_l.joints[0].ny + h_r.joints[0].ny) / 2.0
            now = frame.timestamp

            if self.two_hand_active:
                assert self.smoothed_2h_dist is not None
                assert self.smoothed_2h_cx is not None
                assert self.smoothed_2h_cy is not None
                self.smoothed_2h_dist = 0.75 * self.smoothed_2h_dist + 0.25 * cur_dist
                self.smoothed_2h_cx = 0.75 * self.smoothed_2h_cx + 0.25 * cur_cx
                self.smoothed_2h_cy = 0.75 * self.smoothed_2h_cy + 0.25 * cur_cy

                delta_dist = cur_dist - self.smoothed_2h_dist
                delta_cx = cur_cx - self.smoothed_2h_cx
                delta_cy = cur_cy - self.smoothed_2h_cy

                if (now - self.two_hand_cooldown) > 0.09:
                    if abs(delta_cy) > 0.038 and abs(delta_cy) > abs(delta_cx) * 1.3:
                        if delta_cy < 0:
                            self.ctx.system_bridge.workspace_up()
                            self.ctx.current_gesture = "🚀 WORKSPACE: UP (PINCH)"
                        else:
                            self.ctx.system_bridge.workspace_down()
                            self.ctx.current_gesture = "🚀 WORKSPACE: DOWN (PINCH)"
                        self.ctx.current_color = (0, 255, 255)
                        self.two_hand_cooldown = now + 0.25
                    elif delta_dist > 0.065:
                        self.ctx.system_bridge.maximize_column()
                        self.two_hand_cooldown = now + 0.40
                        self.ctx.current_gesture = "🗖 MAXIMIZE / FULLSCREEN"
                        self.ctx.current_color = (0, 255, 120)
                    elif abs(delta_dist) > 0.024:
                        pct = 5 if delta_dist > 0 else -5
                        self.ctx.system_bridge.resize_column(pct)
                        self.two_hand_cooldown = now
                        self.ctx.current_gesture = f"NIRI RESIZE: {'+' if pct>0 else ''}{pct}%"
                        self.ctx.current_color = (255, 0, 255)
                    elif abs(delta_cx) > 0.032:
                        direction = "left" if delta_cx > 0 else "right"
                        self.ctx.system_bridge.move_column(direction)
                        self.two_hand_cooldown = now
                        self.ctx.current_gesture = f"NIRI MOVE: {direction.upper()}"
                        self.ctx.current_color = (255, 120, 0)
            else:
                self.smoothed_2h_dist = cur_dist
                self.smoothed_2h_cx = cur_cx
                self.smoothed_2h_cy = cur_cy
                self.ctx.current_gesture = "NIRI 2-HAND GRAB"
                self.ctx.current_color = (200, 0, 200)

            self.two_hand_active = True
            return True

        self.reset()
        return False