#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kinect Wayland & Niri Ultimate Modular Controller (Adaptive High-DPI UI Edition)
"""

import argparse
import logging
import math
import os
import subprocess
import sys
import threading
import time
from typing import List, Optional, Tuple

import cv2
import evdev
from evdev import AbsInfo, UInput, ecodes as e
import numpy as np

# Подавление шума из предупреждений
# os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
# os.environ["GLOG_minloglevel"] = "3"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["ABSL_LOG_LEVEL"] = "3"

try:
    from kinect_tracker import HandData, Joint, KinectTracker, TrackingFrame
except ImportError:
    print("[FATAL] Файл 'kinect_tracker.py' не найден в текущей папке.")
    sys.exit(1)

from gestures import load_gestures
from gestures.base import (
    GestureContext,
    SCREEN_H,
    SCREEN_W,
    WORKSPACE_HEIGHT,
    WORKSPACE_WIDTH,
)

logger = logging.getLogger("MainController")

# Начальные размеры окна по умолчанию
DEFAULT_WINDOW_W = 1263
DEFAULT_WINDOW_H = 1010

FINGER_COLORS = [
    (0, 220, 255), (0, 255, 0), (255, 220, 0), (255, 0, 200), (0, 120, 255)
]
FINGER_CHAINS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [0, 17, 18, 19, 20]
]
PALM_CHAINS = [(5, 9), (9, 13), (13, 17)]


class SystemBridge:
    @staticmethod
    def run_cmd(args: List[str]):
        try:
            subprocess.run(args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as ex:
            logger.error("Ошибка исполнения команды %s: %s", args, ex)

    @classmethod
    def workspace_up(cls): cls.run_cmd(["niri", "msg", "action", "focus-workspace-up"])
    @classmethod
    def workspace_down(cls): cls.run_cmd(["niri", "msg", "action", "focus-workspace-down"])
    @classmethod
    def toggle_floating(cls): cls.run_cmd(["niri", "msg", "action", "toggle-window-floating"])
    @classmethod
    def resize_column(cls, delta: int):
        sign = "+" if delta > 0 else ""
        cls.run_cmd(["niri", "msg", "action", "set-column-width", f"{sign}{delta}%"])
    @classmethod
    def move_column(cls, direction: str):
        if direction in ("left", "right"):
            cls.run_cmd(["niri", "msg", "action", f"move-column-{direction}"])
    @classmethod
    def focus_column(cls, direction: str):
        if direction in ("left", "right"):
            cls.run_cmd(["niri", "msg", "action", f"focus-column-{direction}"])
    @classmethod
    def toggle_overview(cls): cls.run_cmd(["niri", "msg", "action", "toggle-overview"])
    @classmethod
    def maximize_column(cls): cls.run_cmd(["niri", "msg", "action", "maximize-column"])


class KeyboardEmergencyListener(threading.Thread):
    def __init__(self, on_kill_cb):
        super().__init__(daemon=True)
        self.on_kill = on_kill_cb
        self.running = True

    def run(self):
        try:
            devices = [evdev.InputDevice(p) for p in evdev.list_devices()]
            keyboards = [d for d in devices if e.EV_KEY in d.capabilities() and e.KEY_BACKSPACE in d.capabilities()[e.EV_KEY]]
            if not keyboards:
                logger.warning("Аппаратная клавиатура для аварийного KillSwitch не найдена.")
                return

            ctrl, alt = False, False
            import select

            while self.running:
                r, _, _ = select.select(keyboards, [], [], 0.2)
                for dev in r:
                    for event in dev.read():
                        if event.type == e.EV_KEY:
                            if event.code in (e.KEY_LEFTCTRL, e.KEY_RIGHTCTRL):
                                ctrl = (event.value > 0)
                            elif event.code in (e.KEY_LEFTALT, e.KEY_RIGHTALT):
                                alt = (event.value > 0)
                            elif event.code == e.KEY_BACKSPACE and event.value == 1:
                                if ctrl and alt:
                                    logger.critical("KILLER SWITCH: Сработал Ctrl+Alt+Backspace!")
                                    self.on_kill()
                                    return
        except Exception as ex:
            logger.error("Ошибка в потоке KeyboardEmergencyListener: %s", ex)

    def stop(self):
        self.running = False


class OneEuroFilter2D:
    def __init__(self, min_cutoff: float = 0.85, beta: float = 0.035, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[np.ndarray] = None
        self.dx_prev: Optional[np.ndarray] = None
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        if self.t_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        assert self.x_prev is not None
        assert self.dx_prev is not None
        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        dx_hat = self._alpha(self.d_cutoff, dt) * dx + (1 - self._alpha(self.d_cutoff, dt)) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * np.linalg.norm(dx_hat)
        x_hat = self._alpha(cutoff, dt) * x + (1 - self._alpha(cutoff, dt)) * self.x_prev
        self.x_prev = x_hat
        return x_hat

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None


class TiltManager:
    def __init__(self, tracker_ref):
        self.tracker = tracker_ref
        self.floor_deg = -15.0
        self.ceiling_deg = 20.0
        self.hand_tilt_enabled = False
        self.last_tilt_time = 0.0

    def get_tilt(self) -> float:
        return self.tracker.kinect.current_tilt

    def step(self, delta: float):
        now = time.time()
        if now - self.last_tilt_time < 0.5:
            return
        cur = self.get_tilt()
        target = float(np.clip(cur + delta, self.floor_deg, self.ceiling_deg))
        self.tracker.kinect.set_target_tilt(target)
        self.last_tilt_time = now

    def set_floor(self):
        self.floor_deg = self.get_tilt()
        logger.info("Новый ПОЛ мотора: %.1f°", self.floor_deg)

    def set_ceiling(self):
        self.ceiling_deg = self.get_tilt()
        logger.info("Новый ПОТОЛОК мотора: %.1f°", self.ceiling_deg)

    def reset_limits(self):
        self.floor_deg = -18.0
        self.ceiling_deg = 25.0
        logger.info("Пределы наклона сброшены: [%.0f° .. %.0f°]", self.floor_deg, self.ceiling_deg)

    def process_hand_tilt(self, hands: List[HandData]):
        if not self.hand_tilt_enabled or not hands:
            return
        top_wrist_y = min(h.joints[0].ny for h in hands)
        if top_wrist_y < 0.16:
            self.step(+2.0)
        elif top_wrist_y > 0.78:
            self.step(-2.0)


class WaylandPointerMouse:
    def __init__(self, width: int = SCREEN_W, height: int = SCREEN_H):
        self.w = width
        self.h = height
        self.device = None
        self.is_active = False
        self.left_pressed = False
        self.right_pressed = False
        self.middle_pressed = False

        if not os.path.exists("/dev/uinput"):
            logger.error("/dev/uinput не найден! Запустите программу с правами доступа к uinput.")
            return

        try:
            cap = {
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE, e.BTN_TOUCH],
                e.EV_ABS: [
                    (e.ABS_X, AbsInfo(value=self.w // 2, min=0, max=self.w, fuzz=0, flat=0, resolution=1)),
                    (e.ABS_Y, AbsInfo(value=self.h // 2, min=0, max=self.h, fuzz=0, flat=0, resolution=1)),
                ],
                e.EV_REL: [
                    e.REL_WHEEL, e.REL_HWHEEL, e.REL_WHEEL_HI_RES, e.REL_HWHEEL_HI_RES
                ]
            }
            self.device = UInput(cap, name="Kinect-Wayland-Pointer", version=0x40)
            self.is_active = True
            logger.info("Виртуальное uinput устройство успешно инициализировано.")
        except Exception as ex:
            logger.error("Ошибка создания uinput устройства: %s", ex)
            self.is_active = False

    def move_abs(self, x: int, y: int):
        if not self.is_active: return
        assert self.device is not None
        self.device.write(e.EV_ABS, e.ABS_X, int(np.clip(x, 0, self.w)))
        self.device.write(e.EV_ABS, e.ABS_Y, int(np.clip(y, 0, self.h)))
        self.device.syn()

    def set_left_btn(self, pressed: bool):
        if not self.is_active or self.left_pressed == pressed: return
        assert self.device is not None
        self.left_pressed = pressed
        self.device.write(e.EV_KEY, e.BTN_LEFT, 1 if pressed else 0)
        self.device.syn()

    def set_right_btn(self, pressed: bool):
        if not self.is_active or self.right_pressed == pressed: return
        assert self.device is not None
        self.right_pressed = pressed
        self.device.write(e.EV_KEY, e.BTN_RIGHT, 1 if pressed else 0)
        self.device.syn()

    def set_middle_btn(self, pressed: bool):
        if not self.is_active or self.middle_pressed == pressed: return
        assert self.device is not None
        self.middle_pressed = pressed
        self.device.write(e.EV_KEY, e.BTN_MIDDLE, 1 if pressed else 0)
        self.device.syn()

    def scroll_v(self, steps: int):
        if not self.is_active or steps == 0: return
        assert self.device is not None
        self.device.write(e.EV_REL, e.REL_WHEEL, steps)
        self.device.write(e.EV_REL, e.REL_WHEEL_HI_RES, steps * 120)
        self.device.syn()

    def scroll_h(self, steps: int):
        if not self.is_active or steps == 0: return
        assert self.device is not None
        self.device.write(e.EV_REL, e.REL_HWHEEL, steps)
        self.device.write(e.EV_REL, e.REL_HWHEEL_HI_RES, steps * 120)
        self.device.syn()

    def release_all(self):
        self.set_left_btn(False)
        self.set_right_btn(False)
        self.set_middle_btn(False)

    def close(self):
        if not self.is_active:
            return
        assert self.device is not None
        self.release_all()
        self.device.close()


class GestureEngine:
    def __init__(self, on_emergency_stop):
        self.mouse = WaylandPointerMouse(SCREEN_W, SCREEN_H)
        self.filter = OneEuroFilter2D(min_cutoff=0.85, beta=0.035)
        self.ctx = GestureContext(self.mouse, SystemBridge, self.filter, on_emergency_stop)
        self.gestures = load_gestures(self.ctx)
        self.prev_tick = time.time()
        self.hand_distance_m = 0.0

    def process(self, frame: TrackingFrame):
        now = frame.timestamp
        dt = max(now - self.prev_tick, 1e-4)
        self.prev_tick = now

        active_hands = [h for h in frame.hands if self.ctx.is_in_box(h)]

        if not active_hands:
            self.mouse.release_all()
            for g in self.gestures:
                g.reset()
            self.ctx.current_gesture = "CLUTCH (HANDS RESTING)"
            self.ctx.current_color = (100, 100, 100)
            self.hand_distance_m = 0.0
            return

        self.hand_distance_m = active_hands[0].depth_m

        for gesture in self.gestures:
            try:
                handled = gesture.process(frame, active_hands, dt)
                if handled:
                    break
            except Exception as ex:
                logger.error("Ошибка при выполнении плагина '%s': %s", gesture.name, ex, exc_info=True)

    # --------------------------------------------------------------------------
    # АДАПТИВНЫЙ FULL HUD (Вписывается в любой размер окна)
    # --------------------------------------------------------------------------
    def draw_full_hud(
        self,
        raw_frame: np.ndarray,
        frame: TrackingFrame,
        tilt_mgr: TiltManager,
        target_w: int = DEFAULT_WINDOW_W,
        target_h: int = DEFAULT_WINDOW_H,
        keep_aspect: bool = True
    ) -> np.ndarray:
        raw_h, raw_w, _ = raw_frame.shape

        if keep_aspect:
            # Сохранение пропорций (Letterbox / Pillarbox)
            scale = min(target_w / float(raw_w), target_h / float(raw_h))
            scaled_w = max(1, int(raw_w * scale))
            scaled_h = max(1, int(raw_h * scale))
            off_x = (target_w - scaled_w) // 2
            off_y = (target_h - scaled_h) // 2

            canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            canvas[:] = (18, 18, 22)  # Фон полос обрамления
            resized_video = cv2.resize(raw_frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            canvas[off_y:off_y + scaled_h, off_x:off_x + scaled_w] = resized_video
        else:
            # Растягивание на всё окно (Fill)
            scaled_w = target_w
            scaled_h = target_h
            off_x = 0
            off_y = 0
            canvas = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Функция преобразования нормализованных координат (0..1) в пиксели холста
        def to_canvas(nx: float, ny: float) -> Tuple[int, int]:
            cx = off_x + int(nx * scaled_w)
            cy = off_y + int(ny * scaled_h)
            return cx, cy

        # 1. 3D Interaction Box
        x_min = (1.0 - WORKSPACE_WIDTH) / 2.0
        x_max = x_min + WORKSPACE_WIDTH
        y_min = (1.0 - WORKSPACE_HEIGHT) / 2.0
        y_max = y_min + WORKSPACE_HEIGHT
        bx1, by1 = to_canvas(x_min, y_min)
        bx2, by2 = to_canvas(x_max, y_max)
        is_active = len([hd for hd in frame.hands if self.ctx.is_in_box(hd)]) > 0
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (0, 255, 0) if is_active else (70, 70, 70), 2, lineType=cv2.LINE_AA)

        # 2. Отрисовка скелета
        line_thickness = max(2, int(scaled_w / 500))
        joint_radius = max(3, int(scaled_w / 350))

        for hand in frame.hands:
            if hand.depth_m > 1.00:
                continue

            # Ладонь
            for s, end in PALM_CHAINS:
                p1 = to_canvas(hand.joints[s].nx, hand.joints[s].ny)
                p2 = to_canvas(hand.joints[end].nx, hand.joints[end].ny)
                cv2.line(canvas, p1, p2, (200, 200, 200), 2, lineType=cv2.LINE_AA)

            p_wrist = to_canvas(hand.joints[0].nx, hand.joints[0].ny)
            p_mcp5 = to_canvas(hand.joints[5].nx, hand.joints[5].ny)
            p_mcp17 = to_canvas(hand.joints[17].nx, hand.joints[17].ny)
            cv2.line(canvas, p_wrist, p_mcp5, (200, 200, 200), 2, lineType=cv2.LINE_AA)
            cv2.line(canvas, p_wrist, p_mcp17, (200, 200, 200), 2, lineType=cv2.LINE_AA)

            # Пальцы
            for f_idx, chain in enumerate(FINGER_CHAINS):
                col = FINGER_COLORS[f_idx]
                for i in range(len(chain) - 1):
                    p1 = to_canvas(hand.joints[chain[i]].nx, hand.joints[chain[i]].ny)
                    p2 = to_canvas(hand.joints[chain[i+1]].nx, hand.joints[chain[i+1]].ny)
                    cv2.line(canvas, p1, p2, col, line_thickness, lineType=cv2.LINE_AA)

            # Точки суставов
            for j in hand.joints:
                pj = to_canvas(j.nx, j.ny)
                cv2.circle(canvas, pj, joint_radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)

            it_pt = to_canvas(hand.joints[8].nx, hand.joints[8].ny)
            cv2.circle(canvas, it_pt, max(6, int(scaled_w / 140)), (0, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(canvas, f"{hand.depth_m:.2f}m", (it_pt[0] + 10, it_pt[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # 3. Адаптивная верхняя панель (HUD Banner)
        bar_h = 58
        cv2.rectangle(canvas, (0, 0), (target_w, bar_h), (18, 18, 22), -1)
        cv2.line(canvas, (0, bar_h), (target_w, bar_h), (55, 55, 60), 1)

        u_stat = "UINPUT: OK" if self.mouse.is_active else "UINPUT: OFF"
        u_col = (0, 255, 0) if self.mouse.is_active else (0, 160, 255)
        tilt_mode = "HAND" if tilt_mgr.hand_tilt_enabled else "KB"
        aspect_str = "FIT (5:4)" if keep_aspect else "FILL"

        f_scale = max(0.40, min(0.55, target_w / 2400.0 + 0.25))
        cv2.putText(canvas, f"FPS: {frame.fps:.1f} | {u_stat} | Tilt: {tilt_mgr.get_tilt():.1f}° ({tilt_mode}) | [{aspect_str}]",
                    (15, 22), cv2.FONT_HERSHEY_SIMPLEX, f_scale, u_col, 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, f"ACTION: {self.ctx.current_gesture}", (15, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, f_scale + 0.12, self.ctx.current_color, 2, lineType=cv2.LINE_AA)

        # 4. Шкала прогресса справа
        gw = max(110, min(260, int(target_w * 0.25)))
        gh = 20
        gx = target_w - gw - 15
        gy = (bar_h - gh) // 2
        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), (45, 45, 50), -1)

        if self.ctx.overview_charge > 0.0:
            fill_o = int(self.ctx.overview_charge * gw)
            cv2.rectangle(canvas, (gx, gy), (gx + fill_o, gy + gh), (0, 255, 255), -1)
            cv2.putText(canvas, "Overview", (gx + 8, gy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        elif self.ctx.fist_hold_ratio > 0.0:
            fill_f = int(self.ctx.fist_hold_ratio * gw)
            cv2.rectangle(canvas, (gx, gy), (gx + fill_f, gy + gh), (0, 255, 255), -1)
            cv2.putText(canvas, "Fist", (gx + 8, gy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        else:
            fill = int(np.clip((0.55 - self.ctx.pinch_val) / (0.55 - 0.25), 0.0, 1.0) * gw)
            cv2.rectangle(canvas, (gx, gy), (gx + fill, gy + gh), self.ctx.current_color, -1)
            cv2.putText(canvas, "Pinch Lock", (gx + 8, gy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        return canvas

    # --------------------------------------------------------------------------
    # АДАПТИВНЫЙ MINIMAL HUD
    # --------------------------------------------------------------------------
    def draw_minimal_hud(
        self,
        frame: TrackingFrame,
        tilt_mgr: TiltManager,
        target_w: int = 420,
        target_h: int = 190
    ) -> np.ndarray:
        w = max(380, target_w)
        h = max(180, target_h)
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (22, 22, 26)

        cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), (55, 55, 65), 1)
        u_stat = "UINPUT: OK" if self.mouse.is_active else "UINPUT: OFF"
        u_col = (0, 255, 0) if self.mouse.is_active else (0, 160, 255)
        cv2.putText(canvas, f"FPS: {frame.fps:.1f} | {u_stat}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, u_col, 1, lineType=cv2.LINE_AA)

        dist_str = f"Z: {self.hand_distance_m:.2f}m (Max 1.0m)" if self.hand_distance_m > 0 else "Z: ---"
        cv2.putText(canvas, dist_str, (max(190, w - 190), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, lineType=cv2.LINE_AA)
        cv2.line(canvas, (15, 36), (w - 15, 36), (45, 45, 55), 1)

        cv2.putText(canvas, "ACTION:", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, lineType=cv2.LINE_AA)
        cv2.putText(canvas, self.ctx.current_gesture, (15, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.ctx.current_color, 2, lineType=cv2.LINE_AA)

        cv2.putText(canvas, f"Cursor: [{self.ctx.cursor_pos[0]:4d}, {self.ctx.cursor_pos[1]:4d}]",
                    (15, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1, lineType=cv2.LINE_AA)

        gx, gy, gw, gh = 70, 134, max(120, w - 85), 16
        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), (40, 40, 48), -1)
        if self.ctx.overview_charge > 0.0:
            fill_o = int(self.ctx.overview_charge * gw)
            cv2.rectangle(canvas, (gx, gy), (gx + fill_o, gy + gh), (0, 255, 255), -1)
        elif self.ctx.fist_hold_ratio > 0.0:
            fill_f = int(self.ctx.fist_hold_ratio * gw)
            cv2.rectangle(canvas, (gx, gy), (gx + fill_f, gy + gh), (0, 255, 255), -1)
        else:
            fill = int(np.clip((0.55 - self.ctx.pinch_val) / (0.55 - 0.25), 0.0, 1.0) * gw)
            cv2.rectangle(canvas, (gx, gy), (gx + fill, gy + gh), self.ctx.current_color, -1)

        tilt_mode = "HAND" if tilt_mgr.hand_tilt_enabled else "KB"
        cv2.putText(canvas, f"Tilt: {tilt_mgr.get_tilt():.1f}° [{tilt_mgr.floor_deg:.0f}°..{tilt_mgr.ceiling_deg:.0f}°] ({tilt_mode})",
                    (15, min(h - 18, 172)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 220, 255), 1, lineType=cv2.LINE_AA)

        return canvas

    def close(self):
        self.mouse.close()

def mask_by_depth(
    rgb_bgr: np.ndarray,
    depth_mm: Optional[np.ndarray],
    min_mm: int = 300,
    max_mm: int = 1000,
    bg_color: Tuple[int, int, int] = (18, 18, 22),
    feather: int = 0,
) -> np.ndarray:
    """
    Оставляет пиксели RGB, глубина которых лежит в [min_mm .. max_mm].
    Всё остальное закрашивается bg_color (фон окна).
    depth_mm приходит как 640x480, RGB — 1280x1024, поэтому ресайзим nearest.
    """
    if depth_mm is None:
        return rgb_bgr

    h, w = rgb_bgr.shape[:2]
    d = cv2.resize(depth_mm, (w, h), interpolation=cv2.INTER_NEAREST)

    # Kinect отдаёт 0 там, где нет данных → это тоже «далеко» и надо скрыть
    valid = (d >= min_mm) & (d <= max_mm)
    mask_u8 = valid.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    valid = mask_u8 > 0

    if feather > 0:
        m = (valid.astype(np.uint8) * 255)
        m = cv2.GaussianBlur(m, (feather | 1, feather | 1), 0).astype(np.float32) / 255.0
        m = m[..., None]
        out = rgb_bgr.astype(np.float32) * m + np.array(bg_color, dtype=np.float32) * (1.0 - m)
        return np.clip(out, 0, 255).astype(np.uint8)

    out = rgb_bgr.copy()
    out[~valid] = bg_color
    return out


def setup_logging(level_name: str):
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        print(f"[WARN] Недопустимый уровень логирования '{level_name}'. Использован INFO.")
        numeric_level = logging.INFO

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%H:%M:%S"
    )


def main():
    parser = argparse.ArgumentParser(description="Kinect Wayland Modular Controller")
    parser.add_argument("--nogui", action="store_true", help="Режим демона БЕЗ графического окна")
    parser.add_argument("--mingui", action="store_true", help="Компактный HUD виджет")
    parser.add_argument("--fill", action="store_true", help="Растягивать изображение без сохранения пропорций")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Уровень логирования (default: INFO)")
    parser.add_argument("--mask", action="store_true",
                        help="Маскировать пиксели дальше 1 м (по depth-карте)")
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger.info("Инициализация контроллера (Mode: %s)...", "DAEMON" if args.nogui else ("MINIMAL" if args.mingui else "FULL"))

    stop_event = threading.Event()

    def trigger_emergency_stop():
        sys.stdout.write("\a")
        sys.stdout.flush()
        stop_event.set()

    kb_listener = KeyboardEmergencyListener(on_kill_cb=trigger_emergency_stop)
    kb_listener.start()

    engine = GestureEngine(on_emergency_stop=trigger_emergency_stop)

    window_name = "kinect-controller"
    keep_aspect = not args.fill

    mask_enabled = bool(args.mask)
    if mask_enabled and (args.nogui or args.mingui):
        logger.warning("--mask работает только в полноэкранном режиме (full HUD). Отключено.")
        mask_enabled = False

    if not args.nogui:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        # Разрешаем тайлинговому WM свободно менять размер окна
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        if args.mingui:
            cv2.resizeWindow(window_name, 420, 190)
        else:
            cv2.resizeWindow(window_name, DEFAULT_WINDOW_W, DEFAULT_WINDOW_H)

    need_raw_images = (not args.nogui and not args.mingui)

    try:
        with KinectTracker(auto_tilt=False, include_raw_images=need_raw_images) as tracker:
            tilt_mgr = TiltManager(tracker)

            for frame in tracker.stream():
                if stop_event.is_set():
                    logger.warning("Получен сигнал экстренной остановки.")
                    break

                tilt_mgr.process_hand_tilt(frame.hands)
                engine.process(frame)

                if not args.nogui:
                    # Проверяем, не закрыл ли пользователь окно в WM
                    try:
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    except Exception:
                        break

                    # Динамически получаем реальный размер окна в Niri
                    fallback_w = 420 if args.mingui else DEFAULT_WINDOW_W
                    fallback_h = 190 if args.mingui else DEFAULT_WINDOW_H
                    win_w, win_h = fallback_w, fallback_h

                    try:
                        rect = cv2.getWindowImageRect(window_name)
                        if rect is not None and len(rect) == 4:
                            _, _, rw, rh = rect
                            if rw > 64 and rh > 64:
                                win_w, win_h = int(rw), int(rh)
                    except Exception:
                        pass

                    if not args.mingui:
                        if frame.rgb is not None:
                            raw_canvas = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)

                            if mask_enabled and frame.depth is not None:
                                raw_canvas = mask_by_depth(
                                    raw_canvas, frame.depth,
                                    min_mm=300, max_mm=1000,
                                    bg_color=(18, 18, 22),
                                    feather=9,
                                )

                            hud = engine.draw_full_hud(
                                raw_canvas, frame, tilt_mgr,
                                target_w=win_w, target_h=win_h,
                                keep_aspect=keep_aspect
                            )
                            cv2.imshow(window_name, hud)
                    else:
                        hud = engine.draw_minimal_hud(
                            frame, tilt_mgr,
                            target_w=win_w, target_h=win_h
                        )
                        cv2.imshow(window_name, hud)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord('q')):
                        break
                    elif key == ord('a'):
                        keep_aspect = not keep_aspect
                        logger.info("Режим пропорций изменен на: %s", "FIT (5:4)" if keep_aspect else "FILL (Растягивание)")
                    elif key in (ord('w'), 82):
                        tilt_mgr.step(+2.0)
                    elif key in (ord('s'), 84):
                        tilt_mgr.step(-2.0)
                    elif key == ord('['):
                        tilt_mgr.set_floor()
                    elif key == ord(']'):
                        tilt_mgr.set_ceiling()
                    elif key == ord('r'):
                        tilt_mgr.reset_limits()
                    elif key == ord('t'):
                        tilt_mgr.hand_tilt_enabled = not tilt_mgr.hand_tilt_enabled
                        logger.info("Жестовый наклон мотора: %s", 'ВКЛ' if tilt_mgr.hand_tilt_enabled else 'ВЫКЛ')
                    elif key == ord('f'):
                        SystemBridge.toggle_floating()
                    elif key == ord('o'):
                        SystemBridge.toggle_overview()
                    elif key == ord('m'):
                        if args.mingui:
                            logger.warning("Маскирование недоступно в режиме --mingui.")
                        else:
                            mask_enabled = not mask_enabled
                            logger.info("Маскирование по глубине: %s", "ВКЛ" if mask_enabled else "ВЫКЛ")

    except KeyboardInterrupt:
        logger.info("Остановка пользователем (SIGINT).")
    except Exception as ex:
        logger.critical("Критический сбой в основном цикле: %s", ex, exc_info=True)
    finally:
        kb_listener.stop()
        engine.close()
        if not args.nogui:
            cv2.destroyAllWindows()
        logger.info("Контроллер завершил работу.")


if __name__ == "__main__":
    main()