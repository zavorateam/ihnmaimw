#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kinect v1 High-Performance Multimodal Tracker & Integration API.
Combines SXGA High-Res Capture, TouchDesigner-Grade Zero-Lag DSP,
MediaPipe Tasks Neural Regressors, and Metric 3D Depth Isolation.
"""

from dataclasses import dataclass, field
import math
import os
import sys
import threading
import time
from typing import Dict, Generator, List, Optional, Tuple
import urllib.request

import cv2
import freenect
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"

# Подавление системных предупреждений Qt/Wayland
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# ==============================================================================
# КОНФИГУРАЦИЯ И ССЫЛКИ НА МОДЕЛИ
# ==============================================================================
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
POSE_MODEL_PATH = "pose_landmarker_full.task"

# Рабочие диапазоны
MIN_TRACKING_DIST_M = 0.30
MAX_TRACKING_DIST_M = 1.00

# Мотор Kinect
KINECT_VFOV_DEG = 43.0
MIN_TILT_DEG = -18.0
MAX_TILT_DEG = 28.0
TILT_DEADZONE_DEG = 2.0
TILT_COOLDOWN_SEC = 0.8
SMOOTHING_ALPHA = 0.20

# Имена 21 сустава руки
JOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24)
]


def ensure_models():
    for p, u in [(HAND_MODEL_PATH, HAND_MODEL_URL), (POSE_MODEL_PATH, POSE_MODEL_URL)]:
        if not os.path.exists(p):
            print(f"[INFO] Загрузка модели {p}...")
            urllib.request.urlretrieve(u, p)
            print(f"[INFO] Сохранено: {p}")


def safe_pt(x: float, y: float, max_w: int, max_h: int) -> Tuple[int, int]:
    if not (math.isfinite(x) and math.isfinite(y)):
        return (0, 0)
    cx = int(np.clip(round(float(x)), 0, max_w - 1))
    cy = int(np.clip(round(float(y)), 0, max_h - 1))
    return (cx, cy)


# ==============================================================================
# СТРУКТУРЫ ДАННЫХ ДЛЯ ВНЕШНЕЙ ИНТЕГРАЦИИ (DATA API)
# ==============================================================================
@dataclass
class Joint:
    """Точка сустава скелета."""
    id: int
    name: str
    nx: float        # Нормализованный X [0.0 .. 1.0]
    ny: float        # Нормализованный Y [0.0 .. 1.0]
    px: int          # Экранный X (в пикселях SXGA 1280x1024)
    py: int          # Экранный Y (в пикселях SXGA 1280x1024)
    z: float         # Метрическое расстояние от камеры в метрах (например, 0.58 м)


@dataclass
class HandData:
    """Полные данные отслеженной кисти руки."""
    name: str                        # "Hand 1" / "Hand 2"
    is_tracked: bool                 # Активен ли трекинг
    depth_m: float                   # Среднее расстояние до кисти (метры)
    palm_center: Tuple[int, int, float] # (px, py, z_meters)
    joints: List[Joint]              # Все 21 сустав со сглаживанием Zero-Lag
    fingertips: List[Joint]          # Кончики 5 пальцев (Thumb, Index, Middle, Ring, Pinky)


@dataclass
class PoseData:
    """Данные скелета верхней части тела."""
    is_tracked: bool
    head: Optional[Joint] = None
    left_shoulder: Optional[Joint] = None
    right_shoulder: Optional[Joint] = None
    left_elbow: Optional[Joint] = None
    right_elbow: Optional[Joint] = None
    left_wrist: Optional[Joint] = None
    right_wrist: Optional[Joint] = None
    torso_center: Optional[Tuple[int, int, float]] = None


@dataclass
class TrackingFrame:
    """Главный контейнер кадра, возвращаемый во внешние проекты."""
    timestamp: float                 # Временная метка
    fps: float                       # Текущий FPS обработки
    tilt_deg: float                  # Текущий угол наклона мотора камеры
    hands: List[HandData]            # Список отслеженных кистей (до 2-х)
    pose: PoseData                   # Скелет тела
    rgb: Optional[np.ndarray] = None # Кадр RGB 1280x1024
    depth: Optional[np.ndarray] = None # Карта глубины в мм (640x480)


# ==============================================================================
# TOUCHDESIGNER ZERO-LAG DSP ENGINE (СУБСТЕППИНГ + ПРЕДИКТОР)
# ==============================================================================
class ZeroLagTouchDesignerDSP:
    def __init__(self, num_points=21, base_omega=20.0, max_omega=95.0, lead_time_sec=0.042):
        self.num_points = num_points
        self.base_omega = base_omega
        self.max_omega = max_omega
        self.lead_time = lead_time_sec
        self.max_speed = 5000.0

        self.pos = None
        self.vel = None
        self.frames_lost = 0

    def reset(self):
        self.pos = None
        self.vel = None
        self.frames_lost = 0

    def update(self, target_pts: np.ndarray, dt: float, bounds=(1280, 1024)) -> np.ndarray:
        max_w, max_h = bounds
        if target_pts is None or not np.all(np.isfinite(target_pts)):
            return self.extrapolate(dt, bounds)

        target_pts = np.clip(target_pts, 0.0, [max_w, max_h]).astype(np.float32)

        if self.pos is None or not np.all(np.isfinite(self.pos)) or dt > 0.25:
            self.pos = np.copy(target_pts)
            self.vel = np.zeros_like(self.pos)
            self.frames_lost = 0
            return self.pos

        dt_total = float(np.clip(dt, 0.001, 0.06))
        substeps = max(1, int(np.ceil(dt_total / 0.005)))
        dt_sub = dt_total / substeps

        for _ in range(substeps):
            delta = target_pts - self.pos
            dist = np.linalg.norm(delta, axis=-1, keepdims=True) + 1e-6
            current_speed = np.mean(np.linalg.norm(self.vel, axis=-1))

            speed_factor = float(np.clip(current_speed / 800.0, 0.0, 1.0))
            omega = self.base_omega + (self.max_omega - self.base_omega) * (speed_factor ** 1.5)
            zeta = 1.05 - 0.25 * speed_factor

            max_step = self.max_speed * dt_sub
            clamped_target = np.where(dist > max_step, self.pos + delta * (max_step / dist), target_pts)

            acc = (omega ** 2) * (clamped_target - self.pos) - 2.0 * zeta * omega * self.vel
            self.vel += acc * dt_sub
            self.vel = np.clip(self.vel, -self.max_speed, self.max_speed)
            self.pos += self.vel * dt_sub
            self.pos = np.clip(self.pos, -30.0, [max_w + 30.0, max_h + 30.0])

        self.frames_lost = 0
        predicted_pos = self.pos + self.vel * self.lead_time
        return np.clip(predicted_pos, 0.0, [max_w, max_h])

    def extrapolate(self, dt: float, bounds=(1280, 1024)) -> np.ndarray:
        if self.pos is None or not np.all(np.isfinite(self.pos)):
            return None

        max_w, max_h = bounds
        dt = float(np.clip(dt, 0.001, 0.03))
        self.vel *= 0.92
        self.pos += self.vel * dt
        self.pos = np.clip(self.pos, 0.0, [max_w, max_h])
        self.frames_lost += 1

        if self.frames_lost > 25:
            self.reset()
            return None

        predicted_pos = self.pos + self.vel * (self.lead_time * 0.5)
        return np.clip(predicted_pos, 0.0, [max_w, max_h])


# ==============================================================================
# ПОТОК ЗАХВАТА KINECT (SXGA 1280x1024)
# ==============================================================================
class KinectSXGAWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None
        self.is_running = True

        self.current_tilt = 0.0
        self.requested_tilt = 8.0
        self.last_tilt_time = 0.0

    def video_cb(self, dev, data, timestamp):
        with self.lock:
            self.latest_rgb = data.copy()

    def depth_cb(self, dev, data, timestamp):
        with self.lock:
            self.latest_depth = data.copy()

    def body_cb(self, dev, ctx):
        if not self.is_running:
            raise freenect.Kill

        now = time.time()
        if now - self.last_tilt_time >= TILT_COOLDOWN_SEC:
            diff = self.requested_tilt - self.current_tilt
            if abs(diff) >= TILT_DEADZONE_DEG:
                step = np.sign(diff) * min(abs(diff), 4.0)
                new_tilt = float(np.clip(self.current_tilt + step, MIN_TILT_DEG, MAX_TILT_DEG))
                try:
                    freenect.set_tilt_degs(dev, new_tilt)
                    self.current_tilt = new_tilt
                    self.last_tilt_time = now
                except Exception:
                    pass

    def get_synced_frames(self):
        with self.lock:
            if self.latest_rgb is None or self.latest_depth is None:
                return None, None
            return self.latest_rgb.copy(), self.latest_depth.copy()

    def set_target_tilt(self, target: float):
        self.requested_tilt = float(np.clip(target, MIN_TILT_DEG, MAX_TILT_DEG))

    def run(self):
        try:
            ctx = freenect.init()
            dev = freenect.open_device(ctx, 0)
            freenect.set_video_mode(dev, freenect.RESOLUTION_HIGH, freenect.VIDEO_RGB)
            freenect.set_depth_mode(dev, freenect.RESOLUTION_MEDIUM, freenect.DEPTH_REGISTERED)
            freenect.set_tilt_degs(dev, self.requested_tilt)

            freenect.runloop(
                dev=dev,
                video=self.video_cb,
                depth=self.depth_cb,
                body=self.body_cb
            )
        except freenect.Kill:
            pass
        except Exception as e:
            print(f"[FATAL Kinect Thread] {e}")

    def stop(self):
        self.is_running = False


# ==============================================================================
# ГЛАВНЫЙ КЛАСС ДВИЖКА ТРЕКИНГА
# ==============================================================================
class KinectTracker:
    """
    Основной класс трекера для использования в сторонних проектах.
    
    Пример использования:
    --------------------
    from main import KinectTracker
    
    with KinectTracker() as tracker:
        for frame in tracker.stream():
            for hand in frame.hands:
                index_tip = hand.joints[8]
                print(f"{hand.name} Index: X={index_tip.px}, Y={index_tip.py}, Z={index_tip.z:.2f}m")
    """
    def __init__(self, auto_tilt: bool = True, include_raw_images: bool = False):
        ensure_models()

        hand_opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.25,
            min_hand_presence_confidence=0.25,
            min_tracking_confidence=0.25
        )
        self.hand_tracker = vision.HandLandmarker.create_from_options(hand_opts)

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            min_pose_detection_confidence=0.25,
            min_pose_presence_confidence=0.25,
            min_tracking_confidence=0.25
        )
        self.pose_tracker = vision.PoseLandmarker.create_from_options(pose_opts)

        self.kinect = KinectSXGAWorker()
        self.dsp_filters = [
            ZeroLagTouchDesignerDSP(num_points=21, base_omega=20.0, max_omega=95.0),
            ZeroLagTouchDesignerDSP(num_points=21, base_omega=20.0, max_omega=95.0)
        ]

        self.auto_tilt = auto_tilt
        self.include_raw_images = include_raw_images
        self.smoothed_target_y = 0.55
        self.last_video_ts = 0
        self.prev_tick = time.time()
        self.fps = 0.0

    def start(self):
        """Запускает поток захвата Kinect."""
        self.kinect.start()
        self.prev_tick = time.time()

    def stop(self):
        """Останавливает устройство и освобождает ресурсы."""
        self.kinect.stop()
        self.hand_tracker.close()
        self.pose_tracker.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def sample_depth_m(self, depth_mm: np.ndarray, nx: float, ny: float) -> float:
        dh, dw = depth_mm.shape
        dx = int(np.clip(nx * dw, 0, dw - 1))
        dy = int(np.clip(ny * dh, 0, dh - 1))
        patch = depth_mm[max(0, dy-5):min(dh, dy+6), max(0, dx-5):min(dw, dx+6)]
        valid = patch[(patch >= 350) & (patch <= 1200)]
        return float(np.median(valid)) / 1000.0 if len(valid) > 0 else 0.75

    def get_data(self) -> Optional[TrackingFrame]:
        """Возвращает текущий обработанный кадр с координатами суставов."""
        rgb_raw, depth_raw = self.kinect.get_synced_frames()
        if rgb_raw is None or depth_raw is None:
            return None

        curr_tick = time.time()
        dt = curr_tick - self.prev_tick
        self.prev_tick = curr_tick
        self.fps = 0.92 * self.fps + 0.08 * (1.0 / max(dt, 1e-5))

        h, w, _ = rgb_raw.shape
        ts_ms = int(curr_tick * 1000)
        if ts_ms <= self.last_video_ts:
            ts_ms = self.last_video_ts + 1
        self.last_video_ts = ts_ms

        # Оптимизированный инференс (Fast-Feed)
        rgb_small = cv2.resize(rgb_raw, (512, 512), interpolation=cv2.INTER_LINEAR)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)

        # 1. Скелет тела
        pose_res = self.pose_tracker.detect_for_video(mp_img, ts_ms)
        pose_data = PoseData(is_tracked=False)
        torso_anchor = None

        if pose_res and pose_res.pose_landmarks:
            p_lms = pose_res.pose_landmarks[0]
            pose_data.is_tracked = True
            
            def create_joint(lm, idx_id, j_name):
                z_val = self.sample_depth_m(depth_raw, lm.x, lm.y)
                px, py = safe_pt(lm.x * w, lm.y * h, w, h)
                return Joint(id=idx_id, name=j_name, nx=lm.x, ny=lm.y, px=px, py=py, z=z_val)

            pose_data.head = create_joint(p_lms[0], 0, "NOSE")
            pose_data.left_shoulder = create_joint(p_lms[11], 11, "LEFT_SHOULDER")
            pose_data.right_shoulder = create_joint(p_lms[12], 12, "RIGHT_SHOULDER")
            pose_data.left_elbow = create_joint(p_lms[13], 13, "LEFT_ELBOW")
            pose_data.right_elbow = create_joint(p_lms[14], 14, "RIGHT_ELBOW")
            pose_data.left_wrist = create_joint(p_lms[15], 15, "LEFT_WRIST")
            pose_data.right_wrist = create_joint(p_lms[16], 16, "RIGHT_WRIST")

            if p_lms[11].visibility > 0.2 and p_lms[12].visibility > 0.2:
                tc_x = int((p_lms[11].x + p_lms[12].x) / 2.0 * w)
                tc_y = int((p_lms[11].y + p_lms[12].y) / 2.0 * h)
                tc_z = (pose_data.left_shoulder.z + pose_data.right_shoulder.z) / 2.0
                pose_data.torso_center = (tc_x, tc_y, tc_z)
                torso_anchor = (tc_x / float(w), tc_y / float(h))

        # 2. Кисти и пальцы
        hand_res = self.hand_tracker.detect_for_video(mp_img, ts_ms)
        hands_list = []
        motor_targets = []
        detected_count = 0

        if hand_res and hand_res.hand_landmarks:
            detected_count = len(hand_res.hand_landmarks)
            sorted_hands = sorted(hand_res.hand_landmarks, key=lambda h_lms: h_lms[0].x)[:2]

            for i, h_lms in enumerate(sorted_hands):
                raw_pts = np.array([[lm.x * w, lm.y * h] for lm in h_lms], dtype=np.float32)
                palm_depth = self.sample_depth_m(depth_raw, h_lms[0].x, h_lms[0].y)

                if MIN_TRACKING_DIST_M <= palm_depth <= MAX_TRACKING_DIST_M:
                    smooth_pts = self.dsp_filters[i].update(raw_pts, dt, bounds=(w, h))
                    if smooth_pts is not None:
                        joints_list = []
                        for j_idx in range(21):
                            pt_x, pt_y = float(smooth_pts[j_idx, 0]), float(smooth_pts[j_idx, 1])
                            px, py = safe_pt(pt_x, pt_y, w, h)
                            j_depth = self.sample_depth_m(depth_raw, pt_x / w, pt_y / h)
                            joints_list.append(Joint(
                                id=j_idx, name=JOINT_NAMES[j_idx],
                                nx=pt_x / w, ny=pt_y / h, px=px, py=py, z=j_depth
                            ))

                        palm_c = (joints_list[0].px, joints_list[0].py, palm_depth)
                        tips = [joints_list[4], joints_list[8], joints_list[12], joints_list[16], joints_list[20]]
                        
                        hands_list.append(HandData(
                            name=f"Hand {i+1}", is_tracked=True, depth_m=palm_depth,
                            palm_center=palm_c, joints=joints_list, fingertips=tips
                        ))

                        motor_targets.append((smooth_pts[9, 0] / float(w), smooth_pts[9, 1] / float(h)))

        # 3. Наведение мотора
        if self.auto_tilt:
            target_pt = None
            if len(motor_targets) > 0:
                tx = float(np.mean([pt[0] for pt in motor_targets]))
                ty = float(np.mean([pt[1] for pt in motor_targets]))
                target_pt = (tx, ty)
            elif torso_anchor is not None:
                target_pt = torso_anchor

            if target_pt is not None and math.isfinite(target_pt[0]) and math.isfinite(target_pt[1]):
                self.smoothed_target_y = (
                    SMOOTHING_ALPHA * target_pt[1] + (1.0 - SMOOTHING_ALPHA) * self.smoothed_target_y
                )
                err = 0.55 - self.smoothed_target_y
                new_tilt = self.kinect.current_tilt + (err * KINECT_VFOV_DEG)
                self.kinect.set_target_tilt(new_tilt)

        return TrackingFrame(
            timestamp=curr_tick,
            fps=self.fps,
            tilt_deg=self.kinect.current_tilt,
            hands=hands_list,
            pose=pose_data,
            rgb=rgb_raw if self.include_raw_images else None,
            depth=depth_raw if self.include_raw_images else None
        )

    def stream(self) -> Generator[TrackingFrame, None, None]:
        """Генератор для удобной итерации по кадрам в цикле."""
        while self.kinect.is_running:
            data = self.get_data()
            if data is not None:
                yield data
            else:
                time.sleep(0.002)


# ==============================================================================
# 5-Й РЕЖИМ: АВАТАР ТЕЛА
# ==============================================================================
def render_anatomical_3d_avatar(rgb_raw: np.ndarray, depth_raw: np.ndarray, 
                                frame_data: TrackingFrame) -> np.ndarray:
    """
    Режим 5: Вырезает пользователя от фона и стола, разделяет части тела
    на анатомические зоны и раскрашивает их неоновыми градиентами.
    """
    h, w, _ = rgb_raw.shape
    depth_resized = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)

    # 1. Отсечение заднего плана (> 1.1м) и ближнего шума (< 0.35м)
    body_mask = (depth_resized >= int(MIN_TRACKING_DIST_M * 1000)) & (depth_resized <= int(MAX_TRACKING_DIST_M * 1000))
    body_mask_u8 = body_mask.astype(np.uint8) * 255

    # Удаление стола (градиентный срез снизу экрана)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    body_mask_u8 = cv2.morphologyEx(body_mask_u8, cv2.MORPH_OPEN, kernel)

    # 2. Базовый темный холст киберпанк-пространства
    avatar_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 3. Анатомическая сегментация по частям тела
    y_indices, x_indices = np.where(body_mask_u8 > 0)
    
    if len(x_indices) > 500:
        # Рельефная текстура из нормалей глубины
        d_float = depth_resized.astype(np.float32)
        dz_dx = cv2.Sobel(d_float, cv2.CV_32F, 1, 0, ksize=3) * 0.04
        dz_dy = cv2.Sobel(d_float, cv2.CV_32F, 0, 1, ksize=3) * 0.04
        shading = np.clip(1.0 - (dz_dx**2 + dz_dy**2) * 0.01, 0.3, 1.2)

        # Цветовая карта частей тела
        color_map = np.zeros((h, w, 3), dtype=np.float32)
        # Торс по умолчанию (Пурпурно-маджентовый)
        color_map[body_mask] = [180, 20, 140]

        # Зона головы (Золотисто-желтый)
        if frame_data.pose.head:
            hx, hy = frame_data.pose.head.px, frame_data.pose.head.py
            cv2.circle(color_map, (hx, hy), 120, (0, 220, 255), -1)

        # Зона левой руки (Неоновый зеленый)
        for hand in frame_data.hands:
            if "1" in hand.name:
                px, py, _ = hand.palm_center
                cv2.circle(color_map, (px, py), 110, (0, 255, 70), -1)
            elif "2" in hand.name:
                px, py, _ = hand.palm_center
                cv2.circle(color_map, (px, py), 110, (255, 200, 0), -1)

        color_map = cv2.GaussianBlur(color_map, (55, 55), 0)
        shaded_avatar = color_map * shading[:, :, None]
        avatar_canvas[body_mask] = np.clip(shaded_avatar[body_mask], 0, 255).astype(np.uint8)

    # 4. Светящийся голографический контур (Holographic Outline)
    contours, _ = cv2.findContours(body_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(avatar_canvas, contours, -1, (255, 255, 255), 2)
    cv2.drawContours(avatar_canvas, contours, -1, (0, 255, 255), 1)

    # 5. Наложение скелетных суставов поверх аватара
    for hand in frame_data.hands:
        for s, e in HAND_CONNECTIONS:
            pt1 = (hand.joints[s].px, hand.joints[s].py)
            pt2 = (hand.joints[e].px, hand.joints[e].py)
            cv2.line(avatar_canvas, pt1, pt2, (255, 255, 255), 2)
        for j in hand.joints:
            cv2.circle(avatar_canvas, (j.px, j.py), 4, (0, 255, 255), -1)

    return avatar_canvas


# ==============================================================================
# ИНТЕРАКТИВНЫЙ ВИЗУАЛИЗАТОР
# ==============================================================================
class InteractiveVisualizer:
    def __init__(self):
        self.tracker = KinectTracker(auto_tilt=True, include_raw_images=True)
        self.display_mode = 1

    def run(self):
        print("==========================================================")
        print("  Kinect v1 High-Performance Tracker (API & Visualizer)   ")
        print("==========================================================")
        print("Режимы отображения:")
        print("  [1] Слой 1: Full Zero-Lag Tracking (SXGA 1280x1024)")
        print("  [2] Слой 2: High-Res Masked Depth Cutoff (<= 1.1m)")
        print("  [3] Слой 3: Исходный видеопоток SXGA RGB")
        print("  [4] Слой 4: Цветовая карта глубины (Registered Depth)")
        print("  [5] Слой 5: 3D-Аватар тела (Real-Time Chromatic Mesh)")
        print("  [m] Вкл/Выкл автоматическое наведение мотора")
        print("  [q] / [ESC] Выход")
        print("==========================================================")

        self.tracker.start()
        cv2.namedWindow("Kinect Master Tracker", cv2.WINDOW_AUTOSIZE)

        try:
            for frame_data in self.tracker.stream():
                rgb_raw = frame_data.rgb
                depth_raw = frame_data.depth
                h, w, _ = rgb_raw.shape

                canvas = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)

                # Отрисовка скелета тела
                if frame_data.pose.is_tracked:
                    p = frame_data.pose
                    joints_map = {
                        0: p.head, 11: p.left_shoulder, 12: p.right_shoulder,
                        13: p.left_elbow, 14: p.right_elbow, 15: p.left_wrist, 16: p.right_wrist
                    }
                    for s, e in POSE_CONNECTIONS:
                        if s in joints_map and e in joints_map and joints_map[s] and joints_map[e]:
                            pt1 = (joints_map[s].px, joints_map[s].py)
                            pt2 = (joints_map[e].px, joints_map[e].py)
                            cv2.line(canvas, pt1, pt2, (255, 170, 0), 2)
                    for j_id, j_obj in joints_map.items():
                        if j_obj:
                            cv2.circle(canvas, (j_obj.px, j_obj.py), 4, (0, 255, 255), -1)

                # Отрисовка кистей и пальцев
                for hand in frame_data.hands:
                    for s, e in HAND_CONNECTIONS:
                        pt1 = (hand.joints[s].px, hand.joints[s].py)
                        pt2 = (hand.joints[e].px, hand.joints[e].py)
                        cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

                    tip_ids = {4, 8, 12, 16, 20}
                    for j_idx, j in enumerate(hand.joints):
                        color = (0, 0, 255) if j_idx in tip_ids else (0, 255, 255)
                        r = 5 if j_idx in tip_ids else 3
                        cv2.circle(canvas, (j.px, j.py), r, color, -1)
                        cv2.circle(canvas, (j.px, j.py), r + 2, (255, 255, 255), 1)

                    wx_px, wy_px = hand.joints[0].px, hand.joints[0].py
                    cv2.putText(canvas, f"{hand.name}: {hand.depth_m:.2f}m", 
                                (max(10, wx_px - 30), max(20, wy_px - 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Формирование слоёв
                if self.display_mode == 1:
                    output = canvas
                    mode_name = "1: Full Tracking + Zero-Lag DSP"
                elif self.display_mode == 2:
                    d_res = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)
                    fg = (d_res >= int(MIN_TRACKING_DIST_M * 1000)) & (d_res <= int(MAX_TRACKING_DIST_M * 1000))
                    output = cv2.bitwise_and(canvas, canvas, mask=(fg.astype(np.uint8) * 255))
                    mode_name = "2: Masked Depth Cutoff 1.1m"
                elif self.display_mode == 3:
                    output = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                    mode_name = "3: Raw SXGA RGB Stream"
                elif self.display_mode == 4:
                    d_vis = (np.clip(depth_raw, 0, 1200) / 1200.0 * 255.0).astype(np.uint8)
                    d_vis = cv2.resize(d_vis, (w, h), interpolation=cv2.INTER_NEAREST)
                    output = cv2.applyColorMap(d_vis, cv2.COLORMAP_JET)
                    mode_name = "4: Depth Distance Map"
                elif self.display_mode == 5:
                    output = render_anatomical_3d_avatar(rgb_raw, depth_raw, frame_data)
                    mode_name = "5: Chromatic Body Avatar"
                else:
                    output = canvas
                    mode_name = "Not stated"

                # HUD
                cv2.rectangle(output, (0, 0), (w, 55), (15, 15, 15), -1)
                cv2.putText(output, f"Mode: {mode_name}", (10, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(output, f"FPS: {frame_data.fps:.1f} | Tilt: {frame_data.tilt_deg:.1f}° | Hands: {len(frame_data.hands)}/2",
                            (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                cv2.imshow("Kinect Master Tracker", output)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
                    self.display_mode = int(chr(key))
                elif key == ord('m'):
                    self.tracker.auto_tilt = not self.tracker.auto_tilt

        finally:
            self.tracker.stop()
            cv2.destroyAllWindows()


def main():
    app = InteractiveVisualizer()
    app.run()


if __name__ == "__main__":
    main()
