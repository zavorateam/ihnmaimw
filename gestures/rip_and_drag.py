#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Жест: 'Отрыв и перенос окна' (Rip to Float & Drag-Drop).
- Полная инвариантность к повороту кисти вокруг своего центра (хват плашмя, ребром, под углом).
- Естественное направление пальцев к столу.
- Математически точное восстановление размера окна в Niri (+15% на -15%).
- Блокировка всех фоновых кликов и курсора при опущенной руке.
"""

import math
from typing import Optional, Tuple
import numpy as np

from .base import BaseGesture, GestureContext

# Пороги щипка
PINCH_ENTER_DRAG = 0.35    # Срабатывание захвата
PINCH_RELEASE_DRAG = 0.48  # Размыкание и стыковка в тайлинг
MOVE_DEADZONE_PX = 6       # Мертвая зона смещения курсора


class RipAndDragGesture(BaseGesture):
    name = "Rip & Drag (Float Move)"
    priority = 95

    def __init__(self, ctx: GestureContext):
        super().__init__(ctx)
        self.is_dragging = False
        self.last_pos: Optional[Tuple[int, int]] = None
        self.cooldown_ts = 0.0
        self.lost_frames = 0

    def reset(self):
        if self.is_dragging:
            self.logger.info("Сброс жеста: возвращаем окно в тайлинг")
            self._dock_to_tiling()
        self.is_dragging = False
        self.last_pos = None
        self.lost_frames = 0

    def _get_robust_palm_scale(self, joints) -> float:
        """
        Инвариантный масштаб ладони: берет максимум из ширины, длины
        и обеих диагоналей ладони. Не сжимается ни при каком 3D-наклоне.
        """
        d_w = self.ctx.dist2d(joints[5], joints[17])   # Ширина между костяшками
        d_l = self.ctx.dist2d(joints[0], joints[9])    # Длина запястье-средний
        d_d1 = self.ctx.dist2d(joints[0], joints[17])  # Диагональ 1
        d_d2 = self.ctx.dist2d(joints[0], joints[5])   # Диагональ 2
        return max(d_w, d_l, d_d1, d_d2, 0.06)

    def _is_hand_aiming_at_table(self, joints, pc_y: float) -> bool:
        """
        Строгая проверка: указательный палец разогнут и активно направлен ВНИЗ,
        рука в позе 'хват сверху' (как будто берет что-то со стола ниже ладони).
        Лежащая плашмя рука с вытянутыми вперёд пальцами этому не удовлетворяет.
        """
        wrist     = joints[0]
        index_mcp = joints[5]
        index_pip = joints[6]
        index_dip = joints[7]
        index_tip = joints[8]

        # 1. Кончик указательного ЗАМЕТНО ниже своего MCP (палец реально смотрит вниз)
        finger_dy = index_tip.ny - index_mcp.ny
        if finger_dy < 0.08:
            return False

        # 2. Вертикальная составляющая доминирует над горизонтальной
        finger_dx = index_tip.nx - index_mcp.nx
        if abs(finger_dy) < abs(finger_dx):
            return False

        # 3. Палец разогнут, а не сжат в кулак/полукрюк:
        #    все три фаланги сопоставимы по длине и почти коллинеарны.
        v1 = np.array([index_pip.nx - index_mcp.nx, index_pip.ny - index_mcp.ny])
        v2 = np.array([index_dip.nx - index_pip.nx, index_dip.ny - index_pip.ny])
        v3 = np.array([index_tip.nx - index_dip.nx, index_tip.ny - index_dip.ny])
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        n3 = np.linalg.norm(v3) + 1e-6

        if n2 < n1 * 0.4 or n3 < n1 * 0.4:
            return False   # одна из фаланг «схлопнута» — палец согнут

        cos12 = np.dot(v1, v2) / (n1 * n2)
        cos23 = np.dot(v2, v3) / (n2 * n3)
        if cos12 < 0.5 or cos23 < 0.5:
            return False   # угол между фалангами > 60° — палец изогнут

        # 4. Запястье выше кончика указательного (ладонь наклонена вниз)
        if (index_tip.ny - wrist.ny) < 0.10:
            return False

        return True

    def _are_other_fingers_folded(self, joints, pc_x: float, pc_y: float, scale: float) -> bool:
        """
        Проверка сжатия среднего (12), безымянного (16) и мизинца (20).
        Радиальное расстояние до геометрического центра ладони (pc)
        математически инвариантно к любому углу поворота кисти в плоскости.
        """
        for tip_idx in (12, 16, 20):
            d_pc = math.hypot(joints[tip_idx].nx - pc_x, joints[tip_idx].ny - pc_y) / scale
            if d_pc > 0.72:  # Если палец распрямлен от центра ладони
                return False
        return True

    def _rip_to_floating(self):
        """Отрывает окно в floating и уменьшает размер на 15% экрана."""
        self.ctx.system_bridge.run_cmd(["niri", "msg", "action", "toggle-window-floating"])
        self.ctx.system_bridge.run_cmd(["niri", "msg", "action", "set-window-width", "-15%"])
        self.ctx.system_bridge.run_cmd(["niri", "msg", "action", "set-window-height", "-15%"])

    def _dock_to_tiling(self):
        """
        Точно восстанавливает размер:
        +15% ровно компенсирует -15% (аддитивные проценты экрана в Niri).
        """
        self.ctx.system_bridge.run_cmd(["niri", "msg", "action", "set-window-width", "+15%"])
        self.ctx.system_bridge.run_cmd(["niri", "msg", "action", "set-window-height", "+15%"])
        self.ctx.system_bridge.run_cmd(["niri", "msg", "action", "toggle-window-floating"])
        self.ctx.system_bridge.run_cmd(["niri", "msg", "action", "reset-window-height"])

    def _move_floating_window(self, dx: int, dy: int):
        self.ctx.system_bridge.run_cmd([
            "niri", "msg", "action", "move-floating-window", "-x", f"{dx:+d}", "-y", f"{dy:+d}"
        ])

    def process(self, frame, active_hands, dt: float) -> bool:
        now = frame.timestamp

        # Защита от потери руки на 1-2 кадра при переносе
        if not active_hands or len(active_hands) > 1:
            if self.is_dragging:
                self.lost_frames += 1
                if self.lost_frames > 12:  # Потеря трекинга более ~0.4 сек
                    self.reset()
                return True
            return False

        self.lost_frames = 0
        hand = active_hands[0]
        joints = hand.joints
        scale = self._get_robust_palm_scale(joints)

        # Истинный геометрический центр ладони (инвариантен к поворотам)
        pc_x = (joints[0].nx + joints[5].nx + joints[9].nx + joints[17].nx) / 4.0
        pc_y = (joints[0].ny + joints[5].ny + joints[9].ny + joints[17].ny) / 4.0

        thumb_tip = joints[4]
        index_tip = joints[8]

        pinch_dist = self.ctx.dist2d(thumb_tip, index_tip) / scale
        raw_x, raw_y = self.ctx.map_to_screen(index_tip.nx, index_tip.ny)
        filtered_pt = self.ctx.filter.filter(np.array([raw_x, raw_y], dtype=np.float32), now)
        smooth_x, smooth_y = int(filtered_pt[0]), int(filtered_pt[1])

        is_hand_down = self._is_hand_aiming_at_table(joints, pc_y)
        other_folded = self._are_other_fingers_folded(joints, pc_x, pc_y, scale)
        is_pinch = pinch_dist < PINCH_ENTER_DRAG

        # =============================================================
        # 1. СОСТОЯНИЕ: Окно в тайлинге (ожидание жеста)
        # =============================================================
        if not self.is_dragging:
            if is_hand_down:
                # Щипок + остальные пальцы сжаты к центру ладони
                if other_folded and is_pinch:
                    if (now - self.cooldown_ts) > 0.8:
                        self.is_dragging = True
                        self.last_pos = (smooth_x, smooth_y)
                        self.cooldown_ts = now
                        self._rip_to_floating()
                        self.ctx.current_gesture = "🪟 WINDOW RIPPED (FLOATING)"
                        self.ctx.current_color = (255, 0, 180)
                        return True

                # Если рука просто направлена к столу — гасим остальные жесты
                self.ctx.mouse.release_all()
                self.ctx.current_gesture = f"👇 READY TO RIP (Pinch: {pinch_dist:.2f})"
                self.ctx.current_color = (180, 100, 255)
                return True

            return False

        # =============================================================
        # 2. СОСТОЯНИЕ: Окно оторвано (перенос по экрану)
        # =============================================================
        # Разжали щипок -> Возврат в сетку тайлинга
        if pinch_dist > PINCH_RELEASE_DRAG:
            self._dock_to_tiling()
            self.is_dragging = False
            self.last_pos = None
            self.cooldown_ts = now + 0.8
            self.ctx.current_gesture = "📌 WINDOW DOCKED TO TILING"
            self.ctx.current_color = (0, 255, 120)
            return True

        # Свободное перемещение окна за рукой
        if self.last_pos is not None:
            dx = smooth_x - self.last_pos[0]
            dy = smooth_y - self.last_pos[1]

            if abs(dx) >= MOVE_DEADZONE_PX or abs(dy) >= MOVE_DEADZONE_PX:
                self._move_floating_window(dx, dy)
                self.last_pos = (smooth_x, smooth_y)

        self.ctx.cursor_pos = (smooth_x, smooth_y)
        self.ctx.mouse.move_abs(smooth_x, smooth_y)
        self.ctx.current_gesture = f"🪟 DRAGGING [{smooth_x}, {smooth_y}]"
        self.ctx.current_color = (255, 0, 180)
        return True