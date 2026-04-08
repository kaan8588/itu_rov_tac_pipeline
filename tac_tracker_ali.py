#!/usr/bin/env python3
"""
Tac challenge 2026 — Otonom Gorev Yoneticisi
===========================================================
Altyapi: ROS2 (Goruntu Yayini) + PyMavlink (Motor Kontrolu)

Hizli Kullanim (Terminal):
  Masaustu Test  : python3 tac_tracker.py --dry-run --video videos/test1.webm --tune
  Havuz (Gercek) : python3 tac_tracker.py

Otonom Gorev Kriterleri (TAC '26):
  - Hedef  : Boru (Sari) Takibi ve Gorsel Tarama
  - ArUco  : Sadece ID 1-99 arasi taranir (Elenen sahte veriler atilir)
  - Sonuclar: Hakemler icin gorev sonunda 'aruco_results.txt' raporlanir
"""

import os
import cv2
import cv2.aruco
import numpy as np
import time
import threading
import argparse
from collections import deque
from pymavlink import mavutil

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

os.environ['MAVLINK20'] = '1'

# ══════════════════════════════════════════════════════════════
# GOREV DURUMLARI (STATE MACHINE)
# ══════════════════════════════════════════════════════════════
ST_INIT    = "INIT"
ST_ARM     = "ARM"
ST_SEARCH  = "SEARCH"
ST_FOLLOW  = "FOLLOW"
ST_MARKER  = "MARKER"
ST_SURFACE = "SURFACE"
ST_DONE    = "DONE"


# ══════════════════════════════════════════════════════════════
# ARUCO YARDIMCILARI
# ══════════════════════════════════════════════════════════════
class ArucoConfirmer:
    """
    N ardisik frame'de gorulmedikce marker kabul edilmez.
    new_ids: bu cagri ile yeni onaylanan ID'ler (loglama icin).
    """
    def __init__(self, confirm=3):
        self.confirm  = confirm
        self.counts   = {}
        self.seen     = set()
        self.ordered  = []
        self.new_ids  = []

    def update(self, ids_list):
        self.new_ids = []
        ids_set = set(ids_list)
        for mid in ids_set:
            if mid < 0:
                continue
            self.counts[mid] = self.counts.get(mid, 0) + 1
            if self.counts[mid] >= self.confirm and mid not in self.seen:
                self.seen.add(mid)
                self.ordered.append(int(mid))
                self.new_ids.append(int(mid))
        for mid in list(self.counts):
            if mid not in ids_set and mid not in self.seen:
                self.counts[mid] = max(0, self.counts[mid] - 1)
        return self.ordered


def build_aruco_detector():
    """
    Strict parametre seti: false positive orani dusurmek icin.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin     = 3
    params.adaptiveThreshWinSizeMax     = 23
    params.adaptiveThreshWinSizeStep    = 4
    params.minMarkerPerimeterRate       = 0.02
    params.maxMarkerPerimeterRate       = 4.0
    # Asagidaki iki parametre orijinal kodda yoktu — eklendi:
    params.errorCorrectionRate          = 0.1   # hata duzeltme toleransi dusuk
    params.maxErroneousBitsInBorderRate = 0.05  # cerceve kusuruna tahammul yok
    params.cornerRefinementMethod       = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_aruco(frame, detector, clahe):
    """
    Dual-pass ArUco tespiti:
      Pass 1 — ham gri goruntu
      Pass 2 — CLAHE uygulanmis gri goruntu (bulanik/soluk marker'lar icin)

    Filtreler:
      - TAC §3.2.1: sadece ID 1-99 kabul edilir
      - Minimum kosegen alani > 400 px (gurultu eleme)
      - Duplikat merkez bastirma: 20px yakininda zaten bir marker varsa atla
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    all_corners  = []
    all_ids      = []
    seen_centers = []

    def _try(img):
        corners, ids, _ = detector.detectMarkers(img)
        if ids is None:
            return
        for i, mid in enumerate(ids.flatten()):
            # TAC §3.2.1: ID aralik filtresi
            if not (1 <= mid <= 99):
                continue
            # Minimum alan filtresi
            if cv2.contourArea(corners[i][0]) < 400:
                continue
            # Duplikat merkez bastirma (20px yariciap)
            pts = corners[i][0]
            cx  = pts[:, 0].mean()
            cy  = pts[:, 1].mean()
            if any((cx - sx) ** 2 + (cy - sy) ** 2 < 400
                   for sx, sy in seen_centers):
                continue
            seen_centers.append((cx, cy))
            all_corners.append(corners[i])
            all_ids.append(int(mid))

    _try(gray)              # Pass 1: ham gray
    _try(clahe.apply(gray)) # Pass 2: CLAHE retry

    return all_corners, (np.array(all_ids) if all_ids else [])


# ══════════════════════════════════════════════════════════════
# GORUNTU ISLEME VE PID TAKIP
# ══════════════════════════════════════════════════════════════
class Smoother:
    def __init__(self, size=5):
        self.buffer = deque(maxlen=size)

    def update(self, val):
        self.buffer.append(val)
        return sum(self.buffer) / len(self.buffer)

    def reset(self):
        self.buffer.clear()


class PipelineDetector:
    """
    TAC §3.2.3: Boru rengi SARI (HSV H~15-38).
    """
    def __init__(self, lower_hsv=None, upper_hsv=None, min_area=500):
        # TAC boru rengi: SARI — H:15-38 varsayilan
        self.lower_hsv = np.array(
            lower_hsv if lower_hsv else [15, 80, 80], dtype=np.uint8)
        self.upper_hsv = np.array(
            upper_hsv if upper_hsv else [38, 255, 255], dtype=np.uint8)
        self.min_area  = min_area

    def detect(self, frame, clahe):
        """
        2. Su mavisi + gri metal exclusion mask
        3. Morfoloji temizleme
        4. En buyuk gecerli kontur secimi
        """
        # --- CLAHE kontrast normalizasyonu ---
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        enhanced = cv2.merge([clahe.apply(l), a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        blur = cv2.GaussianBlur(enhanced, (7, 7), 0)
        hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # --- Exclusion maskler ---
        hsv_orig   = cv2.cvtColor(
            cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
        water_mask = cv2.inRange(hsv_orig,
                                 np.array([85,  40, 30]),
                                 np.array([135, 255, 255]))  # havuz mavisi
        gray_mask  = cv2.inRange(hsv_orig,
                                 np.array([0,  0, 50]),
                                 np.array([179, 30, 255]))   # gri metal
        exclude    = cv2.bitwise_or(water_mask, gray_mask)

        # --- Ana renk maskesi ---
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude))

        # --- Morfoloji ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)

        # --- En buyuk kontur ---
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue

            is_large = area > 1500
            rect_tmp = cv2.minAreaRect(c)
            rw, rh = rect_tmp[1]
            if rw == 0 or rh == 0:
                continue

            aspect = max(rw, rh) / min(rw, rh)
            asp_thr = 1.3 if is_large else 2.0
            if aspect < asp_thr:
                continue

            hull_area = cv2.contourArea(cv2.convexHull(c))
            sol_thr = 0.20 if is_large else 0.35
            if hull_area > 0 and (area / hull_area) < sol_thr:
                continue

            valid_contours.append(c)

        best_contour = None
        if valid_contours:
            best_contour = max(valid_contours, key=cv2.contourArea)
            
        return best_contour, mask


class TrackingController:
    """
    arctan2 tabanli heading hesabini kullanan geometrik takip kontrolu.
    """
    def __init__(self, kp_sway=0.005, kp_yaw=0.02):
        self.kp_sway       = kp_sway
        self.kp_yaw        = kp_yaw
        self.sway_smoother = Smoother(size=5)
        self.yaw_smoother  = Smoother(size=5)

    def process(self, frame_shape, best_contour, mask):
        height, width = frame_shape[:2]
        cx_frame      = width // 2

        sway_effort = yaw_effort = surge_effort = 0.0
        t_info = {'cx_pipe': None, 'cy_pipe': None, 'state': 'BORU YOK'}

        if best_contour is not None:
            # Alt ROI: sway hatasi icin (borunun hemen altina bak)
            roi_top     = int(height * 0.6)
            bottom_mask = mask[roi_top:height, :]
            M           = cv2.moments(bottom_mask)

            if M["m00"] > 500:
                raw_cx = int(M["m10"] / M["m00"])
                raw_cy = int(M["m01"] / M["m00"]) + roi_top
                t_info['state'] = "TAKIP"
            else:
                M_full = cv2.moments(best_contour)
                if M_full["m00"] != 0:
                    raw_cx = int(M_full["m10"] / M_full["m00"])
                    raw_cy = int(M_full["m01"] / M_full["m00"])
                else:
                    raw_cx, raw_cy = cx_frame, height // 2
                t_info['state'] = "YAKLASMA"

            cx_pipe              = int(self.sway_smoother.update(raw_cx))
            t_info['cx_pipe']    = cx_pipe
            t_info['cy_pipe']    = raw_cy
            sway_error           = cx_pipe - cx_frame

            if t_info['state'] == "YAKLASMA":
                sway_effort  = np.clip(sway_error * self.kp_sway, -0.4, 0.4)
                surge_effort = 0.80
            else:
                sway_effort = np.clip(sway_error * self.kp_sway, -1.0, 1.0)

                # Orta ROI: boru acisini hesapla (arctan2 heading)
                roi_bot_mid = max(0, raw_cy - 20)
                roi_top_mid = max(0, roi_bot_mid - 100)
                mid_mask    = mask[roi_top_mid:roi_bot_mid, :]
                M_mid       = cv2.moments(mid_mask)

                raw_heading_error = 0.0
                if M_mid["m00"] > 300:
                    cx_mid = int(M_mid["m10"] / M_mid["m00"])
                    cy_mid = (int(M_mid["m01"] / M_mid["m00"])
                              + roi_top_mid)
                    raw_angle_rad     = np.arctan2(
                        cx_mid - cx_pipe, abs(raw_cy - cy_mid))
                    raw_heading_error = np.degrees(raw_angle_rad)

                heading_error = self.yaw_smoother.update(raw_heading_error)
                yaw_effort    = np.clip(
                    heading_error * self.kp_yaw, -1.0, 1.0)

                surge_effort = (0.50 if abs(sway_effort) < 0.35
                                        and abs(yaw_effort) < 0.35
                                else 0.20)
        else:
            self.sway_smoother.reset()
            self.yaw_smoother.reset()

        return sway_effort, yaw_effort, surge_effort, t_info


# ══════════════════════════════════════════════════════════════
# MAVLINK YARDIMCILARI & MOCK TEST YAPISI
# ══════════════════════════════════════════════════════════════
class MockMaster:
    """dry-run modu icin sahte Pixhawk baglantisi."""
    def __init__(self):
        self.target_system    = 1
        self.target_component = 1

        class MockMav:
            def param_set_send(self, *a, **k):       pass
            def heartbeat_send(self, *a, **k):        pass
            def manual_control_send(self, *a, **k):   pass
            def set_mode_send(self, *a, **k):         pass
            def command_long_send(self, *a, **k):     pass

        self.mav = MockMav()

    def wait_heartbeat(self):              pass
    def mode_mapping(self):                return {'MANUAL': 19}
    def motors_armed(self):                return True
    def recv_match(self, **kw):            return None


def send_heartbeat(master):
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)


def send_manual(master, x=0, y=0, z=500, r=0):
    master.mav.manual_control_send(master.target_system, x, y, z, r, 0)


def set_param(master, name, value):
    master.mav.param_set_send(
        master.target_system, master.target_component,
        name.encode('utf-8'), value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)


# ══════════════════════════════════════════════════════════════
# ANA OTONOM NODE (ROS2 + PYMAVLINK)
# ══════════════════════════════════════════════════════════════
class MissionControllerNode(Node):

    def __init__(self, dry_run=False, video_path=None,
                 show_gui=False, tune=False, total_markers=10):
        super().__init__('tac_mission_tracker')
        self.dry_run    = dry_run
        self.video_path = video_path
        self.show_gui   = show_gui
        self.tune       = tune

        self.state  = ST_INIT
        self.master = None
        self._lock  = threading.Lock()
        self._x = 0
        self._y = 0
        self._z = 500
        self._r = 0

        # ROS2 goruntu yayincisi
        self.bridge    = CvBridge()
        self.image_pub = self.create_publisher(Image, '/tac_vision/hud', 10)

        # Paylasilan CLAHE ornegi (her fonksiyon kendi olusturmuyor)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # Dedektorler
        self.detector   = PipelineDetector()
        self.controller = TrackingController()
        self.aruco_det  = build_aruco_detector()
        self.confirmer  = ArucoConfirmer()

        # TAC §3.2.1: marker sayisi 4-10, bilinmiyor → default max (10)
        self.target_market_count = total_markers

        # Sayaclar / zamanlayicilar
        self.pipe_lost_cnt = 0
        self.state_t       = time.time()
        self.results_saved = False

        # Motor sinirlari
        self.MAX_SURGE = 130
        self.MAX_YAW   = 110
        self.MAX_SWAY  = 90

        self.cap = None
        self._last_hsv_str = ""

        # Canli HSV ayari icin trackbar penceresi
        if self.tune:
            self.show_gui = True
            cv2.namedWindow('TAC HUD', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('TAC HUD', 800, 600)
            cv2.createTrackbar('HMin', 'TAC HUD', 15,  179, lambda x: None)
            cv2.createTrackbar('SMin', 'TAC HUD', 80,  255, lambda x: None)
            cv2.createTrackbar('VMin', 'TAC HUD', 80,  255, lambda x: None)
            cv2.createTrackbar('HMax', 'TAC HUD', 38,  179, lambda x: None)
            cv2.createTrackbar('SMax', 'TAC HUD', 255, 255, lambda x: None)
            cv2.createTrackbar('VMax', 'TAC HUD', 255, 255, lambda x: None)

    # ── MAVLink baglantisi ─────────────────────────────────────
    def connect_mavlink(self):
        if self.dry_run:
            self.get_logger().info(
                "[DRY-RUN] Sahte Pixhawk baglantisi kuruluyor...")
            self.master = MockMaster()
            return

        self.get_logger().info("Pixhawk'a baglaniliyor (/dev/ttyACM0)...")
        self.master = mavutil.mavlink_connection(
            '/dev/ttyACM0', baud=115200, source_system=255)
        self.master.wait_heartbeat()
        self.get_logger().info(
            f"PyMavLink hazir! (Sys ID: {self.master.target_system})")

        for name, val in [("FS_PILOT_EN", 0), ("FS_GCS_EN", 0),
                          ("ARMING_CHECK", 0), ("BRD_SAFETYENABLE", 0)]:
            set_param(self.master, name, val)
            self.get_logger().info(f"  {name} = {val}")

        mode_id = self.master.mode_mapping()['MANUAL']
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)
        self.get_logger().info("  Mod: MANUAL")

    # ── Heartbeat zamanlayicisi (20 Hz) ───────────────────────
    def heartbeat_timer(self):
        try:
            with self._lock:
                x, y, z, r = self._x, self._y, self._z, self._r
            send_heartbeat(self.master)
            send_manual(self.master, x, y, z, r)
        except Exception as e:
            self.get_logger().warning(f"Heartbeat hatasi: {e}")

    # ── State degisimi ─────────────────────────────────────────
    def change_state(self, new_state):
        if self.state != new_state:
            self.get_logger().info(
                f"STATE: {self.state} → {new_state}")
            self.state         = new_state
            self.state_t       = time.time()
            self.pipe_lost_cnt = 0  # her state degisiminde sayaci sifirla

    # ── Motor komutlari ────────────────────────────────────────
    def command_motors(self, x=0, y=0, z=500, r=0):
        with self._lock:
            self._x = max(-1000, min(1000, int(x)))
            self._y = max(-1000, min(1000, int(y)))
            self._z = max(0,     min(1000, int(z)))
            self._r = max(-1000, min(1000, int(r)))

    # ── ARM sekansı ────────────────────────────────────────────
    def arming_sequence(self):
        """
        DUZELTMELER:
          - sleep 0.5 → 0.1 (daha hizli retry, 100 deneme / 10 sn)
          - send_manual eklendi (heartbeat'i destekler, ArduSub'un
            GCS baglantisi oldugunu anlamasi icin gerekli)
          - recv_match eklendi (Pixhawk mesajlarini logla)
        """
        self.get_logger().info("Motorlar kilitleniyor (ARMING)...")
        t_start = time.time()
        while time.time() - t_start < 10:
            send_heartbeat(self.master)
            send_manual(self.master, 0, 0, 500, 0)
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 21196, 0, 0, 0, 0, 0)

            msg = self.master.recv_match(
                type=['STATUSTEXT', 'COMMAND_ACK'], blocking=False)
            if msg:
                t = msg.get_type()
                if t == 'STATUSTEXT':
                    self.get_logger().info(f"  Pixhawk: {msg.text}")
                elif t == 'COMMAND_ACK':
                    self.get_logger().info(
                        f"  ACK cmd={msg.command} result={msg.result}")

            if self.master.motors_armed():
                self.get_logger().info("ARM BASARILI! Motorlar aktif.")
                return True
            time.sleep(0.1)

        self.get_logger().error("ARM BASARISIZ! Sistemi durdurun.")
        return False

    # ── Ana isleme dongusu (30 Hz ROS timer) ──────────────────
    def process_loop(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Canli HSV ayari (--tune modunda) ve Terminal Kaydedici
        if self.tune:
            try:
                win = 'TAC HUD'
                hl, sl, vl = cv2.getTrackbarPos('HMin', win), cv2.getTrackbarPos('SMin', win), cv2.getTrackbarPos('VMin', win)
                hh, sh, vh = cv2.getTrackbarPos('HMax', win), cv2.getTrackbarPos('SMax', win), cv2.getTrackbarPos('VMax', win)
                
                self.detector.lower_hsv = np.array([hl, sl, vl], dtype=np.uint8)
                self.detector.upper_hsv = np.array([hh, sh, vh], dtype=np.uint8)
                
                # Deger degisirse terminalin ayni satirina guzel bir formatta print at
                current_hsv = f"ALT:[{hl:3}, {sl:3}, {vl:3}] UST:[{hh:3}, {sh:3}, {vh:3}]"
                if current_hsv != getattr(self, '_last_hsv_str', ''):
                    import sys
                    sys.stdout.write(f"\r\033[96m[HSV GUNCEL DEGERLERI] {current_hsv}\033[0m")
                    sys.stdout.flush()
                    self._last_hsv_str = current_hsv
            except Exception:
                pass

        # 1. Boru tespiti (CLAHE + exclusion mask ile)
        best_contour, mask = self.detector.detect(frame, self.clahe)
        sway_eff, yaw_eff, surge_eff, t_info = \
            self.controller.process(frame.shape, best_contour, mask)

        # 2. ArUco tespiti (dual-pass + ID 1-99 filtresi)
        aruco_corners, aruco_ids = detect_aruco(
            frame, self.aruco_det, self.clahe)
        ordered_markers = self.confirmer.update(
            aruco_ids if len(aruco_ids) > 0 else [])

        for mid in self.confirmer.new_ids:
            self.get_logger().info(
                f"MARKER onaylandi → ID:{mid:3d} | "
                f"Sira: {','.join(str(x) for x in ordered_markers)}")

        # 3. HUD olustur ve ROS2'ye yayinla
        vis       = frame.copy()
        
        # Saptanan boruyu (maskeyi gecen kismi) transparan yesil ile boya
        if best_contour is not None:
            overlay = vis.copy()
            cv2.drawContours(overlay, [best_contour], -1, (0, 255, 0), -1)  # Iceriyi tamamen boya
            cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)               # Orijinal goruntuyle yari seffaf harmanla
            cv2.drawContours(vis, [best_contour], -1, (0, 255, 0), 2)       # Dis hatlara parlak cizgi at
            
        state_str = t_info.get('state', 'YOK')
        cx        = t_info.get('cx_pipe')
        cy        = t_info.get('cy_pipe')

        # ArUco kutu ve ID cizimi
        if len(aruco_ids) > 0:
            # drawDetectedMarkers icin ids Nx1 array olmali
            ids_draw = np.array([[x] for x in aruco_ids], dtype=np.int32)
            cv2.aruco.drawDetectedMarkers(vis, aruco_corners, ids_draw)

        # Durum bilgisi
        overlay_lines = [
            (f"SYS : {self.state}",
             (0, 255, 0)),
            (f"PIPE: {state_str}",
             (0, 255, 255)),
            (f"ARUCO: {len(ordered_markers)} / {self.target_market_count}",
             (255, 100, 255)),
            (f"IDs : {','.join(str(x) for x in ordered_markers) or '---'}",
             (80, 255, 150)),
        ]
        panel_h = len(overlay_lines) * 28 + 10
        cv2.rectangle(vis, (0, 0), (300, panel_h), (0, 0, 0), -1)
        for i, (txt, col) in enumerate(overlay_lines):
            cv2.putText(vis, txt, (8, 22 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)

        # Boru merkezi
        if cx is not None and cy is not None:
            cv2.circle(vis, (int(cx), int(cy)), 7, (0, 0, 255), -1)

        # Maske thumbnail (sag alt, sadece --tune modunda)
        if self.tune:
            fh, fw   = vis.shape[:2]
            th, tw   = fh // 4, fw // 4
            thumb    = cv2.resize(
                cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (tw, th))
            cv2.putText(thumb, "MASKE", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (200, 200, 200), 1)
            vis[fh - th:fh, fw - tw:fw] = thumb

        # ROS2 yayini
        try:
            img_msg = self.bridge.cv2_to_imgmsg(vis, encoding="bgr8")
            self.image_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"ROS yayin hatasi: {e}")

        # Yerel pencere (--show veya --tune modu)
        if self.show_gui:
            cv2.imshow("TAC HUD", vis)
            cv2.waitKey(1)

        # 4. STATE MACHINE
        pipe_detected = (t_info['state'] != "BORU YOK")

        # ─── ST_SEARCH ─────────────────────────────────────────
        if self.state == ST_SEARCH:
            if pipe_detected:
                self.pipe_lost_cnt = 0
                self.change_state(ST_FOLLOW)
            else:
                self.command_motors(x=0, y=0, z=500, r=80)

        # ─── ST_FOLLOW ─────────────────────────────────────────
        elif self.state == ST_FOLLOW:
            if not pipe_detected:
                self.pipe_lost_cnt += 1
                if self.pipe_lost_cnt > 30:
                    self.change_state(ST_SEARCH)
            else:
                self.pipe_lost_cnt = 0
                # ArUco gorunduyse marker moduna gec (hizini dusurmek icin)
                if len(aruco_ids) > 0:
                    self.change_state(ST_MARKER)
                c_surge = int(surge_eff * self.MAX_SURGE)
                c_sway  = int(sway_eff  * self.MAX_SWAY)
                c_yaw   = int(yaw_eff   * self.MAX_YAW)
                self.command_motors(x=c_surge, y=c_sway, z=500, r=c_yaw)

        # ─── ST_MARKER ─────────────────────────────────────────
        elif self.state == ST_MARKER:
            if not pipe_detected:
                self.pipe_lost_cnt += 1
                if self.pipe_lost_cnt > 20:
                    self.get_logger().warning(
                        "Boru kayboldu! ST_SEARCH'e donuluyor...")
                    self.change_state(ST_SEARCH)
                    self.command_motors(x=0, y=0, z=500, r=0)
                    return
                # Boru gecici kayip — dur, bekle
                self.command_motors(x=0, y=0, z=500, r=0)
            else:
                self.pipe_lost_cnt = 0
                # Marker yakininda daha yavas ilerle
                c_surge = int(surge_eff * self.MAX_SURGE * 0.6)
                c_yaw   = int(yaw_eff   * self.MAX_YAW   * 0.8)
                self.command_motors(x=c_surge, y=0, z=500, r=c_yaw)

            # Hedef sayiya ulasildiysa yuzey
            if len(ordered_markers) >= self.target_market_count:
                self.get_logger().info(
                    f"Hedeflenen tum {self.target_market_count} "
                    f"marker okundu → YUZEYE CIKILIYOR!")
                self.change_state(ST_SURFACE)

        # ─── ST_SURFACE ────────────────────────────────────────
        elif self.state == ST_SURFACE:
            self.command_motors(x=0, y=0, z=800, r=0)
            if time.time() - self.state_t > 8.0:
                self.change_state(ST_DONE)

        # ─── ST_DONE ───────────────────────────────────────────
        elif self.state == ST_DONE:
            self.command_motors(x=0, y=0, z=500, r=0)

            # TAC §3.3.2: Sonuclari dosyaya yaz — hakemler icin +10p/marker
            if not self.results_saved:
                self.results_saved = True
                result_str = ','.join(str(x) for x in self.confirmer.ordered)
                with open("aruco_results.txt", "w") as f:
                    f.write("--- TAC CHALLENGE 2026 - OTONOM SONUCLAR ---\n")
                    f.write(f"Tespit Edilen Marker Sayisi: "
                            f"{len(self.confirmer.ordered)}\n")
                    f.write(f"Marker ID Sirasi: {result_str}\n")
                sep = "=" * 50
                self.get_logger().info(f"\n{sep}")
                self.get_logger().info("  GOREV TAMAMLANDI")
                self.get_logger().info(
                    f"  Marker sirasi : {result_str or 'YOK'}")
                self.get_logger().info(
                    f"  Toplam marker : {len(self.confirmer.ordered)}")
                self.get_logger().info(
                    "  'aruco_results.txt' hakem dosyasi olusturuldu!")
                self.get_logger().info(f"{sep}\n")

    # ── Gorevi baslat ──────────────────────────────────────────
    def start_mission(self):
        self.connect_mavlink()
        self.change_state(ST_ARM)
        if not self.arming_sequence():
            return

        self.change_state(ST_SEARCH)

        if self.video_path:
            self.get_logger().info(
                f"Video uzerinden test: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
        else:
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not self.cap.isOpened():
            self.get_logger().error("Kamera acilamadi! Duruluyor.")
            return

        self.create_timer(0.05,  self.heartbeat_timer)  # 20 Hz
        self.create_timer(0.033, self.process_loop)      # ~30 Hz
        self.get_logger().info(
            f"OTONOM SISTEM AKTIF | Hedef marker: "
            f"{self.target_market_count} | "
            f"ROS topic: /tac_vision/hud")

    # ── Gorevi durdur / temizle ────────────────────────────────
    def stop_mission(self):
        self.get_logger().info("Motorlar durduruluyor (DISARM)...")
        self.command_motors(0, 0, 500, 0)
        with self._lock:
            x, y, z, r = self._x, self._y, self._z, self._r
        if self.master:
            send_manual(self.master, x, y, z, r)
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════
# ROV SISTEM BASLATICISI
# ══════════════════════════════════════════════════════════════
class RoverBootloader:
    """Argumanlari isleyip Otonom Sistemi ayaga kaldiran merkez."""
    
    @staticmethod
    def launch():
        import sys
        
        # Ozel Arguman Yakalayici (Standart Argparse gorunumunu kirdik)
        class TACArgumentParser(argparse.ArgumentParser):
            def error(self, message):
                sys.stderr.write(f"HATA: {message}\n")
                self.print_help()
                sys.exit(2)
                
        parser = TACArgumentParser(description="== TAC 2026 Otonom Kontrol Merkezi ==")
        parser.add_argument('--dry-run', action='store_true', help='Donanimsiz test simulasyonu (Laptop/Masaustu icindir)')
        parser.add_argument('--video', type=str, default=None, help='Test etmek istediginiz videonun yolu (.mp4/.webm)')
        parser.add_argument('--show', action='store_true', help='Kamera goruntusunu (HUD Ekranini) goster')
        parser.add_argument('--tune', action='store_true', help='HSV renk ayarlama panelini ac ve sonuclari terminale logla')
        parser.add_argument('--total-markers', type=int, default=10, help='Gorevdeki maksimum ArUco sayisi')
        
        cmd_args, ros_args = parser.parse_known_args()
        rclpy.init(args=ros_args)
        
        node = MissionControllerNode(
            dry_run       = cmd_args.dry_run,
            video_path    = cmd_args.video,
            show_gui      = cmd_args.show,
            tune          = cmd_args.tune,
            total_markers = cmd_args.total_markers,
        )
        
        try:
            node.start_mission()
            rclpy.spin(node)
        except KeyboardInterrupt:
            print("\n[BILGI] Sistemi Guvenli Cikis moduna aliyorum...")
        finally:
            node.stop_mission()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    RoverBootloader.launch()