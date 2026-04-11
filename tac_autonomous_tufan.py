"""
TAC Challenge 2026 — Otonom Boru Hattı Takip Sistemi
====================================================

Kullanım:
  # Laptoptan test (Pixhawk yok):
  python3 tac_autonomous.py --dry-run --video /path/to/video.webm
  
 Kodu Araca Gönderme :
 scp /home/tufan/Desktop/iturov/otonomboru/tac_autonomous.py iturov@10.42.0.85:/home/iturov/

 Araca Girip Kodu Çalıştırma

    python3 tac_autonomous.py --camera 0

    http://10.42.0.85:5000
    izleme

  # Araçta gerçek görev:

  ssh iturov@10.42.0.85

  python3 tac_autonomous.py --camera 0

  # Canlı yayın: tarayıcıdan http://<ROV_IP>:5000 adresine git

Hareket Eksen Referansı (manual_control_send):
  x  : ileri/geri     -1000..+1000  (+ = ileri)
  y  : sağ/sol        -1000..+1000  (+ = sağa kay)
  z  : dikey          0..1000       (500 = dur)
  r  : yaw            -1000..+1000  (+ = sağa dön)
"""

import cv2
import cv2.aruco
import numpy as np
import argparse
import time
import threading
import http.server
import socketserver
import os
import sys
from collections import deque
from dataclasses import dataclass
from pymavlink import mavutil

os.environ['MAVLINK20'] = '1'

FORCE_ARM_MAGIC = 21196


# ══════════════════════════════════════════════════════════════
# KONFİGÜRASYON
# ══════════════════════════════════════════════════════════════
@dataclass
class Config:
    # ── HSV Renk Aralığı (boru rengi — Geçiş 1: yakın) ──────
    hmin: int = 7
    hmax: int = 35
    smin: int = 110
    smax: int = 255
    vmin: int = 130
    vmax: int = 255

    # ── Boru Algılama ─────────────────────────────────────────
    min_area: int  = 200
    yaw_thr: float = 1.5  # 3.0'dan düşürüldü: en ufak açı bozukluğunda bile anında dönmeye başlar

    # ── Hareket Güçleri (ArduSub Fisheye / Direksiyon Mantığı) ──
    spd_forward: int    = 100
    spd_slow: int       = 60
    spd_search: int     = 80
    spd_search_yaw: int = 60
    spd_yaw: int        = 120
    spd_lateral: int    = 0   # Yengeç yürüyüşü tamamen iptal edildi
    spd_soft_lat: int   = 0   # Yengeç yürüyüşü iptal
    spd_surface: int    = 150

    gain: float      = 0.5   # Orantısal direksiyon kazancı
    dead_zone: float = 0.05  # Merkez deadzone %5
    turn_zone: float = 0.20  # Viraj başlangıç bölgesi

    # İleri ve Dönüş yönünü tersine çevirmek için (True/False)
    rev_x: bool = False
    rev_r: bool = False

    # ── Görev ─────────────────────────────────────────────────
    aruco_dict: int    = cv2.aruco.DICT_ARUCO_ORIGINAL
    aruco_confirm: int = 3
    total_markers: int = 5

    # ── Sistem ────────────────────────────────────────────────
    device: str    = '/dev/ttyACM0'
    baud: int      = 115200
    camera: object = 0
    frame_w: int   = 640
    frame_h: int   = 360
    smooth_n: int  = 3  # 8'den 3'e düşürüldü: Görüntü yumuşatmasını azaltarak fiziksel sinyal gecikmesini (gecikmeli dönüşü) anında ortadan kaldırır

    surface_on_done: bool = True
    surface_secs: float   = 8.0
    pipe_lost_limit: int  = 25
    cam_fail_limit: int   = 50

    dry_run: bool  = False
    video: str     = ''
    flip: int      = None  # -1=180 derece, 0=dikey, 1=yatay

    # ── Ekran & Yayın ────────────────────────────────────────
    show_display: bool = True
    stream_on: bool    = True
    stream_port: int   = 5000


# ══════════════════════════════════════════════════════════════
# STATE MACHINE
# ══════════════════════════════════════════════════════════════
ST_INIT    = "INIT"
ST_ARM     = "ARM"
ST_SEARCH  = "SEARCH"
ST_FOLLOW  = "FOLLOW"
ST_MARKER  = "MARKER"
ST_SURFACE = "SURFACE"
ST_DONE    = "DONE"


# ══════════════════════════════════════════════════════════════
# YARDIMCI SINIFLAR
# ══════════════════════════════════════════════════════════════
class Smoother:
    def __init__(self, n=8):
        self.buf = deque(maxlen=n)

    def update(self, v):
        self.buf.append(v)
        return int(np.mean(self.buf))

    def reset(self):
        self.buf.clear()


class ArucoConfirmer:
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
            if mid < 0: continue
            self.counts[mid] = self.counts.get(mid, 0) + 1
            if self.counts[mid] >= self.confirm and mid not in self.seen:
                self.seen.add(mid)
                self.ordered.append(int(mid))
                self.new_ids.append(int(mid))
        for mid in list(self.counts):
            if mid not in ids_set and mid not in self.seen:
                self.counts[mid] = max(0, self.counts[mid] - 1)
        return self.ordered


# ══════════════════════════════════════════════════════════════
# MJPEG WEB YAYIN SUNUCUSU
# ══════════════════════════════════════════════════════════════
class MJPEGStreamer(http.server.BaseHTTPRequestHandler):
    """Tarayıcıdan http://ROV_IP:5000 ile canlı HUD izleme."""
    server_frame = None
    server_lock  = threading.Lock()
    server_x = 0
    server_y = 0
    server_z = 500
    server_r = 0
    server_state = "INIT"
    server_err = 0
    server_pipe = False

    # Dashboard HTML'ini dosyadan yükle
    _dashboard_html = None

    @classmethod
    def _load_dashboard(cls):
        if cls._dashboard_html is not None:
            return cls._dashboard_html
        html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'dashboard.html')
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                cls._dashboard_html = f.read()
        except FileNotFoundError:
            cls._dashboard_html = "<html><body><h1>dashboard.html bulunamadi!</h1></body></html>"
        return cls._dashboard_html

    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(MJPEGStreamer._load_dashboard().encode("utf-8"))
            return

        if self.path == '/state':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            import json
            with MJPEGStreamer.server_lock:
                state_data = {
                    "x": MJPEGStreamer.server_x,
                    "y": MJPEGStreamer.server_y,
                    "z": MJPEGStreamer.server_z,
                    "r": MJPEGStreamer.server_r,
                    "state": MJPEGStreamer.server_state,
                    "err": MJPEGStreamer.server_err,
                    "pipe": MJPEGStreamer.server_pipe,
                }
            self.wfile.write(json.dumps(state_data).encode("utf-8"))
            return
            
        if self.path != '/stream':
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header('Content-type',
                         'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                with MJPEGStreamer.server_lock:
                    frame = MJPEGStreamer.server_frame
                if frame is None:
                    time.sleep(0.05)
                    continue
                _, jpeg = cv2.imencode('.jpg', frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 75])
                data = jpeg.tobytes()
                self.wfile.write(b'--frame\r\n')
                self.wfile.write(b'Content-Type: image/jpeg\r\n')
                self.wfile.write(f'Content-Length: {len(data)}\r\n'.encode())
                self.wfile.write(b'\r\n')
                self.wfile.write(data)
                self.wfile.write(b'\r\n')
                time.sleep(0.05)  # ~20 FPS
        except (BrokenPipeError, ConnectionResetError):
            pass


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


# ══════════════════════════════════════════════════════════════
# GÖRÜNTÜ İŞLEME
# ══════════════════════════════════════════════════════════════
def detect_pipe(frame, lo, hi, min_area):
    """Dual-pass boru algılama. Geçiş 1: yakın, Geçiş 2: uzak."""
    # CLAHE kontrast normalizasyonu
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    blur = cv2.GaussianBlur(enhanced, (7, 7), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # İstenmeyen renk maskeleri (orijinal frame üzerinden)
    hsv_orig = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0),
                            cv2.COLOR_BGR2HSV)
    water_mask = cv2.inRange(hsv_orig,
                             np.array([85, 40, 30]),
                             np.array([135, 255, 255]))
    skin_mask  = cv2.inRange(hsv_orig,
                             np.array([0, 30, 60]),
                             np.array([28, 180, 255]))
    gray_mask  = cv2.inRange(hsv_orig,
                             np.array([0, 0, 50]),
                             np.array([179, 30, 255]))
    exclude = cv2.bitwise_or(water_mask,
                             cv2.bitwise_or(skin_mask, gray_mask))

    # Geçiş 1: yakın boru (ölçülen: H:14-25, S:181-255, V:201-255)
    # Geçiş 2: uzak boru  (ölçülen: H:32-80, S:147-245, V:109-207)
    passes = [
        (lo, hi, min_area),
        (np.array([30, 130, 90]), np.array([85, 255, 255]), min_area // 2),
    ]

    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_mid   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Fallback maske (HUD'da gösterilir)
    fallback_mask = cv2.inRange(hsv, lo, hi)

    for p_lo, p_hi, p_min in passes:
        mask = cv2.inRange(hsv, p_lo, p_hi)
        mask = cv2.bitwise_and(mask, mask,
                               mask=cv2.bitwise_not(exclude))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_small, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_mid,   iterations=2)
        mask = cv2.dilate(mask, k_small, iterations=1)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        valid = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < p_min:
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
            valid.append(c)

        if valid:
            best = max(valid, key=cv2.contourArea)
            rect = cv2.minAreaRect(best)
            (cx, cy), (w, h), ang = rect
            return mask, best, rect, (int(cx), int(cy)), ang

    return fallback_mask, None, None, None, 0.0


def build_aruco_detector(dict_id):
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin  = 3
    params.adaptiveThreshWinSizeMax  = 23
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate    = 0.02
    params.maxMarkerPerimeterRate    = 4.0
    params.polygonalApproxAccuracyRate = 0.03  # Keskin kareleri zorunlu tut
    params.errorCorrectionRate         = 0.1   # Hata düzeltme toleransını düşür (False positive testini zorlaştır)
    params.maxErroneousBitsInBorderRate = 0.05 # Etrafındaki beyaz çerçevenin kusursuz olmasını bekle (Sahte ID 0 gölgelerini engeller)
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_aruco(frame, mask, detector, clahe):
    """Sadece ham gri ve CLAHE ile tara. Aynı konuma birden çok ID atanmasını engelle."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    all_corners, all_ids = [], []
    seen_centers = []

    def _try(img):
        corners, ids, _ = detector.detectMarkers(img)
        if ids is not None:
            for i, mid in enumerate(ids):
                pts = corners[i][0]
                cx = pts[:, 0].mean()
                cy = pts[:, 1].mean()
                
                # Eğer 20 piksel yakınında zaten bir marker bulduysak atla (çift çizmeyi önler)
                is_duplicate = False
                for sx, sy in seen_centers:
                    if (cx - sx)**2 + (cy - sy)**2 < 400:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    seen_centers.append((cx, cy))
                    all_corners.append(corners[i])
                    all_ids.append(mid)

    # ArUco zaten kendi içinde adaptive thresholding yapıyor, bu yüzden
    # en temiz olan ham gri görüntüye öncelik veriyoruz.
    _try(gray)
    
    # Su altındaki bulanık veya soluk durumlar için CLAHE ile tekrar dene.
    _try(clahe.apply(gray))

    if all_ids:
        return all_corners, np.array(all_ids)
    return [], None


def draw_debug(frame, mask, contour, rect, center, angle,
               state, err, ordered, corners, ids, cfg):
    """HUD overlay çizimi."""
    fw, fh = cfg.frame_w, cfg.frame_h
    vis = frame.copy()

    # Yeşil maske overlay
    ch = np.zeros_like(vis)
    ch[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 0.7, ch, 0.3, 0)

    # Kontur ve bounding box
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 255, 60), 2)
    if rect is not None:
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(vis, [box], -1, (0, 200, 255), 2)

    # Merkez noktası ve hata çizgisi
    if center:
        cx, cy = center
        fc = (fw // 2, fh // 2)
        cv2.circle(vis, (cx, cy), 8, (0, 210, 255), -1)
        cv2.circle(vis, fc, 5, (255, 255, 255), -1)
        cv2.line(vis, (fc[0], cy), (cx, cy), (50, 80, 255), 2)
        # Dead zone & turn zone
        dz = int(fw * cfg.dead_zone)
        tz = int(fw * cfg.turn_zone)
        cv2.rectangle(vis, (fc[0]-dz, fh//5), (fc[0]+dz, 4*fh//5),
                      (80, 220, 80), 1)
        cv2.rectangle(vis, (fc[0]-tz, fh//5), (fc[0]+tz, 4*fh//5),
                      (50, 140, 50), 1)

    # ArUco çizimi
    if ids is not None:
        for i, corner in enumerate(corners):
            pts = corner[0].astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 200), 2)
            mcx = int(pts[:, 0].mean())
            mcy = int(pts[:, 1].mean())
            lbl = f"ID:{ids[i][0]}"
            cv2.putText(vis, lbl, (mcx - 15, mcy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    # Durum paneli
    STATE_COL = {
        ST_INIT: (180, 180, 180), ST_ARM: (255, 200, 0),
        ST_SEARCH: (0, 200, 255), ST_FOLLOW: (80, 220, 80),
        ST_MARKER: (0, 255, 200), ST_SURFACE: (200, 100, 255),
        ST_DONE: (255, 255, 255),
    }
    scol = STATE_COL.get(state, (200, 200, 200))
    lines = [
        (f"STATE  : {state}",                          scol),
        (f"ERR    : {err:+d} px",                      (200, 200, 200)),
        (f"ACI    : {angle:.1f} deg",                  (200, 200, 200)),
        (f"BORU   : {'TESPIT' if center else 'YOK'}",
         (80, 220, 80) if center else (0, 0, 220)),
        (f"MARKER : {len(ordered)} / {cfg.total_markers}",
         (80, 255, 150)),
    ]
    ph = len(lines) * 24 + 16
    cv2.rectangle(vis, (0, 0), (260, ph), (0, 0, 0), -1)
    cv2.rectangle(vis, (0, 0), (260, ph), (80, 80, 80), 1)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(vis, txt, (8, 18 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    # Alt bar (okunan marker ID'leri)
    ids_str = ",".join(str(x) for x in ordered) or "---"
    cv2.rectangle(vis, (0, fh - 30), (fw, fh), (0, 0, 0), -1)
    cv2.putText(vis, f"IDs: {ids_str}", (8, fh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 150), 1, cv2.LINE_AA)

    # Maske thumbnail (sağ alt köşe)
    th2, tw2 = fh // 4, fw // 4
    thumb = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (tw2, th2))
    cv2.putText(thumb, "MASKE", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    vis[fh - th2:fh, fw - tw2:fw] = thumb

    return vis


# ══════════════════════════════════════════════════════════════
# MAVLink YARDIMCILARI
# ══════════════════════════════════════════════════════════════
def set_param(master, name, value):
    master.mav.param_set_send(
        master.target_system, master.target_component,
        name.encode('utf-8'), value,
        mavutil.mavlink.MAV_PARAM_TYPE_REAL32)


def send_heartbeat(master):
    master.mav.heartbeat_send(
        mavutil.mavlink.MAV_TYPE_GCS,
        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)


def send_manual(master, x=0, y=0, z=500, r=0):
    master.mav.manual_control_send(master.target_system, x, y, z, r, 0)


def stop(master):
    send_heartbeat(master)
    send_manual(master, 0, 0, 500, 0)


def clamp(v, lo=-1000, hi=1000):
    return max(lo, min(hi, int(v)))


# ══════════════════════════════════════════════════════════════
# MOCK MAVLINK (dry-run modu)
# ══════════════════════════════════════════════════════════════
class _MockMAV:
    def __getattr__(self, name):
        def noop(*a, **kw): pass
        return noop


class MockMaster:
    target_system    = 1
    target_component = 1
    mav = _MockMAV()

    def wait_heartbeat(self):  pass
    def motors_armed(self):    return True
    def mode_mapping(self):    return {'MANUAL': 19}
    def recv_match(self, **kw): return None


# ══════════════════════════════════════════════════════════════
# ANA SINIF
# ══════════════════════════════════════════════════════════════
class TacAutonomous:

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self.state     = ST_INIT
        self.master    = None
        self.smoother  = Smoother(cfg.smooth_n)
        self.confirmer = ArucoConfirmer(cfg.aruco_confirm)
        self.detector  = build_aruco_detector(cfg.aruco_dict)
        self.clahe     = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.lo        = np.array([cfg.hmin, cfg.smin, cfg.vmin])
        self.hi        = np.array([cfg.hmax, cfg.smax, cfg.vmax])

        self.last_err      = 0
        self.last_angle    = 0.0
        self.pipe_lost_cnt = 0
        self.state_t       = time.time()
        self._running      = True
        self._lock         = threading.Lock()
        self._last_x = 0
        self._last_y = 0
        self._last_z = 500
        self._last_r = 0

        # Web yayın sunucusu
        self.server = None
        if cfg.stream_on:
            try:
                self.server = ThreadedHTTPServer(
                    ('', cfg.stream_port), MJPEGStreamer)
                threading.Thread(target=self.server.serve_forever,
                                 daemon=True).start()
                print(f"📡 Web sunucusu başlatıldı (Port: {cfg.stream_port})")
            except Exception as e:
                print(f"⚠️  Yayın başlatılamadı: {e}")

    # ── Bağlan & hazırla ──────────────────────────────────────
    def connect(self):
        cfg = self.cfg
        if cfg.dry_run:
            print("[DRY-RUN] Sahte MAVLink bağlantısı kullanılıyor")
            self.master = MockMaster()
            return

        print(f"Pixhawk'a bağlanılıyor: {cfg.device} @ {cfg.baud}...")
        self.master = mavutil.mavlink_connection(
            cfg.device, baud=cfg.baud, source_system=255)
        self.master.wait_heartbeat()
        print(f"✅ Bağlantı kuruldu! Sistem ID: {self.master.target_system}")

        # Failsafe parametreleri
        for name, val in [("FS_PILOT_EN", 0), ("FS_GCS_EN", 0),
                           ("ARMING_CHECK", 0), ("BRD_SAFETYENABLE", 0)]:
            set_param(self.master, name, val)
            print(f"  ⚙️  {name} = {val}")
        time.sleep(1)

        # MANUAL mod
        mode_id = self.master.mode_mapping()['STABILIZE']
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)
        print("  Mod: MANUAL")

    # ── Heartbeat thread ──────────────────────────────────────
    def _heartbeat_thread(self):
        while self._running:
            try:
                with self._lock:
                    x, y, z, r = (self._last_x, self._last_y,
                                  self._last_z, self._last_r)
                send_heartbeat(self.master)
                send_manual(self.master, x=x, y=y, z=z, r=r)
            except Exception as e:
                print(f"HB thread hata: {e}")
            time.sleep(0.05)  # 20 Hz

    # ── ARM ───────────────────────────────────────────────────
    def arm(self):
        print("ARM deneniyor...")
        start = time.time()
        while time.time() - start < 10:
            send_heartbeat(self.master)
            send_manual(self.master, 0, 0, 500, 0)

            self.master.mav.command_long_send(
                self.master.target_system, self.master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, FORCE_ARM_MAGIC, 0, 0, 0, 0, 0)

            msg = self.master.recv_match(
                type=['STATUSTEXT', 'COMMAND_ACK'], blocking=False)
            if msg:
                t = msg.get_type()
                if t == 'STATUSTEXT':
                    print(f"  Pixhawk: {msg.text}")
                elif t == 'COMMAND_ACK':
                    print(f"  ACK cmd={msg.command} result={msg.result}")

            if self.master.motors_armed():
                print("🟢 MOTORLAR ÇALIŞTI!")
                return True
            time.sleep(0.1)

        print("❌ ARM başarısız!")
        return False

    # ── Hareket gönder ────────────────────────────────────────
    def move(self, x=0, y=0, z=500, r=0):
        with self._lock:
            self._last_x = clamp(x)
            self._last_y = clamp(y)
            self._last_z = clamp(z, 0, 1000)
            self._last_r = clamp(r)

    def halt(self):
        self.move(0, 0, 500, 0)

    # ── State yardımcıları ────────────────────────────────────
    def _set_state(self, s):
        if s == self.state:
            return
        print(f"STATE: {self.state} → {s}")
        self.state   = s
        self.state_t = time.time()

    def _state_elapsed(self):
        return time.time() - self.state_t

    # ── Kontrol hesabı ────────────────────────────────────────
    def _compute_control(self, center, angle, fwd_spd):
        cfg = self.cfg
        raw_err = center[0] - cfg.frame_w // 2
        s_err   = self.smoother.update(raw_err)
        self.last_err   = s_err
        self.last_angle = angle

        fw = cfg.frame_w
        dead_p = int(fw * cfg.dead_zone)

        # Yan (Lateral) Yürüyüş İptal - Kamera İleri Baktığı İçin
        y = 0  

        # -- Direksiyon (Proportional Yaw) Mantığı --
        # S_err ne kadar büyükse o kadar sert direksiyon kırar.
        if abs(s_err) <= dead_p:
            r = 0
        else:
            # Hata oranını ekrana göre normalize et (-1.0 ile 1.0 arası)
            err_norm = s_err / (fw / 2.0)
            # Yönelim gücü = max(spd_yaw) * normalize edilmiş hata * kazanç(gain)
            turn_power = int(cfg.spd_yaw * err_norm * cfg.gain)
            
            # Limitleri uygula (Hızın cfg.spd_yaw değerini geçmesini engelle)
            if turn_power > cfg.spd_yaw: turn_power = cfg.spd_yaw
            if turn_power < -cfg.spd_yaw: turn_power = -cfg.spd_yaw
            r = turn_power

        # -- Dinamik Frenleme Sitemi (Araba Mantığı) --
        # Direksiyon (r) ne kadar çok kırılmışsa, ileri fırlamamak için gazı o kadar çok kes
        turn_ratio = abs(r) / max(1, cfg.spd_yaw)  # 0 ile 1 arası
        
        if turn_ratio > 0.6:
            # Çok keskin viraj: İleri gitmeyi durdur (Sadece Dön)
            fwd_spd = 0
        elif turn_ratio > 0.2:
            # Orta derece viraj: Yavaşla (Orantılı Fren)
            fwd_spd = int(fwd_spd * (1.0 - turn_ratio))

        # Yön çevirme kontrolü (Eğer araç sağa dönmesi gerekirken sola dönüyorsa rev_r=True yapın)
        if cfg.rev_x:
            fwd_spd = -fwd_spd
        if cfg.rev_r:
            r = -r

        return fwd_spd, y, r

    # ── State machine ────────────────────────────────────────
    def _state_machine(self, center, angle, ids_flat, ordered):
        cfg = self.cfg

        if self.state == ST_SEARCH:
            if center is not None:
                self._set_state(ST_FOLLOW)
                self.smoother.reset()
            else:
                self.move(x=0, y=0, r=cfg.spd_search_yaw)
            return

        if self.state == ST_FOLLOW:
            if self.pipe_lost_cnt > cfg.pipe_lost_limit:
                self._set_state(ST_SEARCH)
                self.halt()
                return
            if ids_flat and center is not None:
                self._set_state(ST_MARKER)
            if center is not None:
                x, y, r = self._compute_control(center, angle, cfg.spd_forward)
                self.move(x=x, y=y, r=r)
            return

        if self.state == ST_MARKER:
            if center is None and self.pipe_lost_cnt > 15:
                # Boruyu kaybettiyse görevi bitirme, boruyu aramaya geri dön!
                self._set_state(ST_SEARCH)
                self.halt()
                return

            if center is not None:
                x, y, r = self._compute_control(center, angle, cfg.spd_slow)
                self.move(x=x, y=y, r=r)
            else:
                self.halt()

            # Sadece belirlenen toplam marker sayısına ulaşınca GÖREVİ BİTİR ve YÜZEYE ÇIK
            if len(ordered) >= cfg.total_markers:
                print(f"Hedeflenen tüm {cfg.total_markers} marker okundu → GÖREV TAMAM (SURFACE)")
                self._set_state(ST_SURFACE)
            return

        if self.state == ST_SURFACE:
            if not cfg.surface_on_done:
                self._set_state(ST_DONE)
                self.halt()
                return
            self.move(x=0, y=0,
                      z=clamp(500 + cfg.spd_surface, 0, 1000), r=0)
            if self._state_elapsed() > cfg.surface_secs:
                self._set_state(ST_DONE)
            return

        if self.state == ST_DONE:
            self.halt()

    # ── Ana döngü ─────────────────────────────────────────────
    def run(self):
        cfg = self.cfg

        # Bağlan
        self.connect()

        # ARM
        self._set_state(ST_ARM)
        if not self.arm():
            return
        self._set_state(ST_SEARCH)

        # Heartbeat thread
        threading.Thread(target=self._heartbeat_thread, daemon=True).start()

        # Kamera veya video
        src = cfg.video if cfg.video else cfg.camera
        cap = cv2.VideoCapture(src)
        if not cfg.video:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.frame_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_h)
        if not cap.isOpened():
            print(f"Kaynak açılamadı: {src}")
            return
        if cfg.video:
            print(f"Video dosyası kullanılıyor: {cfg.video}")

        print("\n🚀 Otonom başlatıldı!")
        if cfg.stream_on:
            print(f"📡 İzleme Linki (Test için) : http://localhost:{cfg.stream_port}")
            print(f"📡 İzleme Linki (Denizaltı) : http://10.42.0.85:{cfg.stream_port}")
        print("Çıkmak için Ctrl+C  |  q=çık  p=duraklat\n")

        paused = False
        cam_fail_cnt = 0

        try:
            while self._running:
                t_start = time.time()

                if not paused:
                    ret, raw = cap.read()
                    if not ret:
                        if cfg.video:
                            print("Video bitti.")
                            break
                        cam_fail_cnt += 1
                        if cam_fail_cnt >= cfg.cam_fail_limit:
                            print("Kamera bağlantısı koptu!")
                            break
                        time.sleep(0.1)
                        continue
                    cam_fail_cnt = 0

                    frame = cv2.resize(raw, (cfg.frame_w, cfg.frame_h))
                    if cfg.flip is not None:
                        frame = cv2.flip(frame, cfg.flip)  # Kamerayı istenilen eksende çevir

                    # ── Görüntü işleme ──────────────────────
                    mask, contour, rect, center, angle = detect_pipe(
                        frame, self.lo, self.hi, cfg.min_area)

                    # Pipe lost sayacı
                    if center is not None:
                        self.pipe_lost_cnt = 0
                    else:
                        self.pipe_lost_cnt += 1

                    # ArUco
                    corners, ids = detect_aruco(
                        frame, mask, self.detector, self.clahe)
                    ids_flat = (ids.flatten().tolist()
                                if ids is not None else [])
                    ordered = self.confirmer.update(ids_flat)

                    for mid in self.confirmer.new_ids:
                        print(f"  ✅ MARKER ID {mid:3d} onaylandı | "
                              f"Toplam: {','.join(str(x) for x in ordered)}")

                    # ── State machine ────────────────────────
                    self._state_machine(center, angle, ids_flat, ordered)

                    if self.state == ST_DONE:
                        self._print_result(ordered)
                        break

                    # ── Debug görsel & Yayın ─────────────────
                    vis = draw_debug(frame, mask, contour, rect, center,
                                     angle, self.state, self.last_err,
                                     ordered, corners, ids, cfg)

                    # Web yayınına gönder
                    if cfg.stream_on:
                        with MJPEGStreamer.server_lock:
                            MJPEGStreamer.server_frame = vis.copy()
                            with self._lock:
                                MJPEGStreamer.server_x = self._last_x
                                MJPEGStreamer.server_y = self._last_y
                                MJPEGStreamer.server_z = self._last_z
                                MJPEGStreamer.server_r = self._last_r
                            MJPEGStreamer.server_state = self.state
                            MJPEGStreamer.server_err = self.last_err
                            MJPEGStreamer.server_pipe = (center is not None)

                    # Yerel ekran (SSH korumalı)
                    if cfg.show_display:
                        try:
                            #cv2.imshow("TAC Autonomous", vis)
                            k = cv2.waitKey(1) & 0xFF
                            if k == ord('q'):
                                print("Çıkılıyor...")
                                break
                            elif k == ord('p'):
                                paused = not paused
                                self.halt()
                                print("DURAKLADI" if paused else "DEVAM")
                        except Exception:
                            print("Headless ortam: Pencere devre dışı.")
                            cfg.show_display = False

                # Frame rate limiter (~30 FPS max)
                elapsed = time.time() - t_start
                sleep_t = max(0.0, 1.0 / 30.0 - elapsed)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except KeyboardInterrupt:
            print("\nCtrl+C — durduruluyor.")

        finally:
            self._running = False
            print("DISARM ediliyor...")
            self.halt()
            time.sleep(0.2)
            if self.master and not self.cfg.dry_run:
                self.master.mav.command_long_send(
                    self.master.target_system, self.master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 0, 0, 0, 0, 0, 0, 0)
            cap.release()
            cv2.destroyAllWindows()
            print("Bitti.")

    def _print_result(self, ordered):
        if hasattr(self, '_result_printed'):
            return
        self._result_printed = True
        sep = "=" * 52
        result = ",".join(str(x) for x in ordered)
        print(f"\n{sep}")
        print("  GÖREV TAMAMLANDI")
        print(f"  Marker sırası : {result or 'YOK'}")
        print(f"  Toplam marker : {len(ordered)}")
        print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════
# ARGPARSE & MAIN
# ══════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(
        description="TAC Challenge 2026 Otonom Boru Takip")
    ap.add_argument("--device",  default="/dev/ttyACM0",
                    help="Pixhawk serial portu")
    ap.add_argument("--baud",    type=int, default=115200)
    ap.add_argument("--camera",  default="0",
                    help="Kamera index (0,1,...) veya /dev/videoX")
    ap.add_argument("--hmin",    type=int, default=7)
    ap.add_argument("--hmax",    type=int, default=35)
    ap.add_argument("--smin",    type=int, default=110)
    ap.add_argument("--smax",    type=int, default=255)
    ap.add_argument("--vmin",    type=int, default=130)
    ap.add_argument("--vmax",    type=int, default=255)
    ap.add_argument("--fwd",     type=int, default=100, dest="spd_forward",
                    help="İleri hız gücü (0-1000)")
    ap.add_argument("--slow",    type=int, default=60, dest="spd_slow",
                    help="Marker yakını yavaş hız")
    ap.add_argument("--gain",    type=float, default=0.8,
                    help="P kontrol kazancı")
    ap.add_argument("--total-markers", type=int, default=5)
    ap.add_argument("--no-surface",  action="store_true")
    ap.add_argument("--no-display",  action="store_true",
                    help="Yerel pencereyi devre dışı bırak")
    ap.add_argument("--no-stream",   action="store_true",
                    help="Web yayınını devre dışı bırak")
    ap.add_argument("--port",        type=int, default=5000,
                    help="Web yayın portu")
    ap.add_argument("--dry-run",     action="store_true",
                    help="Pixhawk olmadan test modu (sahte MAVLink)")
    ap.add_argument("--flip",        type=int, default=None,
                    help="Görüntü ters çevirme ekseni: -1 (180 derece), 0 (Dikey), 1 (Yatay)")
    ap.add_argument("--video",       default="",
                    help="Kamera yerine video dosyası kullan")
    ap.add_argument("--aruco-dict",  default="ORIGINAL",
                    choices=["ORIGINAL", "4X4_100", "4X4_50", "5X5_100"])
    return ap.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    cfg.device      = args.device
    cfg.baud        = args.baud
    cfg.hmin        = args.hmin
    cfg.hmax        = args.hmax
    cfg.smin        = args.smin
    cfg.smax        = args.smax
    cfg.vmin        = args.vmin
    cfg.vmax        = args.vmax
    cfg.spd_forward = args.spd_forward
    cfg.spd_slow    = args.spd_slow
    cfg.gain        = args.gain
    cfg.total_markers   = args.total_markers
    cfg.surface_on_done = not args.no_surface
    cfg.show_display    = not args.no_display
    cfg.stream_on       = not args.no_stream
    cfg.stream_port     = args.port
    cfg.dry_run         = args.dry_run
    
    # Otomatik Flip Mantığı: Canlı kameraysa ters çevir, test videosu ise olduğu gibi bırak
    if not args.video and args.flip is None:
        cfg.flip = -1
    else:
        cfg.flip = args.flip

    cfg.video           = args.video

    try:
        cfg.camera = int(args.camera)
    except ValueError:
        cfg.camera = args.camera

    DICT_MAP = {
        "ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
        "4X4_100":  cv2.aruco.DICT_4X4_100,
        "4X4_50":   cv2.aruco.DICT_4X4_50,
        "5X5_100":  cv2.aruco.DICT_5X5_100,
    }
    cfg.aruco_dict = DICT_MAP[args.aruco_dict]

    bot = TacAutonomous(cfg)
    bot.run()


if __name__ == "__main__":
    main()
