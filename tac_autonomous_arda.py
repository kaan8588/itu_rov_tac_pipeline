#!/usr/bin/env python3
"""
tac_autonomous.py — TAC Challenge 2026
Pure Python otonom pipeline inspection.
pymavlink (serial) + OpenCV — ROS2 gerektirmez.

Kullanım:
  python tac_autonomous.py
  python tac_autonomous.py --device /dev/ttyACM0 --baud 115200
  python tac_autonomous.py --hmin 15 --hmax 38 --no-display

manual_control_send(x, y, z, r, buttons)
  x  : ileri/geri     -1000..+1000  (+ = ileri)
  y  : sağ/sol strafe -1000..+1000  (+ = sağ)
  z  : dikey          0..1000       (500 = dur)
  r  : yaw            -1000..+1000  (+ = sağa dön)
"""

import cv2
import numpy as np
import time
import os
import threading
import argparse
from collections import deque
from pymavlink import mavutil

os.environ['MAVLINK20'] = '1'

# ══════════════════════════════════════════════════════════════
# AYARLAR — argparse ile override edilebilir
# ══════════════════════════════════════════════════════════════
class Config:
    # Serial
    device   = '/dev/ttyACM0'
    baud     = 115200

    # Kamera
    camera   = 0          # USB kamera index veya '/dev/video0'
    frame_w  = 640
    frame_h  = 360

    # HSV — havuzda tuner ile kalibre et!
    hmin, hmax = 15, 40
    smin, smax = 80, 255
    vmin, vmax = 80, 255

    # Görüntü kontrol
    min_area   = 400
    dead_zone  = 0.10     # ±%10 → düz git
    turn_zone  = 0.28     # ±%28 → yavaş dön
    yaw_thr    = 20.0     # derece → yaw düzelt
    smooth_n   = 8

    # Hız değerleri (-1000..+1000)
    spd_forward  =  350   # normal ileri
    spd_slow     =  180   # marker yakınında yavaş
    spd_search   =  100   # arama modunda çok yavaş ileri
    spd_lateral  =  400   # tam lateral
    spd_soft_lat =  200   # yumuşak lateral
    spd_yaw      =  150   # yaw düzeltme
    spd_search_yaw= 120   # arama yaw
    spd_surface  = -400   # yukarı çık (z ekseni ters)

    # ArUco
    aruco_dict   = cv2.aruco.DICT_ARUCO_ORIGINAL
    aruco_confirm= 3

    # Görev
    total_markers   = 5
    surface_on_done = True
    surface_secs    = 8.0    # kaç saniye yüzeye çıkılsın

    # Pipe lost
    pipe_lost_limit = 15     # kaç frame üst üste kaybolunca SEARCH'e dön

    # Display
    show_display = True


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
            if 1 <= mid <= 99:
                self.counts[mid] = self.counts.get(mid, 0) + 1
                if self.counts[mid] >= self.confirm and mid not in self.seen:
                    self.seen.add(mid)
                    self.ordered.append(int(mid))
                    self.new_ids.append(int(mid))
        for mid in list(self.counts):
            if mid not in ids_set and mid not in self.seen:
                self.counts[mid] = 0
        return self.ordered


# ══════════════════════════════════════════════════════════════
# GÖRÜNTÜ İŞLEME
# ══════════════════════════════════════════════════════════════
def build_aruco_detector(dict_id):
    adict  = cv2.aruco.getPredefinedDictionary(dict_id)
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin    = 5
    params.adaptiveThreshWinSizeMax    = 23
    params.adaptiveThreshWinSizeStep   = 4
    params.minMarkerPerimeterRate      = 0.03
    params.errorCorrectionRate         = 0.6
    params.polygonalApproxAccuracyRate = 0.05
    return cv2.aruco.ArucoDetector(adict, params)


def detect_pipe(frame, lo, hi, min_area):
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.dilate(mask, k, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None, None, None, 0.0
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return mask, None, None, None, 0.0
    rect = cv2.minAreaRect(best)
    (cx, cy), (w, h), raw_ang = rect
    angle = (raw_ang + 90 if w < h else raw_ang) % 180
    return mask, best, (int(cx), int(cy)), rect, angle


def detect_aruco(frame, mask, detector, clahe):
    if mask is None or cv2.countNonZero(mask) < 100:
        return [], None
    k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    mask_big = cv2.dilate(mask, k, iterations=1)
    roi      = cv2.bitwise_and(frame, frame, mask=mask_big)
    gray     = clahe.apply(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def draw_debug(frame, mask, contour, rect, center, angle,
               state, err, ordered, corners, ids, cfg):
    fw, fh = cfg.frame_w, cfg.frame_h
    vis = frame.copy()

    ch = np.zeros_like(vis); ch[:,:,1] = mask
    vis = cv2.addWeighted(vis, 0.7, ch, 0.3, 0)

    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0,255,60), 2)
    if rect is not None:
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(vis, [box], -1, (0,200,255), 2)
    if center:
        cx, cy = center
        fc = (fw//2, fh//2)
        cv2.circle(vis, (cx,cy), 8, (0,210,255), -1)
        cv2.circle(vis, fc, 5, (255,255,255), -1)
        cv2.line(vis, (fc[0],cy), (cx,cy), (50,80,255), 2)
        dz = int(fw*cfg.dead_zone); tz = int(fw*cfg.turn_zone)
        cv2.rectangle(vis,(fc[0]-dz,fh//5),(fc[0]+dz,4*fh//5),(80,220,80),1)
        cv2.rectangle(vis,(fc[0]-tz,fh//5),(fc[0]+tz,4*fh//5),(50,140,50),1)

    if ids is not None:
        for i, corner in enumerate(corners):
            pts = corner[0].astype(np.int32)
            cv2.polylines(vis,[pts],True,(0,255,200),2)
            mcx=int(pts[:,0].mean()); mcy=int(pts[:,1].mean())
            lbl=f"ID:{ids[i][0]}"
            (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
            cv2.rectangle(vis,(mcx-tw//2-4,mcy-th-6),(mcx+tw//2+4,mcy+2),(0,0,0),-1)
            cv2.putText(vis,lbl,(mcx-tw//2,mcy-2),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,200),2)

    SCOL = {
        ST_INIT:"Bekle",ST_ARM:"ARM ediliyor",ST_SEARCH:"Boru aranıyor",
        ST_FOLLOW:"Takip","MARKER":"Marker okunuyor",
        ST_SURFACE:"Yüzeye çıkılıyor",ST_DONE:"TAMAMLANDI"
    }
    STATE_COL = {
        ST_INIT:(180,180,180),ST_ARM:(255,200,0),ST_SEARCH:(0,200,255),
        ST_FOLLOW:(80,220,80),ST_MARKER:(0,255,200),
        ST_SURFACE:(200,100,255),ST_DONE:(255,255,255)
    }
    scol = STATE_COL.get(state,(200,200,200))
    lines = [
        (f"STATE  : {state}",                       scol),
        (f"ERR    : {err:+d} px",                   (200,200,200)),
        (f"ACI    : {angle:.1f} deg",               (200,200,200)),
        (f"BORU   : {'TESPIT' if center else 'YOK'}",(80,220,80) if center else (0,0,220)),
        (f"MARKER : {len(ordered)} / {cfg.total_markers}", (80,255,150)),
    ]
    ph = len(lines)*24+16
    cv2.rectangle(vis,(0,0),(260,ph),(0,0,0),-1)
    cv2.rectangle(vis,(0,0),(260,ph),(80,80,80),1)
    for i,(txt,col) in enumerate(lines):
        cv2.putText(vis,txt,(8,18+i*24),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1,cv2.LINE_AA)

    ids_str = ",".join(str(x) for x in ordered) or "---"
    cv2.rectangle(vis,(0,fh-30),(fw,fh),(0,0,0),-1)
    cv2.putText(vis,f"IDs: {ids_str}",(8,fh-8),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(80,255,150),1,cv2.LINE_AA)

    th2,tw2 = fh//4, fw//4
    thumb = cv2.resize(cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR),(tw2,th2))
    cv2.putText(thumb,"MASKE",(4,14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)
    vis[fh-th2:fh,fw-tw2:fw] = thumb

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
    """
    x: ileri/geri  -1000..+1000
    y: sağ/sol     -1000..+1000
    z: dikey       0..1000  (500=dur)
    r: yaw         -1000..+1000
    """
    master.mav.manual_control_send(master.target_system, x, y, z, r, 0)


def stop(master):
    send_heartbeat(master)
    send_manual(master, x=0, y=0, z=500, r=0)


def clamp(v, lo=-1000, hi=1000):
    return max(lo, min(hi, int(v)))


# ══════════════════════════════════════════════════════════════
# ANA SINIF
# ══════════════════════════════════════════════════════════════
class TacAutonomous:

    def __init__(self, cfg: Config):
        self.cfg      = cfg
        self.state    = ST_INIT
        self.master   = None
        self.smoother = Smoother(cfg.smooth_n)
        self.confirmer= ArucoConfirmer(cfg.aruco_confirm)
        self.detector = build_aruco_detector(cfg.aruco_dict)
        self.clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        self.lo       = np.array([cfg.hmin, cfg.smin, cfg.vmin])
        self.hi       = np.array([cfg.hmax, cfg.smax, cfg.vmax])

        self.last_err     = 0
        self.last_angle   = 0.0
        self.pipe_lost_cnt= 0
        self.state_t      = time.time()
        self._running     = True
        self._last_x      = 0
        self._last_y      = 0
        self._last_r      = 0

    # ── Bağlan & hazırla ──────────────────────────────────────
    def connect(self):
        cfg = self.cfg
        print(f"Pixhawk'a bağlanılıyor: {cfg.device} @ {cfg.baud}...")
        self.master = mavutil.mavlink_connection(
            cfg.device, baud=cfg.baud, source_system=255)
        self.master.wait_heartbeat()
        print(f"Bağlantı kuruldu! Sistem ID: {self.master.target_system}")

        # Parametreler
        for name, val in [("FS_PILOT_EN", 0), ("FS_GCS_EN", 0),
                           ("ARMING_CHECK", 0), ("BRD_SAFETYENABLE", 0)]:
            set_param(self.master, name, val)
            print(f"  {name} = {val}")
        time.sleep(1)

        # MANUAL mod
        mode_id = self.master.mode_mapping()['MANUAL']
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)
        print("Mod: MANUAL")

    # ── Heartbeat thread (arka planda sürekli sinyal) ──────────
    def _heartbeat_thread(self):
        """Her 50ms'de heartbeat + son hareket komutunu tekrar gönder."""
        while self._running:
            try:
                send_heartbeat(self.master)
                send_manual(self.master,
                            x=self._last_x, y=self._last_y,
                            z=500,          r=self._last_r)
            except Exception:
                pass
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
                0, 1, 21196, 0, 0, 0, 0, 0)

            msg = self.master.recv_match(
                type=['STATUSTEXT','COMMAND_ACK'], blocking=False)
            if msg:
                t = msg.get_type()
                if t == 'STATUSTEXT':
                    print(f"  Pixhawk: {msg.text}")
                elif t == 'COMMAND_ACK':
                    print(f"  ACK cmd={msg.command} result={msg.result}")

            if self.master.motors_armed():
                print("MOTORLAR CALISTI!")
                return True
            time.sleep(0.1)

        print("ARM basarisiz!")
        return False

    # ── Hareket gönder ────────────────────────────────────────
    def move(self, x=0, y=0, r=0):
        """x=ileri, y=lateral, r=yaw. Heartbeat thread gönderir."""
        self._last_x = clamp(x)
        self._last_y = clamp(y)
        self._last_r = clamp(r)

    # ── Dur ───────────────────────────────────────────────────
    def halt(self):
        self.move(0, 0, 0)

    # ── State yardımcıları ────────────────────────────────────
    def _set_state(self, s):
        if s == self.state: return
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
        dead = int(fw * cfg.dead_zone)
        turn = int(fw * cfg.turn_zone)

        # Lateral
        if   abs(s_err) <= dead: y = 0
        elif abs(s_err) <= turn: y =  cfg.spd_soft_lat if s_err > 0 else -cfg.spd_soft_lat
        else:                    y =  cfg.spd_lateral   if s_err > 0 else -cfg.spd_lateral

        # Yaw (boru açısına küçük düzeltme)
        ah = min(angle, 180 - angle)
        if ah >= cfg.yaw_thr:
            r = cfg.spd_yaw if angle < 90 else -cfg.spd_yaw
        else:
            r = 0

        return fwd_spd, y, r

    # ── State machine (her frame çağrılır) ───────────────────
    def _state_machine(self, center, angle, ids_flat, ordered):
        cfg = self.cfg

        if self.state == ST_SEARCH:
            if center is not None:
                self._set_state(ST_FOLLOW)
                self.smoother.reset()
            else:
                self.move(x=cfg.spd_search, y=0, r=cfg.spd_search_yaw)
            return

        if self.state == ST_FOLLOW:
            if self.pipe_lost_cnt > cfg.pipe_lost_limit:
                self._set_state(ST_SEARCH)
                self.halt()
                return
            if ids_flat:
                self._set_state(ST_MARKER)
            if center is not None:
                x, y, r = self._compute_control(center, angle, cfg.spd_forward)
                self.move(x=x, y=y, r=r)
            return

        if self.state == ST_MARKER:
            # Boru kayboldu + marker beklentisi doldu → surface
            if center is None and self.pipe_lost_cnt > 8:
                if len(ordered) > 0:
                    self._set_state(ST_SURFACE)
                else:
                    self._set_state(ST_SEARCH)
                self.halt()
                return

            if center is not None:
                x, y, r = self._compute_control(center, angle, cfg.spd_slow)
                self.move(x=x, y=y, r=r)
            else:
                self.halt()

            if len(ordered) >= cfg.total_markers:
                print(f"Tüm {cfg.total_markers} marker okundu → SURFACE")
                self._set_state(ST_SURFACE)
            return

        if self.state == ST_SURFACE:
            if not cfg.surface_on_done:
                self._set_state(ST_DONE)
                self.halt()
                return
            # z ekseni: 500 merkez, 500'ün altı = aşağı, üstü = yukarı
            # yukarı çıkmak için z > 500
            send_heartbeat(self.master)
            send_manual(self.master, x=0, y=0,
                        z=clamp(500 + abs(cfg.spd_surface), 0, 1000), r=0)
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

        # Heartbeat thread başlat
        hb_thread = threading.Thread(target=self._heartbeat_thread, daemon=True)
        hb_thread.start()

        # Kamera
        cap = cv2.VideoCapture(cfg.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cfg.frame_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_h)
        if not cap.isOpened():
            print(f"Kamera açılamadı: {cfg.camera}")
            return

        print("\nBaşladı! q=çık  p=duraklat\n")
        paused = False

        try:
            while self._running:
                if not paused:
                    ret, raw = cap.read()
                    if not ret:
                        print("Kamera frame'i alınamadı!")
                        time.sleep(0.1)
                        continue

                    frame = cv2.resize(raw, (cfg.frame_w, cfg.frame_h))

                    # ── Görüntü işleme ──────────────────────
                    mask, contour, center, rect, angle = detect_pipe(
                        frame, self.lo, self.hi, cfg.min_area)

                    # Pipe lost sayacı
                    if center is not None:
                        self.pipe_lost_cnt = 0
                    else:
                        self.pipe_lost_cnt += 1

                    # ArUco
                    corners, ids = detect_aruco(
                        frame, mask, self.detector, self.clahe)
                    ids_flat = ids.flatten().tolist() if ids is not None else []
                    ordered  = self.confirmer.update(ids_flat)

                    for mid in self.confirmer.new_ids:
                        print(f"  MARKER ID {mid:3d} onaylandi | "
                              f"Toplam: {','.join(str(x) for x in ordered)}")

                    # ── State machine ────────────────────────
                    if self.state not in (ST_DONE, ST_SURFACE):
                        self._state_machine(center, angle, ids_flat, ordered)

                    elif self.state == ST_SURFACE:
                        self._state_machine(center, angle, ids_flat, ordered)

                    elif self.state == ST_DONE:
                        self._print_result(ordered)
                        break

                    # ── Debug görsel ─────────────────────────
                    if cfg.show_display:
                        vis = draw_debug(frame, mask, contour, rect, center,
                                         angle, self.state, self.last_err,
                                         ordered, corners, ids, cfg)
                        cv2.imshow("TAC Autonomous", vis)

                k = cv2.waitKey(1) & 0xFF
                if   k == ord('q'):
                    print("Çıkılıyor...")
                    break
                elif k == ord('p'):
                    paused = not paused
                    self.halt()
                    print("DURAKLADI" if paused else "DEVAM")

        except KeyboardInterrupt:
            print("\nCtrl+C — durduruluyor.")

        finally:
            self._running = False
            print("DISARM ediliyor...")
            self.halt()
            time.sleep(0.2)
            if self.master:
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
        print("  GOREV TAMAMLANDI")
        print(f"  Marker sirasi : {result or 'YOK'}")
        print(f"  Toplam marker : {len(ordered)}")
        print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════
# ARGPARSE & MAIN
# ══════════════════════════════════════════════════════════════
def parse_args():
    ap = argparse.ArgumentParser(description="TAC Challenge 2026 Otonom")
    ap.add_argument("--device",       default="/dev/ttyACM0")
    ap.add_argument("--baud",   type=int, default=115200)
    ap.add_argument("--camera", default="0",
                    help="Kamera index (0,1,...) veya /dev/videoX")
    ap.add_argument("--hmin",   type=int, default=15)
    ap.add_argument("--hmax",   type=int, default=40)
    ap.add_argument("--smin",   type=int, default=80)
    ap.add_argument("--smax",   type=int, default=255)
    ap.add_argument("--vmin",   type=int, default=80)
    ap.add_argument("--vmax",   type=int, default=255)
    ap.add_argument("--fwd",    type=int, default=350,  dest="spd_forward")
    ap.add_argument("--slow",   type=int, default=180,  dest="spd_slow")
    ap.add_argument("--total-markers", type=int, default=5)
    ap.add_argument("--no-surface", action="store_true")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--aruco-dict", default="ORIGINAL",
                    choices=["ORIGINAL","4X4_100","4X4_50","5X5_100"])
    return ap.parse_args()


def main():
    args = parse_args()
    cfg  = Config()

    cfg.device     = args.device
    cfg.baud       = args.baud
    cfg.hmin, cfg.hmax = args.hmin, args.hmax
    cfg.smin, cfg.smax = args.smin, args.smax
    cfg.vmin, cfg.vmax = args.vmin, args.vmax
    cfg.spd_forward    = args.spd_forward
    cfg.spd_slow       = args.spd_slow
    cfg.total_markers  = args.total_markers
    cfg.surface_on_done= not args.no_surface
    cfg.show_display   = not args.no_display

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
