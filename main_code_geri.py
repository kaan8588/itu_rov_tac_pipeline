#!/usr/bin/env python3
"""
tac_autonomous_ros2.py — TAC Challenge 2026
ROS2 entegreli otonom pipeline inspection.
pymavlink (serial) + OpenCV + ROS2 (rclpy)

ROS2 Özellikleri:
  - /tac/state          : String    — mevcut state (SEARCH, FOLLOW, vb.)
  - /tac/markers        : String    — tespit edilen marker ID'leri (virgülle ayrılmış)
  - /tac/marker_count   : Int32     — toplam onaylanmış marker sayısı
  - /tac/pipe_detected  : Bool      — boru görülüyor mu
  - /tac/pipe_error     : Int32     — boru merkez hatası (px)
  - /tac/pipe_angle     : Float32   — boru açısı (derece)
  - /tac/cmd_vel        : Twist     — hareket komutu (debug/loglama için)
  - /tac/debug_image    : Image     — debug görüntüsü (rviz2'de izle)
  - /camera/image_raw   : Image     — (subscribe) dış kamera kaynağı (opsiyonel)

Parametreler — tümü ROS2 param olarak override edilebilir:
  ros2 run tac_challenge tac_autonomous_ros2 --ros-args \
    -p device:=/dev/ttyACM0 \
    -p hmin:=15 -p hmax:=82 \
    -p spd_forward:=350

Kullanım:
  # Direkt çalıştır (standalone):
  python3 tac_autonomous_ros2.py

  # ROS2 launch ile:
  ros2 run tac_challenge tac_autonomous_ros2

  # Parametrelerle:
  ros2 run tac_challenge tac_autonomous_ros2 --ros-args \
    -p device:=/dev/ttyACM0 -p baud:=115200 \
    -p hmin:=15 -p hmax:=82 -p smin:=101 -p vmin:=143 \
    -p use_ros_camera:=true -p no_display:=true
"""

import cv2
import numpy as np
import time
import os
import threading
import argparse
from collections import deque
from pymavlink import mavutil

# ROS2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String, Int32, Bool, Float32
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

os.environ['MAVLINK20'] = '1'


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
ST_RETURN  = "RETURN"
ST_HOME    = "HOME"


# ══════════════════════════════════════════════════════════════
# YARDIMCI SINIFLAR (orijinalden aynen)
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
    def __init__(self, confirm=3, decay=2):
        self.confirm  = confirm
        self.decay    = decay      # kaç frame görmeyince 1 düşür
        self.counts   = {}
        self.missing  = {}         # art arda kaç frame görülmedi
        self.seen     = set()
        self.ordered  = []
        self.new_ids  = []

    def update(self, ids_list):
        self.new_ids = []
        ids_set = set(ids_list)

        # Görülen marker'ların sayacını artır
        for mid in ids_set:
            if mid < 0:
                continue
            self.counts[mid] = self.counts.get(mid, 0) + 1
            self.missing[mid] = 0  # görüldü, missing sıfırla
            if self.counts[mid] >= self.confirm and mid not in self.seen:
                self.seen.add(mid)
                self.ordered.append(int(mid))
                self.new_ids.append(int(mid))

        # Görülmeyen marker'ların sayacını yavaşça düşür
        for mid in list(self.counts):
            if mid not in ids_set and mid not in self.seen:
                self.missing[mid] = self.missing.get(mid, 0) + 1
                if self.missing[mid] >= self.decay:
                    self.counts[mid] = max(0, self.counts[mid] - 1)
                    self.missing[mid] = 0

        return self.ordered


# ══════════════════════════════════════════════════════════════
# GÖRÜNTÜ İŞLEME (orijinalden aynen)
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


def detect_pipe(frame, params):
    """params: dict-like with hmin,hmax,smin,smax,vmin,vmax,min_area,min_ratio"""
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    lo   = np.array([params['hmin'], params['smin'], params['vmin']])
    hi   = np.array([params['hmax'], params['smax'], params['vmax']])
    mask = cv2.inRange(hsv, lo, hi)

    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.dilate(mask, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask, None, None, None, 0.0

    valid = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < params['min_area']:
            continue
        _, (w, h), _ = cv2.minAreaRect(c)
        if w == 0 or h == 0:
            continue
        ratio = max(w, h) / min(w, h)
        if ratio < params['min_ratio']:
            continue
        valid.append(c)

    if not valid:
        return mask, None, None, None, 0.0

    best = max(valid, key=cv2.contourArea)
    rect = cv2.minAreaRect(best)
    (cx, cy), (w, h), raw_ang = rect
    angle = (raw_ang + 90 if w < h else raw_ang) % 180
    # En yakın piksel: konturun en alt noktası (y en büyük = kameraya en yakın)
    bottom_pt = tuple(best[best[:, :, 1].argmax()][0])
    target = (bottom_pt[0], bottom_pt[1])
    return mask, best, target, rect, angle


def detect_aruco(frame, mask, detector, clahe):
    gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and mask is not None and cv2.countNonZero(mask) > 100:
        k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
        mask_big = cv2.dilate(mask, k, iterations=1)
        valid_corners, valid_ids = [], []
        for i, corner in enumerate(corners):
            cx = int(corner[0][:, 0].mean())
            cy = int(corner[0][:, 1].mean())
            if 0 <= cy < mask_big.shape[0] and 0 <= cx < mask_big.shape[1]:
                if mask_big[cy, cx] > 0:
                    valid_corners.append(corner)
                    valid_ids.append(ids[i])
        if valid_ids:
            return valid_corners, np.array(valid_ids)
        else:
            return [], None

    return corners, ids


def draw_debug(frame, mask, contour, rect, center, angle,
               state, err, ordered, corners, ids, params):
    fw, fh = params['frame_w'], params['frame_h']
    vis = frame.copy()

    ch = np.zeros_like(vis)
    ch[:, :, 1] = mask
    vis = cv2.addWeighted(vis, 0.7, ch, 0.3, 0)

    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 255, 60), 2)
    if rect is not None:
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.drawContours(vis, [box], -1, (0, 200, 255), 2)
    if center:
        cx, cy = center
        fc = (fw // 2, fh // 2)
        cv2.circle(vis, (cx, cy), 8, (0, 210, 255), -1)
        cv2.circle(vis, fc, 5, (255, 255, 255), -1)
        cv2.line(vis, (fc[0], cy), (cx, cy), (50, 80, 255), 2)
        dz = int(fw * params['dead_zone'])
        tz = int(fw * params['turn_zone'])
        cv2.rectangle(vis, (fc[0]-dz, fh//5), (fc[0]+dz, 4*fh//5), (80, 220, 80), 1)
        cv2.rectangle(vis, (fc[0]-tz, fh//5), (fc[0]+tz, 4*fh//5), (50, 140, 50), 1)

    if ids is not None:
        for i, corner in enumerate(corners):
            pts = corner[0].astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 200), 2)
            mcx = int(pts[:, 0].mean())
            mcy = int(pts[:, 1].mean())
            lbl = f"ID:{ids[i][0]}"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (mcx-tw//2-4, mcy-th-6), (mcx+tw//2+4, mcy+2), (0, 0, 0), -1)
            cv2.putText(vis, lbl, (mcx-tw//2, mcy-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

    STATE_COL = {
        ST_INIT: (180, 180, 180), ST_ARM: (255, 200, 0), ST_SEARCH: (0, 200, 255),
        ST_FOLLOW: (80, 220, 80), ST_MARKER: (0, 255, 200),
        ST_SURFACE: (200, 100, 255), ST_DONE: (255, 255, 255),
        ST_RETURN: (255, 140, 0), ST_HOME: (0, 255, 255)
    }
    scol = STATE_COL.get(state, (200, 200, 200))
    ah = min(angle, 180 - angle)
    is_corner = ah >= params['corner_angle_thr']
    corner_txt = f"KOSE {'SAG' if angle < 90 else 'SOL'}!" if is_corner else "duz"

    total_markers = params.get('total_markers', 0)
    lines = [
        (f"STATE  : {state}", scol),
        (f"ERR    : {err:+d} px", (200, 200, 200)),
        (f"ACI    : {angle:.1f} deg  {corner_txt}",
         (0, 80, 255) if is_corner else (200, 200, 200)),
        (f"BORU   : {'TESPIT' if center else 'YOK'}",
         (80, 220, 80) if center else (0, 0, 220)),
        (f"MARKER : {len(ordered)}"
         + (f" / {total_markers}" if total_markers > 0 else ""),
         (80, 255, 150)),
    ]
    ph = len(lines) * 18 + 10
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (200, ph), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
    for i, (txt, col) in enumerate(lines):
        cv2.putText(vis, txt, (6, 14+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    ids_str = ",".join(str(x) for x in ordered) or "---"
    cv2.rectangle(vis, (0, fh-30), (fw, fh), (0, 0, 0), -1)
    cv2.putText(vis, f"IDs: {ids_str}", (8, fh-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 150), 1, cv2.LINE_AA)

    th2, tw2 = fh // 4, fw // 4
    thumb = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (tw2, th2))
    cv2.putText(thumb, "MASKE", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
    vis[fh-th2:fh, fw-tw2:fw] = thumb

    return vis


# ══════════════════════════════════════════════════════════════
# MAVLink YARDIMCILARI (orijinalden aynen)
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
    send_manual(master, x=0, y=0, z=500, r=0)

def clamp(v, lo=-1000, hi=1000):
    return max(lo, min(hi, int(v)))


# ══════════════════════════════════════════════════════════════
# ROS2 NODE
# ══════════════════════════════════════════════════════════════
class TacAutonomousNode(Node):
    """
    ROS2 Node — TAC Challenge 2026 Otonom Pipeline Inspection.

    Orijinal tac_autonomous.py'nin tüm mantığını içerir.
    Ek olarak ROS2 topic'leri publish eder ve opsiyonel olarak
    ROS2 kamera topic'inden subscribe olabilir.
    """

    def __init__(self):
        super().__init__('tac_autonomous')

        # ── ROS2 Parametreleri (tümü declare & okunur) ────────
        self._declare_params()
        self._read_params()

        # ── Publishers ────────────────────────────────────────
        self.pub_state        = self.create_publisher(String,  '/tac/state',         10)
        self.pub_markers      = self.create_publisher(String,  '/tac/markers',       10)
        self.pub_marker_count = self.create_publisher(Int32,   '/tac/marker_count',  10)
        self.pub_pipe_detected= self.create_publisher(Bool,    '/tac/pipe_detected', 10)
        self.pub_pipe_error   = self.create_publisher(Int32,   '/tac/pipe_error',    10)
        self.pub_pipe_angle   = self.create_publisher(Float32, '/tac/pipe_angle',    10)
        self.pub_cmd_vel      = self.create_publisher(Twist,   '/tac/cmd_vel',       10)
        self.pub_debug_image  = self.create_publisher(Image,   '/tac/debug_image',   5)

        # ── Subscriber (opsiyonel ROS2 kamera) ────────────────
        self.bridge = CvBridge()
        self._ros_frame = None
        self._ros_frame_lock = threading.Lock()

        if self.use_ros_camera:
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
            self.create_subscription(
                Image, self.camera_topic, self._camera_cb, qos)
            self.get_logger().info(
                f"ROS2 kamera: {self.camera_topic} dinleniyor")

        # ── State machine & helper nesneleri ──────────────────
        self.state           = ST_INIT
        self.master          = None
        self.smoother        = Smoother(self.smooth_n)
        self.confirmer       = ArucoConfirmer(self.aruco_confirm)
        self.detector        = build_aruco_detector(self.aruco_dict)
        self.clahe           = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self.follow_start_time = None
        self.follow_elapsed    = 0.0

        self.last_err        = 0
        self.last_angle      = 0.0
        self.pipe_lost_cnt   = 0
        self.pipe_end_cnt    = 0
        self.last_corner_dir = 0
        self.corner_cnt      = 0
        self.state_t         = time.time()
        self._running        = True
        self._last_x         = 0
        self._last_y         = 0
        self._last_r         = 0
        self._result_printed = False

        self.move_log           = []
        self._return_log        = []
        self._return_idx        = 0
        self._return_step_start = 0.0
        self._last_frame_time   = time.time()

        self.get_logger().info("TacAutonomousNode başlatıldı")

    # ── ROS2 Param tanımları ──────────────────────────────────
    def _declare_params(self):
        # MAVLink
        self.declare_parameter('device',           '/dev/ttyACM0')
        self.declare_parameter('baud',             115200)

        # Kamera
        self.declare_parameter('camera',           0)
        self.declare_parameter('frame_w',          640)
        self.declare_parameter('frame_h',          360)

        # ROS2 kamera entegrasyonu
        self.declare_parameter('use_ros_camera',   False)
        self.declare_parameter('camera_topic',     '/camera/image_raw')

        # HSV
        self.declare_parameter('hmin',             15)
        self.declare_parameter('hmax',             82)
        self.declare_parameter('smin',             101)
        self.declare_parameter('smax',             255)
        self.declare_parameter('vmin',             143)
        self.declare_parameter('vmax',             255)

        # Kontrol
        self.declare_parameter('min_area',         50)
        self.declare_parameter('dead_zone',        0.12)
        self.declare_parameter('turn_zone',        0.18)
        self.declare_parameter('yaw_thr',          10.0)
        self.declare_parameter('smooth_n',         8)
        self.declare_parameter('min_ratio',        1.5)

        # Köşe
        self.declare_parameter('corner_angle_thr', 35.0)
        self.declare_parameter('corner_slow_spd',  80)
        self.declare_parameter('corner_yaw_spd',   120)

        # Hız
        self.declare_parameter('spd_forward',      350)
        self.declare_parameter('spd_slow',         180)
        self.declare_parameter('spd_search',       150)
        self.declare_parameter('spd_lateral',      430)
        self.declare_parameter('spd_soft_lat',     250)
        self.declare_parameter('spd_yaw',          40)
        self.declare_parameter('spd_search_yaw',   90)
        self.declare_parameter('spd_surface',     -400)

        # ArUco
        self.declare_parameter('aruco_dict',       'ORIGINAL')
        self.declare_parameter('aruco_confirm',    2)

        # Görev
        self.declare_parameter('total_markers',    0)
        self.declare_parameter('surface_on_done',  True)
        self.declare_parameter('surface_secs',     8.0)
        self.declare_parameter('pipe_lost_limit',  60)
        self.declare_parameter('pipe_end_limit',   60)
        self.declare_parameter('result_file',      'pipeline_result.txt')

        # Display
        self.declare_parameter('no_display',       False)
        self.declare_parameter('publish_debug',    True)
        self.declare_parameter('return_cx', 1.0)
        self.declare_parameter('return_cy', 1.0)
        self.declare_parameter('return_cr', 1.0)

    def _read_params(self):
        g = self.get_parameter

        self.device         = g('device').value
        self.baud           = g('baud').value
        self.camera_id      = g('camera').value
        self.frame_w        = g('frame_w').value
        self.frame_h        = g('frame_h').value

        self.use_ros_camera = g('use_ros_camera').value
        self.camera_topic   = g('camera_topic').value

        self.hmin           = g('hmin').value
        self.hmax           = g('hmax').value
        self.smin           = g('smin').value
        self.smax           = g('smax').value
        self.vmin           = g('vmin').value
        self.vmax           = g('vmax').value

        self.min_area       = g('min_area').value
        self.dead_zone      = g('dead_zone').value
        self.turn_zone      = g('turn_zone').value
        self.yaw_thr        = g('yaw_thr').value
        self.smooth_n       = g('smooth_n').value
        self.min_ratio      = g('min_ratio').value

        self.corner_angle_thr = g('corner_angle_thr').value
        self.corner_slow_spd  = g('corner_slow_spd').value
        self.corner_yaw_spd   = g('corner_yaw_spd').value

        self.spd_forward    = g('spd_forward').value
        self.spd_slow       = g('spd_slow').value
        self.spd_search     = g('spd_search').value
        self.spd_lateral    = g('spd_lateral').value
        self.spd_soft_lat   = g('spd_soft_lat').value
        self.spd_yaw        = g('spd_yaw').value
        self.spd_search_yaw = g('spd_search_yaw').value
        self.spd_surface    = g('spd_surface').value

        dict_map = {
            "ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
            "4X4_100":  cv2.aruco.DICT_4X4_100,
            "4X4_50":   cv2.aruco.DICT_4X4_50,
            "5X5_100":  cv2.aruco.DICT_5X5_100,
        }
        self.aruco_dict     = dict_map.get(g('aruco_dict').value,
                                            cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_confirm  = g('aruco_confirm').value

        self.total_markers  = g('total_markers').value
        self.surface_on_done= g('surface_on_done').value
        self.surface_secs   = g('surface_secs').value
        self.pipe_lost_limit= g('pipe_lost_limit').value
        self.pipe_end_limit = g('pipe_end_limit').value
        self.result_file    = g('result_file').value

        self.show_display   = not g('no_display').value
        self.publish_debug  = g('publish_debug').value
        self.return_cx      = g('return_cx').value
        self.return_cy      = g('return_cy').value
        self.return_cr      = g('return_cr').value

    def _get_pipe_params(self):
        """detect_pipe ve draw_debug için parametre dict'i döndür."""
        return {
            'hmin': self.hmin, 'hmax': self.hmax,
            'smin': self.smin, 'smax': self.smax,
            'vmin': self.vmin, 'vmax': self.vmax,
            'min_area': self.min_area,
            'min_ratio': self.min_ratio,
            'frame_w': self.frame_w, 'frame_h': self.frame_h,
            'dead_zone': self.dead_zone, 'turn_zone': self.turn_zone,
            'corner_angle_thr': self.corner_angle_thr,
            'total_markers': self.total_markers,
        }

    # ── ROS2 kamera callback ─────────────────────────────────
    def _camera_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self._ros_frame_lock:
                self._ros_frame = frame
        except Exception as e:
            self.get_logger().warn(f"Kamera dönüşüm hatası: {e}")

    def _get_ros_frame(self):
        with self._ros_frame_lock:
            f = self._ros_frame
            self._ros_frame = None
            return f

    # ── Publisher yardımcıları ────────────────────────────────
    def _publish_state(self):
        msg = String()
        msg.data = self.state
        self.pub_state.publish(msg)

    def _publish_markers(self, ordered):
        msg_str = String()
        msg_str.data = ",".join(str(x) for x in ordered)
        self.pub_markers.publish(msg_str)

        msg_cnt = Int32()
        msg_cnt.data = len(ordered)
        self.pub_marker_count.publish(msg_cnt)

    def _publish_pipe_info(self, center, angle):
        msg_det = Bool()
        msg_det.data = center is not None
        self.pub_pipe_detected.publish(msg_det)

        msg_err = Int32()
        msg_err.data = self.last_err
        self.pub_pipe_error.publish(msg_err)

        msg_ang = Float32()
        msg_ang.data = float(angle)
        self.pub_pipe_angle.publish(msg_ang)

    def _publish_cmd_vel(self, x, y, r):
        msg = Twist()
        msg.linear.x  = float(x)    # ileri/geri
        msg.linear.y  = float(y)    # lateral
        msg.angular.z = float(r)    # yaw
        self.pub_cmd_vel.publish(msg)

    def _publish_debug_image(self, vis):
        if not self.publish_debug:
            return
        try:
            img_msg = self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = 'camera'
            self.pub_debug_image.publish(img_msg)
        except Exception as e:
            self.get_logger().warn(f"Debug image publish hatası: {e}")

    # ── MAVLink bağlantı ─────────────────────────────────────
    def connect(self):
        self.get_logger().info(
            f"Pixhawk'a bağlanılıyor: {self.device} @ {self.baud}...")
        self.master = mavutil.mavlink_connection(
            self.device, baud=self.baud, source_system=255)
        self.master.wait_heartbeat()
        self.get_logger().info(
            f"Bağlantı kuruldu! Sistem ID: {self.master.target_system}")

        for name, val in [("FS_PILOT_EN", 0), ("FS_GCS_EN", 0),
                           ("ARMING_CHECK", 0), ("BRD_SAFETYENABLE", 0)]:
            set_param(self.master, name, val)
            self.get_logger().info(f"  {name} = {val}")
        time.sleep(1)

        mode_id = self.master.mode_mapping()['MANUAL']
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)
        self.get_logger().info("Mod: MANUAL")

    # ── Heartbeat thread ─────────────────────────────────────
    def _heartbeat_thread(self):
        while self._running:
            try:
                send_heartbeat(self.master)
                send_manual(self.master,
                            x=self._last_x, y=self._last_y,
                            z=500, r=self._last_r)
            except Exception:
                pass
            time.sleep(0.05)

    # ── ARM ───────────────────────────────────────────────────
    def arm(self):
        self.get_logger().info("ARM deneniyor...")
        start = time.time()
        while time.time() - start < 10:
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
                self.get_logger().info("MOTORLAR ÇALIŞTI!")
                return True
            time.sleep(0.1)
        self.get_logger().error("ARM başarısız!")
        return False

    # ── Hareket ───────────────────────────────────────────────
    def move(self, x=0, y=0, r=0):
        self._last_x = clamp(x)
        self._last_y = clamp(y)
        self._last_r = clamp(r)
        self._publish_cmd_vel(self._last_x, self._last_y, self._last_r)

    def halt(self):
        self.move(0, 0, 0)

    # ── State yardımcıları ────────────────────────────────────
    def _set_state(self, s):
        if s == self.state:
            return
        if s == ST_FOLLOW:
            self.follow_start_time = time.time()
        elif self.state == ST_FOLLOW and s != ST_FOLLOW:
            self.follow_elapsed += time.time() - (self.follow_start_time or time.time())
        if s == ST_RETURN:
            self._return_log        = list(reversed(self.move_log))
            self._return_idx        = 0
            self._return_step_start = time.time()
        self.get_logger().info(f"STATE: {self.state} → {s}")
        self.state   = s
        self.state_t = time.time()
        self._publish_state()

    def _state_elapsed(self):
        return time.time() - self.state_t

    # ── Kontrol hesabı (orijinalden aynen) ────────────────────
    def _compute_control(self, center, angle, fwd_spd):
        raw_err = center[0] - self.frame_w // 2
        s_err   = self.smoother.update(raw_err)
        self.last_err   = s_err
        self.last_angle = angle

        fw   = self.frame_w
        dead = int(fw * self.dead_zone)
        turn = int(fw * self.turn_zone)

        if   abs(s_err) <= dead: y = 0
        elif abs(s_err) <= turn: y = self.spd_soft_lat if s_err > 0 else -self.spd_soft_lat
        else:                    y = self.spd_lateral  if s_err > 0 else -self.spd_lateral

        ah = min(angle, 180 - angle)
        is_corner = ah >= self.corner_angle_thr

        if is_corner:
            self.corner_cnt += 1
            self.last_corner_dir = 1 if angle < 90 else -1
            x = self.corner_slow_spd
            r = self.corner_yaw_spd if angle < 90 else -self.corner_yaw_spd
        else:
            self.corner_cnt = 0
            r = (self.spd_yaw if angle < 90 else -self.spd_yaw) \
                if ah >= self.yaw_thr else 0
            x = fwd_spd

        return x, y, r

    # ── State machine (orijinalden aynen) ─────────────────────
    def _state_machine(self, center, angle, ids_flat, ordered):
        # SEARCH
        if self.state == ST_SEARCH:
            if center is not None:
                self._set_state(ST_FOLLOW)
                self.smoother.reset()
                self.pipe_end_cnt = 0
            else:
                if self.last_corner_dir != 0:
                    r_search = self.corner_yaw_spd * self.last_corner_dir
                    self.move(x=self.spd_search, y=0, r=r_search)
                else:
                    self.move(x=self.spd_search, y=0, r=self.spd_search_yaw)
            return

        # FOLLOW
        if self.state == ST_FOLLOW:
            if self.pipe_lost_cnt > self.pipe_lost_limit:
                self._set_state(ST_SEARCH)
                self.halt()
                return

            if center is not None:
                spd = self.spd_slow if ids_flat else self.spd_forward
                x, y, r = self._compute_control(center, angle, spd)
                self.move(x=x, y=y, r=r)

            if center is None:
                self.pipe_end_cnt += 1
            else:
                self.pipe_end_cnt = 0

            if self.pipe_end_cnt >= self.pipe_end_limit:
                self.get_logger().info(
                    f"Boru sonu: {len(ordered)} marker → SURFACE")
                self._set_state(ST_SURFACE)
                self.halt()
                return
            return

        # SURFACE
        if self.state == ST_SURFACE:
            if not self.surface_on_done:
                self._set_state(ST_DONE)
                self.halt()
                return
            send_heartbeat(self.master)
            send_manual(self.master, x=0, y=0,
                        z=clamp(500 + abs(self.spd_surface), 0, 1000), r=0)
            if self._state_elapsed() > self.surface_secs:
                self._set_state(ST_DONE)
            return

        # DONE
        if self.state == ST_DONE:
            self._print_result(ordered)
            self._set_state(ST_RETURN)
            return

        # RETURN
        if self.state == ST_RETURN:
            if self._return_idx >= len(self._return_log):
                self.halt()
                self._set_state(ST_HOME)
                return
            x, y, r, dt = self._return_log[self._return_idx]
            self.move(
                int(-x * self.return_cx),
                int(-y * self.return_cy),
                int(-r * self.return_cr)
            )
            if time.time() - self._return_step_start >= dt:
                self._return_idx += 1
                self._return_step_start = time.time()
            return

        # HOME
        if self.state == ST_HOME:
            self.halt()
            self.get_logger().info("Başlangıç noktasına dönüldü.")
            return

    # ── Sonuç yazdır & kaydet ─────────────────────────────────
    def _print_result(self, ordered):
        if self._result_printed:
            return
        self._result_printed = True
        sep    = "=" * 52
        result = ",".join(str(x) for x in ordered)
        lines  = [
            sep,
            "  TAC Challenge 2026 — PIPELINE INSPECTION",
            "  GOREV TAMAMLANDI",
            f"  Marker sirasi : {result or 'YOK'}",
            f"  Toplam marker : {len(ordered)}",
            sep,
        ]
        for l in lines:
            self.get_logger().info(l)

        try:
            with open(self.result_file, "w") as f:
                f.write("\n".join(lines) + "\n")
            self.get_logger().info(f"  Sonuç dosyası: {self.result_file}")
        except Exception as e:
            self.get_logger().error(f"  Dosya kaydedilemedi: {e}")

    # ── Ana döngü ─────────────────────────────────────────────
    def run(self):
        self.connect()
        self._set_state(ST_ARM)
        if not self.arm():
            return
        self._set_state(ST_SEARCH)

        hb_thread = threading.Thread(
            target=self._heartbeat_thread, daemon=True)
        hb_thread.start()

        # Kamera: ROS2 topic veya doğrudan OpenCV
        cap = None
        if not self.use_ros_camera:
            cap = cv2.VideoCapture(self.camera_id)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
            if not cap.isOpened():
                self.get_logger().error(
                    f"Kamera açılamadı: {self.camera_id}")
                return

        self.get_logger().info("Başladı! q=çık  p=duraklat")
        paused = False
        params = self._get_pipe_params()

        try:
            while self._running and rclpy.ok():
                # ROS2 callback'leri işle (non-blocking)
                rclpy.spin_once(self, timeout_sec=0.001)

                if paused:
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord('p'):
                        paused = False
                        self.get_logger().info("DEVAM")
                    elif k == ord('q'):
                        break
                    continue

                # ── Frame al ─────────────────────────────────
                if self.use_ros_camera:
                    raw = self._get_ros_frame()
                    if raw is None:
                        time.sleep(0.01)
                        continue
                else:
                    ret, raw = cap.read()
                    if not ret:
                        self.get_logger().warn("Kamera frame'i alınamadı!")
                        time.sleep(0.1)
                        continue

                frame = cv2.resize(
                    cv2.rotate(raw, cv2.ROTATE_180),
                    (self.frame_w, self.frame_h))

                now = time.time()
                dt  = now - self._last_frame_time
                self._last_frame_time = now
                if self.state in (ST_SEARCH, ST_FOLLOW):
                    self.move_log.append((self._last_x, self._last_y, self._last_r, dt))

                # ── Pipe detection ────────────────────────────
                mask, contour, center, rect, angle = detect_pipe(
                    frame, params)

                self.pipe_lost_cnt = (
                    0 if center is not None
                    else self.pipe_lost_cnt + 1)

                # ── ArUco detection ───────────────────────────────────────
                if self.state not in (ST_RETURN, ST_HOME):
                    corners, ids = detect_aruco(
                        frame, mask, self.detector, self.clahe)
                    if ids is not None and len(corners) > 0:
                        centers_y = [int(c[0][:, 1].mean()) for c in corners]
                        nearest_idx = max(range(len(centers_y)), key=lambda i: centers_y[i])
                        ids_flat = [ids[nearest_idx][0]]
                    else:
                        ids_flat = []
                    ordered = self.confirmer.update(ids_flat)
                else:
                    corners, ids = [], None
                    ids_flat = []
                    ordered = self.confirmer.ordered

                for mid in self.confirmer.new_ids:
                    self.get_logger().info(
                        f"  MARKER ID {mid:3d} onaylandı | "
                        f"Toplam: {','.join(str(x) for x in ordered)}")

                # ── State machine ─────────────────────────────
                prev_state = self.state
                self._state_machine(center, angle, ids_flat, ordered)

                if prev_state != ST_SURFACE and self.state == ST_SURFACE:
                    self._print_result(ordered)

                if self.state == ST_HOME:
                    break

                # ── ROS2 publish ──────────────────────────────
                self._publish_state()
                self._publish_markers(ordered)
                self._publish_pipe_info(center, angle)

                # ── Debug görüntüsü ──────────────────────────
                if self.show_display or self.publish_debug:
                    vis = draw_debug(
                        frame, mask, contour, rect, center,
                        angle, self.state, self.last_err,
                        ordered, corners, ids, params)

                    if self.publish_debug:
                        self._publish_debug_image(vis)

                    # if self.show_display:
                        # cv2.imshow("TAC Autonomous (ROS2)", vis)

                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    self.get_logger().info("Çıkılıyor...")
                    break
                elif k == ord('p'):
                    paused = True
                    self.halt()
                    self.get_logger().info("DURAKLADI")

        except KeyboardInterrupt:
            self.get_logger().info("Ctrl+C — durduruluyor.")

        finally:
            self._running = False
            self.get_logger().info("DISARM ediliyor...")
            self.halt()
            time.sleep(0.2)
            if self.master:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 0, 0, 0, 0, 0, 0, 0)
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            self.get_logger().info("Bitti.")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = TacAutonomousNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
