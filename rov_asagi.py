#!/usr/bin/env python3
"""
tac_dive.py — TAC Challenge 2026
Aracı Z ekseninde düz aşağı indirir, ardından hover'a geçer.

Kullanım:
    python3 tac_dive.py
    python3 tac_dive.py --device /dev/ttyACM0 --baud 115200
    python3 tac_dive.py --depth_secs 5 --dive_throttle 300

ROS2 ile:
    ros2 run tac_challenge tac_dive --ros-args \
        -p device:=/dev/ttyACM0 \
        -p depth_secs:=5.0 \
        -p dive_throttle:=300
"""

import time
import threading
import argparse
import os
from pymavlink import mavutil

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32

os.environ['MAVLINK20'] = '1'

# ── State sabitleri ───────────────────────────────────────────
ST_INIT    = "INIT"
ST_ARM     = "ARM"
ST_DIVE    = "DIVE"      # aşağı iniyor
ST_HOVER   = "HOVER"     # hedef derinliğe ulaştı, hover
ST_SURFACE = "SURFACE"   # (opsiyonel) yüzeye dön
ST_DONE    = "DONE"

# ── MAVLink yardımcıları (tac_autonomous_ros2.py'den aynen) ───
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
    z ekseni throttle:
        500  → nötr / hover
        <500 → aşağı in  (örn. 300 = orta hızda iniş)
        >500 → yukarı çık
    x, y, r sıfır bırakılır → düz iniş
    """
    master.mav.manual_control_send(master.target_system, x, y, z, r, 0)

def clamp(v, lo=0, hi=1000):
    return max(lo, min(hi, int(v)))


# ── ROS2 Node ─────────────────────────────────────────────────
class TacDiveNode(Node):
    """
    Tek görevli basit node: aracı Z ekseninde düz aşağı indirir.

    Parametreler:
        device         — seri port (/dev/ttyACM0)
        baud           — baud hızı (115200)
        depth_secs     — kaç saniye aşağı inilsin (varsayılan: 5.0)
        dive_throttle  — iniş throttle'ı, 0-499 arası (varsayılan: 300)
        hover_secs     — hover süresi, sonra opsiyonel yüzey (varsayılan: 3.0)
        surface        — True ise hover'dan sonra yüzeye çık (varsayılan: True)
        surface_secs   — yüzey çıkış süresi (varsayılan: 6.0)
    """

    def __init__(self):
        super().__init__('tac_dive')

        # ── Parametreler ─────────────────────────────────────
        self.declare_parameter('device',          '/dev/ttyACM0')
        self.declare_parameter('baud',            115200)
        self.declare_parameter('depth_secs',      5.0)
        self.declare_parameter('dive_throttle',   300)   # <500 = aşağı
        self.declare_parameter('hover_secs',      3.0)
        self.declare_parameter('surface',         True)
        self.declare_parameter('surface_secs',    6.0)
        self.declare_parameter('surface_throttle',700)   # >500 = yukarı

        g = self.get_parameter
        self.device           = g('device').value
        self.baud             = g('baud').value
        self.depth_secs       = g('depth_secs').value
        self.dive_throttle    = clamp(g('dive_throttle').value, 0, 499)
        self.hover_secs       = g('hover_secs').value
        self.do_surface       = g('surface').value
        self.surface_secs     = g('surface_secs').value
        self.surface_throttle = clamp(g('surface_throttle').value, 501, 1000)

        # ── Publishers ───────────────────────────────────────
        self.pub_state     = self.create_publisher(String,  '/tac/state',     10)
        self.pub_depth_cmd = self.create_publisher(Float32, '/tac/depth_cmd', 10)

        # ── İç durum ─────────────────────────────────────────
        self.state   = ST_INIT
        self.state_t = time.time()
        self.master  = None
        self._running = True
        self._current_z = 500   # heartbeat thread için

        self.get_logger().info("TacDiveNode hazır.")
        self.get_logger().info(
            f"  Iniş süresi   : {self.depth_secs} sn")
        self.get_logger().info(
            f"  Iniş throttle : {self.dive_throttle}  (500=hover, <500=aşağı)")
        self.get_logger().info(
            f"  Hover süresi  : {self.hover_secs} sn")

    # ── Heartbeat thread ─────────────────────────────────────
    def _heartbeat_thread(self):
        """50 ms'de bir heartbeat + manual_control gönderir."""
        while self._running:
            try:
                send_heartbeat(self.master)
                send_manual(self.master, x=0, y=0, z=self._current_z, r=0)
            except Exception:
                pass
            time.sleep(0.05)

    # ── Bağlantı ─────────────────────────────────────────────
    def connect(self):
        self.get_logger().info(
            f"Pixhawk bağlanılıyor: {self.device} @ {self.baud}...")
        self.master = mavutil.mavlink_connection(
            self.device, baud=self.baud, source_system=255)
        self.master.wait_heartbeat()
        self.get_logger().info(
            f"Bağlandı! Sistem ID: {self.master.target_system}")

        # Güvenlik parametreleri
        for name, val in [("FS_PILOT_EN", 0), ("FS_GCS_EN", 0),
                           ("ARMING_CHECK", 0), ("BRD_SAFETYENABLE", 0)]:
            set_param(self.master, name, val)
            self.get_logger().info(f"  {name} = {val}")
        time.sleep(1)

        # MANUAL mod
        mode_id = self.master.mode_mapping()['MANUAL']
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id)
        self.get_logger().info("Mod: MANUAL")

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

    # ── Yardımcılar ──────────────────────────────────────────
    def _set_state(self, s):
        if s == self.state:
            return
        self.get_logger().info(f"STATE: {self.state} → {s}")
        self.state   = s
        self.state_t = time.time()
        msg = String()
        msg.data = s
        self.pub_state.publish(msg)

    def _elapsed(self):
        return time.time() - self.state_t

    def _set_z(self, z):
        """Throttle'ı güncelle ve ROS2'ye yayınla."""
        self._current_z = clamp(z, 0, 1000)
        msg = Float32()
        msg.data = float(self._current_z)
        self.pub_depth_cmd.publish(msg)

    # ── Ana çalışma döngüsü ──────────────────────────────────
    def run(self):
        self.connect()
        self._set_state(ST_ARM)

        if not self.arm():
            return

        # Heartbeat thread başlat
        hb = threading.Thread(target=self._heartbeat_thread, daemon=True)
        hb.start()

        self._set_state(ST_DIVE)
        self.get_logger().info(
            f"DIVE BAŞLIYOR — throttle={self.dive_throttle}, "
            f"süre={self.depth_secs}s")

        try:
            while self._running and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)

                # ── DIVE: aşağı in ───────────────────────────
                if self.state == ST_DIVE:
                    self._set_z(self.dive_throttle)   # <500 = aşağı
                    if self._elapsed() >= self.depth_secs:
                        self.get_logger().info(
                            f"Hedef derinliğe ulaşıldı ({self.depth_secs}s)")
                        self._set_state(ST_HOVER)

                # ── HOVER: yerinde dur ───────────────────────
                elif self.state == ST_HOVER:
                    self._set_z(500)                  # nötr
                    if self._elapsed() >= self.hover_secs:
                        if self.do_surface:
                            self._set_state(ST_SURFACE)
                        else:
                            self._set_state(ST_DONE)

                # ── SURFACE: yüzeye çık ──────────────────────
                elif self.state == ST_SURFACE:
                    self._set_z(self.surface_throttle)  # >500 = yukarı
                    if self._elapsed() >= self.surface_secs:
                        self._set_state(ST_DONE)

                # ── DONE ─────────────────────────────────────
                elif self.state == ST_DONE:
                    self._set_z(500)
                    self.get_logger().info("GÖREV TAMAMLANDI.")
                    break

        except KeyboardInterrupt:
            self.get_logger().info("Ctrl+C — durduruluyor.")

        finally:
            self._running = False
            self._set_z(500)        # önce hover'a al
            time.sleep(0.3)
            self.get_logger().info("DISARM ediliyor...")
            if self.master:
                self.master.mav.command_long_send(
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                    0, 0, 0, 0, 0, 0, 0, 0)
            self.get_logger().info("Bitti.")


# ── Main ──────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = TacDiveNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    # Standalone çalıştırma için argparse desteği
    # (ROS2 parametreleri --ros-args ile de geçersiz kılınabilir)
    parser = argparse.ArgumentParser(description="TAC Dive — Düz Aşağı İniş")
    parser.add_argument('--device',          default='/dev/ttyACM0')
    parser.add_argument('--baud',            type=int, default=115200)
    parser.add_argument('--depth_secs',      type=float, default=5.0,
                        help='Kaç saniye aşağı inilsin')
    parser.add_argument('--dive_throttle',   type=int, default=300,
                        help='İniş throttle (0-499, düşük=hızlı iniş)')
    parser.add_argument('--hover_secs',      type=float, default=3.0)
    parser.add_argument('--no_surface',      action='store_true',
                        help='Yüzeye çıkma, hover\'da kal')
    parser.add_argument('--surface_secs',    type=float, default=6.0)
    parser.add_argument('--surface_throttle',type=int, default=700)
    args = parser.parse_args()

    import sys
    # ROS2 argparse'ı ile çakışmaması için sys.argv temizlenir
    sys.argv = [sys.argv[0]]

    main()
