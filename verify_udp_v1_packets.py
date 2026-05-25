"""verify_udp_v1_packets.py

Step 4 helper.

Listens on the same UDP port as the C++ camera_pose_subscriber, parses v1
packets emitted by `track_robot_and_box_multicam.py --udp-publish` (without
--motion-file), and prints rolling stats:

  * packet rate (Hz)
  * staleness
  * torso pos/quat / box pos/quat (latest)
  * comparison to a step-1 calibration JSON (if --calib-json provided)

Use this BEFORE touching the C++ side. Workflow:

  Terminal A (PC):
    python3 track_robot_and_box_multicam.py \
        --cam1-serial ... --udp-publish \
        --csv-out /tmp/_verify.csv

  Terminal B (PC, separate shell):
    python3 verify_udp_v1_packets.py \
        --calib-json config/start_pose_calibration.json

The listener prints every 1.0s. Expected:
  * pkt_rate ~ 30-60 Hz (matches tracker frame rate)
  * torso_v / box_v almost always 1
  * latest torso pos ~= calibration torso_pos_mean (mm-level agreement)
  * latest box pos ~= calibration object_pos_mean

If those checks pass, the v1 wire format is rock-solid and we can wire
C++ step 5 without surprises.
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Parser for v1 ASCII packet (17 numeric fields, no prefix)
# "<ts_ns> <torso_v> <tx> <ty> <tz> <tqw> <tqx> <tqy> <tqz>
#  <box_v>   <bx> <by> <bz> <bqw> <bqx> <bqy> <bqz>\n"
# ---------------------------------------------------------------------------
def parse_v1(buf: bytes):
    s = buf.decode("ascii", errors="replace").strip()
    # Tolerate v2 packets silently — return None so they're not mis-counted.
    if s.startswith("v2 ") or s.startswith("v2\t"):
        return None
    parts = s.split()
    if len(parts) != 17:
        return None
    try:
        ts_ns = int(parts[0])
        torso_v = int(parts[1])
        tx, ty, tz = (float(x) for x in parts[2:5])
        tqw, tqx, tqy, tqz = (float(x) for x in parts[5:9])
        box_v = int(parts[9])
        bx, by, bz = (float(x) for x in parts[10:13])
        bqw, bqx, bqy, bqz = (float(x) for x in parts[13:17])
    except ValueError:
        return None
    return {
        "ts_ns": ts_ns,
        "torso_v": torso_v,
        "torso_pos":  np.array([tx, ty, tz]),
        "torso_quat": np.array([tqw, tqx, tqy, tqz]),
        "box_v": box_v,
        "box_pos":  np.array([bx, by, bz]),
        "box_quat": np.array([bqw, bqx, bqy, bqz]),
    }


def quat_norm(q):
    n = float(np.linalg.norm(q))
    return q / n if n > 1e-12 else q


def quat_dot_abs(a, b):
    return float(abs(np.dot(quat_norm(a), quat_norm(b))))


def quat_angle_deg(a, b):
    d = max(0.0, min(1.0, quat_dot_abs(a, b)))
    return float(np.degrees(2.0 * np.arccos(d)))


def load_calib(path: Path):
    d = json.loads(path.read_text())
    return {
        "torso_pos":  np.asarray(d["torso_pos_mean_xyz_m"], float),
        "torso_quat": np.asarray(d["torso_quat_mean_wxyz"], float),
        "object_pos":  np.asarray(d["object_pos_mean_xyz_m"], float),
        "object_quat": np.asarray(d["object_quat_mean_wxyz"], float),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bind", default="0.0.0.0",
                   help="UDP bind address (default 0.0.0.0)")
    p.add_argument("--port", type=int, default=9999,
                   help="UDP port (default 9999, matches camera_pose_subscriber default)")
    p.add_argument("--calib-json", type=Path, default=None,
                   help="Optional step-1 JSON to compare live torso/box "
                        "pose against (mm-level agreement expected if the "
                        "robot+box are still in start pose).")
    p.add_argument("--print-every-sec", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help="If >0, exit after this many seconds.")
    args = p.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.bind, args.port))
    sock.settimeout(0.5)
    print(f"[verify] listening on {args.bind}:{args.port}  (Ctrl-C to exit)")

    calib = None
    if args.calib_json is not None:
        calib = load_calib(args.calib_json)
        print(f"[verify] loaded calibration: {args.calib_json}")

    n_total = 0
    n_v1    = 0
    n_bad   = 0
    n_v2_seen = 0
    last_print = time.monotonic()
    start_t    = last_print
    last_pkt   = None
    last_recv_t = None

    try:
        while True:
            try:
                buf, _ = sock.recvfrom(8192)
            except socket.timeout:
                buf = None

            now = time.monotonic()
            if buf is not None:
                n_total += 1
                pkt = parse_v1(buf)
                if pkt is None:
                    if buf.startswith(b"v2 ") or buf.startswith(b"v2\t"):
                        n_v2_seen += 1
                    else:
                        n_bad += 1
                else:
                    n_v1 += 1
                    last_pkt = pkt
                    last_recv_t = now

            if (now - last_print) >= args.print_every_sec:
                dt = now - last_print
                rate = n_v1 / max(dt, 1e-6)
                print(f"--- {time.strftime('%H:%M:%S')} | "
                      f"pkt_rate={rate:5.1f} Hz | "
                      f"v1={n_v1:6d} v2={n_v2_seen:4d} bad={n_bad:4d}")
                if last_pkt is not None and last_recv_t is not None:
                    age_ms = (now - last_recv_t) * 1000.0
                    tp = last_pkt["torso_pos"]
                    tq = last_pkt["torso_quat"]
                    bp = last_pkt["box_pos"]
                    bq = last_pkt["box_quat"]
                    print(f"    age={age_ms:5.1f} ms  "
                          f"torso_v={last_pkt['torso_v']} box_v={last_pkt['box_v']}")
                    print(f"    torso_pos = [{tp[0]:+.4f}, {tp[1]:+.4f}, {tp[2]:+.4f}] m  "
                          f"torso_quat = [{tq[0]:+.3f}, {tq[1]:+.3f}, {tq[2]:+.3f}, {tq[3]:+.3f}]")
                    print(f"    box_pos   = [{bp[0]:+.4f}, {bp[1]:+.4f}, {bp[2]:+.4f}] m  "
                          f"box_quat   = [{bq[0]:+.3f}, {bq[1]:+.3f}, {bq[2]:+.3f}, {bq[3]:+.3f}]")
                    if calib is not None:
                        tp_err_mm = float(np.linalg.norm(tp - calib["torso_pos"]) * 1000.0)
                        tq_err_deg = quat_angle_deg(tq, calib["torso_quat"])
                        bp_err_mm = float(np.linalg.norm(bp - calib["object_pos"]) * 1000.0)
                        bq_err_deg = quat_angle_deg(bq, calib["object_quat"])
                        print(f"    vs calib  : torso pos_err={tp_err_mm:6.1f} mm  "
                              f"quat_err={tq_err_deg:5.2f} deg | "
                              f"box pos_err={bp_err_mm:6.1f} mm  quat_err={bq_err_deg:5.2f} deg")
                n_v1 = 0
                n_v2_seen = 0
                n_bad = 0
                last_print = now

            if args.max_seconds > 0 and (now - start_t) >= args.max_seconds:
                print(f"[verify] reached --max-seconds {args.max_seconds}, exiting.")
                break

    except KeyboardInterrupt:
        print()
        print(f"[verify] stopped by user. total packets received: {n_total}")
    finally:
        sock.close()


if __name__ == "__main__":
    main()
