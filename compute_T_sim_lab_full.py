"""compute_T_sim_lab_full.py

Step 1 of the 3-step sub8_45 sim2real calibration plan (2026-05-25).

Given:
  * A multicam tracker CSV captured in FixStand pose with the new
    `pelvis_link_*` columns (= pelvis link/root pose estimated from pelvis
    FRONT tag id 8 via the hardcoded R_tag_to_body + offset).
  * The reference motion NPZ in its original SIM world frame.

Compute a full SE(3) T_sim_lab:
    pose_sim = T_sim_lab @ pose_lab

that lands the FixStand pelvis_link directly on top of the NPZ frame 0
pelvis. All 6 DOF of the transform are fit (rotation + translation), unlike
the legacy yaw-only solver in `compute_T_sim_lab_pelvis.py`.

Also prints diagnostic info:
  - lab origin (0,0,0)  in sim frame  (= where the floor tag 1 lands)
  - mean box position   in sim frame  (= where the current box ends up)
  - NPZ frame 0 expected positions for comparison

Usage:
  python3 compute_T_sim_lab_full.py \
    --tracker-csv outputs/fixstand_step1_<ts>.csv \
    --ref-npz unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/sub8_45/params/sub8_largebox_045_original_extended.npz \
    --out-json config/T_sim_lab.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from utils.runtime_alignment import (
    R_to_quat_wxyz,
    average_pose_samples,
    quat_wxyz_to_R,
)


def _safe_float(row, key):
    v = row.get(key, "")
    if v in (None, ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_samples_from_csv(csv_path):
    """Return (pelvis_pos_list, pelvis_R_list, box_pos_list, box_R_list).

    Skips any frame where the required columns are missing or non-numeric.
    """
    pelvis_pos, pelvis_R, box_pos, box_R = [], [], [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            px = _safe_float(row, "pelvis_link_pos_x")
            py = _safe_float(row, "pelvis_link_pos_y")
            pz = _safe_float(row, "pelvis_link_pos_z")
            qw = _safe_float(row, "pelvis_link_quat_w")
            qx = _safe_float(row, "pelvis_link_quat_x")
            qy = _safe_float(row, "pelvis_link_quat_y")
            qz = _safe_float(row, "pelvis_link_quat_z")
            if None in (px, py, pz, qw, qx, qy, qz):
                continue
            pelvis_pos.append(np.array([px, py, pz]))
            pelvis_R.append(quat_wxyz_to_R(np.array([qw, qx, qy, qz])))

            bx = _safe_float(row, "obj_pos_x")
            by = _safe_float(row, "obj_pos_y")
            bz = _safe_float(row, "obj_pos_z")
            bqw = _safe_float(row, "obj_quat_w")
            bqx = _safe_float(row, "obj_quat_x")
            bqy = _safe_float(row, "obj_quat_y")
            bqz = _safe_float(row, "obj_quat_z")
            if None not in (bx, by, bz, bqw, bqx, bqy, bqz):
                box_pos.append(np.array([bx, by, bz]))
                box_R.append(quat_wxyz_to_R(np.array([bqw, bqx, bqy, bqz])))

    return pelvis_pos, pelvis_R, box_pos, box_R


def R_to_rpy_deg(R):
    """ZYX intrinsic Euler angles (yaw, pitch, roll) in degrees."""
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    roll = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    pitch = float(np.degrees(np.arctan2(-R[2, 0], sy)))
    yaw = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tracker-csv", required=True,
                    help="multicam tracker CSV captured in FixStand pose (must "
                         "contain pelvis_link_* columns)")
    ap.add_argument("--ref-npz", required=True,
                    help="reference motion NPZ (original sim frame)")
    ap.add_argument("--out-json", default="config/T_sim_lab.json")
    args = ap.parse_args()

    # ---- read CSV ----
    pelvis_pos_samples, pelvis_R_samples, box_pos_samples, box_R_samples = \
        read_samples_from_csv(args.tracker_csv)
    n = len(pelvis_pos_samples)
    if n < 10:
        raise SystemExit(
            f"too few pelvis_link samples: {n}. Make sure the tracker was "
            "running with tag 8 visible and the CSV has pelvis_link_* columns.")
    pelvis_pos_lab, pelvis_R_lab = average_pose_samples(pelvis_pos_samples, pelvis_R_samples)

    print(f"[step1] read {n} pelvis_link samples from {args.tracker_csv}")
    print(f"  pelvis_link_pos_lab (mean) = {pelvis_pos_lab}")
    print(f"  pelvis_link_quat (wxyz)    = {R_to_quat_wxyz(pelvis_R_lab)}")
    rl_lab, pi_lab, yw_lab = R_to_rpy_deg(pelvis_R_lab)
    print(f"  pelvis_link rpy (deg)      = roll={rl_lab:+.2f}  pitch={pi_lab:+.2f}  yaw={yw_lab:+.2f}")

    box_pos_lab_mean = None
    if box_pos_samples:
        box_pos_lab_mean, box_R_lab_mean = average_pose_samples(box_pos_samples, box_R_samples)
        print(f"  [bonus] box samples = {len(box_pos_samples)}, mean lab pos = {box_pos_lab_mean}")

    # ---- read NPZ ref frame 0 ----
    npz = np.load(args.ref_npz)
    pelvis_pos_sim = npz["body_pos_w"][0, 0].astype(float)
    pelvis_R_sim = quat_wxyz_to_R(npz["body_quat_w"][0, 0])
    torso_pos_sim = npz["body_pos_w"][0, 15].astype(float)
    obj_pos_sim = npz["object_pos_w"][0].astype(float)
    print()
    print(f"[step1] NPZ ref frame 0:")
    print(f"  pelvis_pos_sim         = {pelvis_pos_sim}")
    print(f"  pelvis_quat_sim (wxyz) = {R_to_quat_wxyz(pelvis_R_sim)}")
    rl_sim, pi_sim, yw_sim = R_to_rpy_deg(pelvis_R_sim)
    print(f"  pelvis rpy (deg)       = roll={rl_sim:+.2f}  pitch={pi_sim:+.2f}  yaw={yw_sim:+.2f}")
    print(f"  torso_link_pos_sim     = {torso_pos_sim}")
    print(f"  object_pos_sim         = {obj_pos_sim}")

    # ---- compute full 6-DOF T_sim_lab ----
    # pose_sim = T_sim_lab @ pose_lab  (pelvis lands on ref pelvis):
    #   R_sim_lab = pelvis_R_sim @ pelvis_R_lab.T
    #   t_sim_lab = pelvis_pos_sim - R_sim_lab @ pelvis_pos_lab
    R_sim_lab = pelvis_R_sim @ pelvis_R_lab.T
    t_sim_lab = pelvis_pos_sim - R_sim_lab @ pelvis_pos_lab
    T_sim_lab = np.eye(4)
    T_sim_lab[:3, :3] = R_sim_lab
    T_sim_lab[:3, 3] = t_sim_lab

    # ---- sanity ----
    pelvis_sim_check = R_sim_lab @ pelvis_pos_lab + t_sim_lab
    err_pos = float(np.linalg.norm(pelvis_sim_check - pelvis_pos_sim))
    R_check = R_sim_lab @ pelvis_R_lab
    err_R = float(np.linalg.norm(R_check - pelvis_R_sim))
    print()
    print(f"[step1] T_sim_lab fit residuals (should be ~0):")
    print(f"  pelvis pos residual    = {err_pos:.3e} m")
    print(f"  pelvis rot residual    = {err_R:.3e} (frobenius)")

    # ---- R_sim_lab decomposition ----
    roll, pitch, yaw = R_to_rpy_deg(R_sim_lab)
    print()
    print(f"[step1] R_sim_lab decomposition:")
    print(f"  roll  = {roll:+.2f} deg   (gravity tilt; nonzero -> pelvis not perfectly upright)")
    print(f"  pitch = {pitch:+.2f} deg   (gravity tilt)")
    print(f"  yaw   = {yaw:+.2f} deg   (lab vs sim heading)")
    print(f"  translation t_sim_lab = {t_sim_lab}")

    # ---- lab origin in sim ----
    lab_origin_in_sim = (R_sim_lab @ np.zeros(3) + t_sim_lab).tolist()
    print()
    print(f"[step1] Lab origin (floor tag 1) in sim frame:")
    print(f"  pos_sim = {lab_origin_in_sim}")

    # ---- box in sim ----
    box_block = None
    if box_pos_lab_mean is not None:
        box_pos_in_sim = R_sim_lab @ box_pos_lab_mean + t_sim_lab
        diff = box_pos_in_sim - obj_pos_sim
        print()
        print(f"[step1] Current box position vs NPZ frame 0 ref:")
        print(f"  box_lab_mean      = {box_pos_lab_mean}")
        print(f"  box_in_sim        = {box_pos_in_sim}")
        print(f"  npz_ref0_obj      = {obj_pos_sim}")
        print(f"  delta             = {diff}    |.|={np.linalg.norm(diff)*1000:.1f}mm")
        box_block = {
            "box_pos_lab_mean": box_pos_lab_mean.tolist(),
            "box_pos_in_sim": box_pos_in_sim.tolist(),
            "delta_vs_npz_ref0": diff.tolist(),
            "delta_mm": float(np.linalg.norm(diff) * 1000.0),
        }

    # ---- save JSON ----
    out = {
        "description": "Full 6-DOF T_sim_lab from FixStand pelvis_link <-> NPZ frame 0 pelvis.",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_csv": str(args.tracker_csv),
        "source_npz": str(args.ref_npz),
        "mode": "pelvis-link-full-6dof",
        "n_samples": int(n),
        "T_sim_lab_4x4": T_sim_lab.tolist(),
        "R_sim_lab_3x3": R_sim_lab.tolist(),
        "t_sim_lab_xyz": t_sim_lab.tolist(),
        "rpy_deg": {"roll": roll, "pitch": pitch, "yaw": yaw},
        "lab_origin_in_sim_xyz": lab_origin_in_sim,
        "pelvis_link_lab_mean": {
            "pos_xyz": pelvis_pos_lab.tolist(),
            "quat_wxyz": R_to_quat_wxyz(pelvis_R_lab).tolist(),
            "rpy_deg": [rl_lab, pi_lab, yw_lab],
        },
        "npz_frame0": {
            "pelvis_pos_sim": pelvis_pos_sim.tolist(),
            "pelvis_quat_sim_wxyz": R_to_quat_wxyz(pelvis_R_sim).tolist(),
            "pelvis_rpy_deg": [rl_sim, pi_sim, yw_sim],
            "torso_link_pos_sim": torso_pos_sim.tolist(),
            "object_pos_sim": obj_pos_sim.tolist(),
        },
        "residuals": {
            "pelvis_pos_m": err_pos,
            "pelvis_R_frobenius": err_R,
        },
    }
    if box_block is not None:
        out["box"] = box_block

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"[step1] saved {args.out_json}")


if __name__ == "__main__":
    main()
