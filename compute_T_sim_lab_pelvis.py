"""compute_T_sim_lab_pelvis.py

Step 2 of the simplified deploy calibration flow.

Given:
  * `config/start_pose_calibration.json`   (Step 1 output: mean lab-frame
    pelvis/torso/object poses)
  * the REFERENCE motion NPZ in its original SIM world frame (NOT the lab-
    aligned `_processed_v2.npz` that the legacy align_npz_to_lab.py wrote!)

compute the single yaw-only rigid transform

        T_sim_lab          (4x4)
        pose_sim = T_sim_lab @ pose_lab

that maps any lab-frame pose into the NPZ's sim world frame. The anchor is
the PELVIS (body 0) at NPZ frame 0. The pelvis is chosen instead of the
torso because:

  * pelvis is approximately vertical in both FixStand and NPZ frame 0
    (small roll/pitch mismatch),
  * torso is bent ~25 deg forward in NPZ frame 0 but vertical in FixStand,
    so anchoring on torso would bake that mismatch into yaw and translation.

Only yaw is solved; roll/pitch are pinned by the lab-frame "+Z = down" vs
sim "+Z = up" convention (= one 180-degree flip around X). Translation is
the full xyz shift that lands the real pelvis on top of the ref pelvis.

This script does NOT touch the NPZ file. It just emits one JSON with
T_sim_lab + diagnostics. Step 3 (R_obj_tag) and the runtime C++ code both
read from this JSON.

Usage:
  python3 compute_T_sim_lab_pelvis.py \
    --calib-json config/start_pose_calibration.json \
    --ref-npz unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/sub8_45/params/sub8_largebox_045_original_extended.npz \
    --out-json config/T_sim_lab.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from utils.runtime_alignment import (
    R_BODY_FLIP,
    R_to_quat_wxyz,
    compute_T_sim_lab_pelvis_yaw,
    quat_wxyz_to_R,
    rotation_to_rot6d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def angle_between_R_deg(A: np.ndarray, B: np.ndarray) -> float:
    """Geodesic distance between two 3x3 rotation matrices, in degrees."""
    dR = A.T @ B
    c = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def transform_pos_lab_to_sim(T_sim_lab: np.ndarray, pos_lab: np.ndarray) -> np.ndarray:
    R = T_sim_lab[:3, :3]
    t = T_sim_lab[:3, 3]
    return R @ np.asarray(pos_lab, float) + t


def transform_R_lab_to_sim(T_sim_lab: np.ndarray, R_lab: np.ndarray,
                           body_flip: bool = False) -> np.ndarray:
    """R_sim = R_sim_lab @ R_lab @ R_BODY_FLIP (only torso needs body_flip).

    R_BODY_FLIP = diag(1,-1,-1) converts the head-tag z-down body convention
    into mjlab z-up body convention. Pelvis tag uses --pelvis-tag-up-axis
    to encode its convention upstream, so no flip is applied. Box is also
    not flipped (tracker already emits in mjlab AABB body frame).
    """
    R_eff = R_lab @ R_BODY_FLIP if body_flip else R_lab
    return T_sim_lab[:3, :3] @ R_eff


def load_calib(path: Path) -> dict:
    data = json.loads(path.read_text())
    expected = ("pelvis_pos_mean_xyz_m", "pelvis_R_mean_3x3",
                "torso_pos_mean_xyz_m",  "torso_R_mean_3x3",
                "object_pos_mean_xyz_m", "object_R_mean_3x3")
    missing = [k for k in expected if k not in data]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    return data


def load_ref_pose(npz_path: Path, frame: int, pelvis_idx: int, torso_idx: int):
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ not found: {npz_path}")
    npz = np.load(npz_path)
    n_frames = int(npz["body_pos_w"].shape[0])
    n_bodies = int(npz["body_pos_w"].shape[1])
    if not (0 <= frame < n_frames):
        raise ValueError(f"frame {frame} out of [0,{n_frames})")
    if not (0 <= pelvis_idx < n_bodies):
        raise ValueError(f"pelvis_idx {pelvis_idx} out of [0,{n_bodies})")
    if not (0 <= torso_idx < n_bodies):
        raise ValueError(f"torso_idx {torso_idx} out of [0,{n_bodies})")

    pelvis_pos  = np.asarray(npz["body_pos_w"][frame, pelvis_idx], float)
    pelvis_quat = np.asarray(npz["body_quat_w"][frame, pelvis_idx], float)
    torso_pos   = np.asarray(npz["body_pos_w"][frame, torso_idx],  float)
    torso_quat  = np.asarray(npz["body_quat_w"][frame, torso_idx], float)
    has_object  = ("object_pos_w" in npz.files and "object_quat_w" in npz.files)
    if has_object:
        obj_pos  = np.asarray(npz["object_pos_w"][frame], float)
        obj_quat = np.asarray(npz["object_quat_w"][frame], float)
    else:
        obj_pos  = np.zeros(3, float)
        obj_quat = np.array([1.0, 0.0, 0.0, 0.0])

    return {
        "pelvis_pos": pelvis_pos, "pelvis_R": quat_wxyz_to_R(pelvis_quat),
        "pelvis_quat": pelvis_quat,
        "torso_pos":  torso_pos,  "torso_R":  quat_wxyz_to_R(torso_quat),
        "torso_quat": torso_quat,
        "object_pos": obj_pos,    "object_R": quat_wxyz_to_R(obj_quat),
        "object_quat": obj_quat,
        "has_object": has_object,
        "n_frames": n_frames, "n_bodies": n_bodies,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib-json", type=Path, required=True,
                   help="Step 1 output (start_pose_calibration.json).")
    p.add_argument("--ref-npz", type=Path,
                   default=Path("unitree_rl_mjlab/deploy/robots/g1/config/"
                                "policy/mimic/sub8_45/params/"
                                "sub8_largebox_045_original_extended.npz"),
                   help="Original (unaligned) reference motion NPZ.")
    p.add_argument("--ref-frame", type=int, default=0,
                   help="Which NPZ frame is the start pose (default 0).")
    p.add_argument("--pelvis-body-idx", type=int, default=0,
                   help="NPZ body index for pelvis (default 0).")
    p.add_argument("--torso-body-idx", type=int, default=15,
                   help="NPZ body index for torso_link (default 15 in mjlab "
                        "G1 30-body order).")
    p.add_argument("--real-fwd-axis-tag-local", type=str, default="-z",
                   help="Which pelvis-tag-local axis points toward robot's "
                        "chest (body forward). For back-mounted G1 pelvis "
                        "tag with --pelvis-tag-up-axis=-y in the tracker, "
                        "body forward = -z_tag (default).")
    p.add_argument("--ref-fwd-axis-body-local", type=str, default="+x",
                   help="Robot forward in mjlab body coords (G1 = +x).")
    p.add_argument("--out-json", type=Path, required=True,
                   help="Output JSON path (e.g. config/T_sim_lab.json).")
    args = p.parse_args()

    calib = load_calib(args.calib_json)
    ref   = load_ref_pose(args.ref_npz, args.ref_frame,
                          args.pelvis_body_idx, args.torso_body_idx)

    real_pelvis_pos = np.asarray(calib["pelvis_pos_mean_xyz_m"], float)
    real_pelvis_R   = np.asarray(calib["pelvis_R_mean_3x3"],     float)
    real_torso_pos  = np.asarray(calib["torso_pos_mean_xyz_m"],  float)
    real_torso_R    = np.asarray(calib["torso_R_mean_3x3"],      float)
    real_obj_pos    = np.asarray(calib["object_pos_mean_xyz_m"], float)
    real_obj_R      = np.asarray(calib["object_R_mean_3x3"],     float)

    T_sim_lab, diag = compute_T_sim_lab_pelvis_yaw(
        real_pelvis_pos_lab = real_pelvis_pos,
        real_pelvis_R_lab   = real_pelvis_R,
        ref_pelvis_pos_sim  = ref["pelvis_pos"],
        ref_pelvis_R_sim    = ref["pelvis_R"],
        real_fwd_axis_tag_local = args.real_fwd_axis_tag_local,
        ref_fwd_axis_body_local = args.ref_fwd_axis_body_local,
    )

    # ----------------- Validation -----------------
    # 1) Pelvis position should land EXACTLY on the ref pelvis (it's the anchor).
    pelvis_pos_sim_pred = transform_pos_lab_to_sim(T_sim_lab, real_pelvis_pos)
    pelvis_pos_err_mm   = float(np.linalg.norm(pelvis_pos_sim_pred - ref["pelvis_pos"]) * 1000.0)

    # 2) Torso position — xy should be close (rigid body link), z may differ
    #    if torso is bent forward in NPZ vs vertical in real start pose.
    torso_pos_sim_pred = transform_pos_lab_to_sim(T_sim_lab, real_torso_pos)
    torso_pos_err_xyz_mm = [float(v * 1000.0) for v in (torso_pos_sim_pred - ref["torso_pos"])]
    torso_pos_err_mag_mm = float(np.linalg.norm(torso_pos_sim_pred - ref["torso_pos"]) * 1000.0)
    # Torso rotation is body_flip=True (head tag z-down body conv).
    torso_R_sim_pred = transform_R_lab_to_sim(T_sim_lab, real_torso_R, body_flip=True)
    torso_R_err_deg  = angle_between_R_deg(torso_R_sim_pred, ref["torso_R"])

    # 3) Object position — should be in the right ballpark; rotation will
    #    have a fixed offset that step 3 (R_obj_tag) is responsible for.
    obj_pos_sim_pred = transform_pos_lab_to_sim(T_sim_lab, real_obj_pos)
    obj_pos_err_xyz_mm = [float(v * 1000.0) for v in (obj_pos_sim_pred - ref["object_pos"])]
    obj_pos_err_mag_mm = float(np.linalg.norm(obj_pos_sim_pred - ref["object_pos"]) * 1000.0)
    obj_R_sim_pred = transform_R_lab_to_sim(T_sim_lab, real_obj_R, body_flip=False)
    obj_R_err_deg  = angle_between_R_deg(obj_R_sim_pred, ref["object_R"])

    # ----------------- Save -----------------
    R_sim_lab = T_sim_lab[:3, :3]
    t_sim_lab = T_sim_lab[:3, 3]
    q_sim_lab = R_to_quat_wxyz(R_sim_lab)

    out = {
        "meta": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "calib_json": str(args.calib_json.resolve()),
            "ref_npz":   str(args.ref_npz.resolve()),
            "ref_frame": int(args.ref_frame),
            "pelvis_body_idx": int(args.pelvis_body_idx),
            "torso_body_idx":  int(args.torso_body_idx),
            "real_fwd_axis_tag_local": args.real_fwd_axis_tag_local,
            "ref_fwd_axis_body_local": args.ref_fwd_axis_body_local,
            "method": "pelvis-anchored yaw-only (gravity-preserving)",
        },
        "T_sim_lab_4x4": T_sim_lab.tolist(),
        "R_sim_lab_3x3": R_sim_lab.tolist(),
        "t_sim_lab_xyz_m": [float(x) for x in t_sim_lab],
        "q_sim_lab_wxyz":  [float(x) for x in q_sim_lab],
        "delta_yaw_deg":   float(np.degrees(diag["delta_yaw_rad"]))
                           if diag.get("delta_yaw_rad") is not None else None,
        "diagnostics": {
            "ref_pelvis_pos_sim_m":  [float(x) for x in ref["pelvis_pos"]],
            "real_pelvis_pos_lab_m": [float(x) for x in real_pelvis_pos],
            "pelvis_pos_residual_mm": pelvis_pos_err_mm,
            "ref_torso_pos_sim_m":  [float(x) for x in ref["torso_pos"]],
            "torso_pos_pred_sim_m": [float(x) for x in torso_pos_sim_pred],
            "torso_pos_residual_xyz_mm": torso_pos_err_xyz_mm,
            "torso_pos_residual_mag_mm": torso_pos_err_mag_mm,
            "torso_R_residual_deg":      torso_R_err_deg,
            "ref_object_pos_sim_m":  [float(x) for x in ref["object_pos"]],
            "object_pos_pred_sim_m": [float(x) for x in obj_pos_sim_pred],
            "object_pos_residual_xyz_mm": obj_pos_err_xyz_mm,
            "object_pos_residual_mag_mm": obj_pos_err_mag_mm,
            "object_R_residual_deg":      obj_R_err_deg,
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2))

    # ----------------- Console summary -----------------
    print("===========================================================")
    print(f"T_sim_lab (yaw-only, pelvis anchor)  ->  {args.out_json}")
    print(f"  ref_npz   : {args.ref_npz}")
    print(f"  ref_frame : {args.ref_frame}    (n_frames={ref['n_frames']}, n_bodies={ref['n_bodies']})")
    print(f"  delta_yaw : {out['delta_yaw_deg']:+.3f} deg")
    print(f"  t_sim_lab : [{t_sim_lab[0]:+.4f}, {t_sim_lab[1]:+.4f}, {t_sim_lab[2]:+.4f}] m")
    print("--- validation residuals ---")
    print(f"  pelvis pos          : {pelvis_pos_err_mm:7.3f} mm   (should be ~0; pelvis is the anchor)")
    print(f"  torso  pos (xyz/mag): [{torso_pos_err_xyz_mm[0]:+.1f}, "
          f"{torso_pos_err_xyz_mm[1]:+.1f}, {torso_pos_err_xyz_mm[2]:+.1f}] mm / "
          f"{torso_pos_err_mag_mm:.1f} mm   (xy small, z can be ~10cm if torso bent fwd in NPZ)")
    print(f"  torso  R            : {torso_R_err_deg:7.2f} deg   (depends on FixStand-vs-bent-fwd diff)")
    print(f"  object pos (xyz/mag): [{obj_pos_err_xyz_mm[0]:+.1f}, "
          f"{obj_pos_err_xyz_mm[1]:+.1f}, {obj_pos_err_xyz_mm[2]:+.1f}] mm / "
          f"{obj_pos_err_mag_mm:.1f} mm   (small if start pose well placed)")
    print(f"  object R            : {obj_R_err_deg:7.2f} deg   (THIS is what step 3 R_obj_tag will fix)")
    print("===========================================================")


if __name__ == "__main__":
    main()
