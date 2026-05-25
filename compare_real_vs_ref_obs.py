"""compare_real_vs_ref_obs.py

Step 6 (validation): compute the *torso-relative* object pose for both

  REAL  = start_pose_calibration  -->  T_sim_lab  -->  torso body frame
  REF   = NPZ frame 0             -->  torso body frame (directly)

and print them side-by-side. This is the single most important sanity check
before deploy: the policy was trained on the REF distribution, so the REAL
side at the start pose has to land in a *similar pocket* — same xyz quadrant
relative to torso, similar yaw on the box, etc. — or the policy will
immediately diverge.

We compute the full set of obs the tag-history policy reads:

  motion_anchor_pos_b      (3,)
  motion_anchor_ori_b      (6,)  6D rotation
  object_pos_torso         (3,)
  object_ori6_torso        (6,)
  ref_object_pos_torso     (3,)
  ref_object_ori6_torso    (6,)

The first two compare torso-of-ref vs torso-of-real (so non-zero IFF the
robot's actual torso pose differs from NPZ's). The middle two are
"current real object expressed in current real torso" (what the policy
sees from the AprilTag stream at deploy). The last two are "NPZ object
expressed in current real torso" (the *target* pose the policy is asked
to follow).

A perfect start pose makes (object_pos_torso, object_ori6_torso) ≈
(ref_object_pos_torso, ref_object_ori6_torso), meaning real box is
already where ref says it should be. Any deviation is exactly the
"setup error" the policy will have to correct.

Usage:
  python3 compare_real_vs_ref_obs.py \
    --calib-json config/start_pose_calibration.json \
    --t-sim-lab config/T_sim_lab.json \
    --r-obj-tag config/R_obj_tag.json \
    --ref-npz unitree_rl_mjlab/deploy/robots/g1/config/policy/mimic/sub8_45/params/sub8_largebox_045_original_extended.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils.runtime_alignment import (
    R_BODY_FLIP,
    compute_six_obs,
    quat_wxyz_to_R,
    R_to_quat_wxyz,
    rotation_to_rot6d,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def transform_pos(T: np.ndarray, p_lab: np.ndarray) -> np.ndarray:
    return T[:3, :3] @ np.asarray(p_lab, float) + T[:3, 3]


def transform_R(T: np.ndarray, R_lab: np.ndarray, body_flip: bool) -> np.ndarray:
    R_eff = R_lab @ R_BODY_FLIP if body_flip else R_lab
    return T[:3, :3] @ R_eff


def angle_between_R_deg(A: np.ndarray, B: np.ndarray) -> float:
    dR = A.T @ B
    c = float(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def fmt_vec(v, w=8, p=4):
    return "[" + ", ".join(f"{float(x):+{w}.{p}f}" for x in np.asarray(v).reshape(-1)) + "]"


def fmt_diff(a, b, w=8, p=4):
    d = np.asarray(a, float) - np.asarray(b, float)
    return fmt_vec(d, w, p)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calib-json", type=Path, required=True,
                   help="Step 1 output (start_pose_calibration.json).")
    p.add_argument("--t-sim-lab", type=Path, required=True,
                   help="Step 2 output (T_sim_lab.json).")
    p.add_argument("--r-obj-tag", type=Path, required=True,
                   help="Step 3 output (R_obj_tag.json; identity if skipped).")
    p.add_argument("--ref-npz", type=Path,
                   default=Path("unitree_rl_mjlab/deploy/robots/g1/config/"
                                "policy/mimic/sub8_45/params/"
                                "sub8_largebox_045_original_extended.npz"),
                   help="Original (unaligned) reference motion NPZ.")
    p.add_argument("--ref-frame", type=int, default=0)
    p.add_argument("--torso-body-idx", type=int, default=15,
                   help="NPZ body idx for torso_link (default 15).")
    args = p.parse_args()

    # ----- Load calib (real lab-frame poses) -----
    calib = json.loads(args.calib_json.read_text())
    real_torso_pos_lab = np.array(calib["torso_pos_mean_xyz_m"], float)
    real_torso_R_lab   = np.array(calib["torso_R_mean_3x3"],     float)
    real_obj_pos_lab   = np.array(calib["object_pos_mean_xyz_m"], float)
    real_obj_R_lab     = np.array(calib["object_R_mean_3x3"],    float)

    # ----- Load T_sim_lab -----
    T = json.loads(args.t_sim_lab.read_text())
    T_sim_lab = np.array(T["T_sim_lab_4x4"], float)

    # ----- Load R_obj_tag -----
    R_obj_tag = np.array(json.loads(args.r_obj_tag.read_text())["R_obj_tag_3x3"],
                         float)

    # ----- Load NPZ ref frame -----
    npz = np.load(args.ref_npz)
    ref_torso_pos_sim = np.asarray(npz["body_pos_w"][args.ref_frame, args.torso_body_idx], float)
    ref_torso_quat    = np.asarray(npz["body_quat_w"][args.ref_frame, args.torso_body_idx], float)
    ref_torso_R_sim   = quat_wxyz_to_R(ref_torso_quat)
    ref_obj_pos_sim   = np.asarray(npz["object_pos_w"][args.ref_frame], float)
    ref_obj_quat      = np.asarray(npz["object_quat_w"][args.ref_frame], float)
    ref_obj_R_sim     = quat_wxyz_to_R(ref_obj_quat)

    # ----- Transform real (lab) -> sim -----
    real_torso_pos_sim = transform_pos(T_sim_lab, real_torso_pos_lab)
    real_torso_R_sim   = transform_R(T_sim_lab, real_torso_R_lab, body_flip=True)
    real_obj_pos_sim   = transform_pos(T_sim_lab, real_obj_pos_lab)
    # Box: NO body_flip (tracker already in mjlab AABB). Apply R_obj_tag at the end.
    real_obj_R_sim     = transform_R(T_sim_lab, real_obj_R_lab, body_flip=False) @ R_obj_tag

    # ----- Six obs -----
    obs = compute_six_obs(
        real_torso_pos=real_torso_pos_sim, real_torso_R=real_torso_R_sim,
        real_box_pos  =real_obj_pos_sim,   real_box_R  =real_obj_R_sim,
        ref_torso_pos =ref_torso_pos_sim,  ref_torso_R =ref_torso_R_sim,
        ref_box_pos   =ref_obj_pos_sim,    ref_box_R   =ref_obj_R_sim,
    )

    map_b   = obs["motion_anchor_pos_b"]
    mao_b   = obs["motion_anchor_ori_b"]
    opt     = obs["object_pos_torso"]
    oot     = obs["object_ori6_torso"]
    rpt     = obs["ref_object_pos_torso"]
    rot     = obs["ref_object_ori6_torso"]

    # ----- Summary scalars (apples-to-apples in REAL torso frame, matching the
    #       obs the policy actually reads). The deploy obs subtracts these two
    #       internally; we surface that diff as the "box alignment error":
    #         box_pos_err_b = object_pos_torso - ref_object_pos_torso
    #                       = R_t_real.T @ (real_box_pos - ref_box_pos)
    #       (real torso cancels). The rotation analogue is the geodesic angle
    #       between the two rotations the policy sees:
    #         R_real_b = R_t_real.T @ R_real_box
    #         R_ref_b  = R_t_real.T @ R_ref_box
    #         diff = angle between R_real_b and R_ref_b
    #              = angle between R_real_box and R_ref_box
    #       (real torso cancels, so the angle is invariant to torso choice).
    R_t_real = real_torso_R_sim
    box_pos_err_b_m = R_t_real.T @ (real_obj_pos_sim - ref_obj_pos_sim)
    box_pos_err_mm  = box_pos_err_b_m * 1000.0
    box_pos_err_mag_mm = float(np.linalg.norm(box_pos_err_mm))
    box_R_err_deg = angle_between_R_deg(real_obj_R_sim, ref_obj_R_sim)
    torso_pos_err_mm = float(np.linalg.norm(real_torso_pos_sim - ref_torso_pos_sim) * 1000.0)
    torso_R_err_deg  = angle_between_R_deg(real_torso_R_sim, ref_torso_R_sim)

    # ----- Print -----
    print("=" * 78)
    print(" REAL vs REF — torso-relative object pose validation")
    print("=" * 78)
    print()
    print(f"  ref_npz   : {args.ref_npz}")
    print(f"  ref_frame : {args.ref_frame}")
    print(f"  calib     : {args.calib_json}")
    print(f"  T_sim_lab : delta_yaw={T.get('delta_yaw_deg', '?'):+.3f} deg")
    print(f"  R_obj_tag : {('identity' if np.allclose(R_obj_tag, np.eye(3), atol=1e-6) else 'non-identity')}")
    print()
    print("--- inputs (sim frame, after T_sim_lab) ---")
    print(f"  real torso pos : {fmt_vec(real_torso_pos_sim)}   ref torso pos : {fmt_vec(ref_torso_pos_sim)}")
    print(f"  real obj   pos : {fmt_vec(real_obj_pos_sim)}   ref obj   pos : {fmt_vec(ref_obj_pos_sim)}")
    print()
    print("--- the 12 actor obs at this start pose ---")
    print()
    print("                          value (3 or 6 dim)                                   note")
    print("  motion_anchor_pos_b   = " + fmt_vec(map_b, 8, 4) + "   (real torso pos vs ref torso pos, in real torso frame)")
    print("  motion_anchor_ori_b   = " + fmt_vec(mao_b, 7, 3) + "   (real-vs-ref torso ORIENTATION, 6D)")
    print()
    print("  object_pos_torso      = " + fmt_vec(opt, 8, 4) + "   (REAL box, in REAL torso frame)")
    print("  ref_object_pos_torso  = " + fmt_vec(rpt, 8, 4) + "   (REF  box, in REAL torso frame)")
    print("  diff (real - ref)     = " + fmt_vec(opt - rpt, 8, 4) + "   <-- box positional alignment error")
    print()
    print("  object_ori6_torso     = " + fmt_vec(oot, 7, 3) + "   (REAL box ori, in REAL torso frame)")
    print("  ref_object_ori6_torso = " + fmt_vec(rot, 7, 3) + "   (REF  box ori, in REAL torso frame)")
    print("  diff                  = " + fmt_vec(np.array(oot) - np.array(rot), 7, 3))
    print()
    print("--- summary scalars (REAL torso frame, matching deploy obs) ---")
    print(f"  box pos err (in real torso frame)  : [{box_pos_err_mm[0]:+6.1f}, "
          f"{box_pos_err_mm[1]:+6.1f}, {box_pos_err_mm[2]:+6.1f}] mm   |.| = {box_pos_err_mag_mm:.1f} mm")
    print(f"  box rotation err                   : {box_R_err_deg:.2f} deg")
    print(f"  torso position residual            : {torso_pos_err_mm:.1f} mm  "
          f"(EXPECTED to be ~25cm: NPZ torso bent fwd vs FixStand vertical)")
    print(f"  torso orientation residual         : {torso_R_err_deg:.2f} deg "
          f"(EXPECTED to be ~90 deg: same posture mismatch)")
    print()
    print("--- interpretation guide ---")
    print("  GOOD start pose: box_rel_pos < 30 mm  and  box_rel_rot < 5 deg")
    print("  ACCEPTABLE    : box_rel_pos < 70 mm  and  box_rel_rot < 10 deg")
    print("  TOO FAR (re-do): box_rel_pos > 100 mm or  box_rel_rot > 15 deg")
    print("  Torso residual is expected to be large (~10cm/100deg) because NPZ frame 0")
    print("  has the torso bent ~25 deg forward, while FixStand is vertical. The policy")
    print("  is robust to this — what matters most is the BOX side.")
    print("=" * 78)


if __name__ == "__main__":
    main()
