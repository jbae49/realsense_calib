#!/usr/bin/env python3
"""Align reference motion npz into the lab/origin frame defined by a shadow CSV.

This script computes a single 4x4 rigid transform T_lab_world such that:

    T_lab_world @ T_world_torso_npz_frame_F  ==  T_lab_torso_csv_mean

where:
  * T_world_torso_npz_frame_F = pose of body --anchor-body-idx (default 15 = torso_link)
                                at npz frame --ref-frame (default 0)
                                NOTE: in the 30-body depth-first order of the mjlab
                                G1 model, body 15 is torso_link. body 16 is
                                left_shoulder_pitch_link. The training cfg uses
                                anchor_body_name="torso_link" — see
                                mjlab/tasks/tracking/config/g1/env_cfgs.py.
                                Earlier versions of this script defaulted to 16,
                                which was the cause of the post-alignment 130°/163°
                                rotation errors observed during deploy.
  * T_lab_torso_csv_mean      = mean torso pose over the first --num-frames of CSV

The transform is then applied to every world-frame quantity in the npz:
  * body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w
  * object_pos_w, object_quat_w, object_lin_vel_w, object_ang_vel_w

Joint-space and time-invariant fields (joint_pos, joint_vel, contact_mask, fps,
...) are kept verbatim. The npz's INTERNAL relative configuration (e.g., box yaw
relative to torso) is preserved -- this script does NOT change relative poses,
only the global frame they're expressed in.

The original --ref-npz is NEVER modified. A sidecar JSON with diagnostics and
the transform matrix is written next to --out-npz.

Typical use:
    python align_npz_to_lab.py \\
        --obs-csv outputs/test_actor_obs_20260523_152217.csv \\
        --ref-npz humanoid_project/src/assets/OmniRetarget/processed/sub8_largebox_045_original.npz \\
        --out-npz outputs/sub8_45_coords_processed_v1.npz
"""

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np


# ---------------- math helpers ----------------
def quat_to_R(q):
    """quaternion (wxyz) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    """3x3 R -> quaternion (wxyz). Shepperd's method."""
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    return q / np.linalg.norm(q)


def quat_mul_left_const(q_const, quat_arr):
    """Hamilton product q_const ⊗ quat_arr (last dim = 4, wxyz)."""
    w1, x1, y1, z1 = q_const
    a = quat_arr.reshape(-1, 4)
    w2, x2, y2, z2 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    out = np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)
    return out.reshape(quat_arr.shape)


def average_R(quats):
    """Mean rotation via SVD-projected mean of rotation matrices."""
    Rs = np.stack([quat_to_R(q) for q in quats], axis=0).mean(axis=0)
    U, _, Vt = np.linalg.svd(Rs)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = U @ Vt
    return R


def angle_between_R(R1, R2):
    return float(np.arccos(np.clip((np.trace(R1.T @ R2) - 1) / 2, -1, 1)))


# ---------------- CSV loader ----------------
def load_csv_pose(rows, prefix):
    out = []
    for r in rows:
        try:
            pos = np.array([float(r[f"{prefix}_pos_{a}"]) for a in "xyz"])
            qu  = np.array([float(r[f"{prefix}_quat_{a}"]) for a in "wxyz"])
            if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(qu)):
                continue
            out.append((pos, qu))
        except (KeyError, ValueError):
            continue
    return out


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--obs-csv", required=True,
                    help="CSV from track_robot_and_box_multicam.py (lab-frame torso/obj poses)")
    ap.add_argument("--ref-npz", required=True,
                    help="Original reference motion npz (NEVER modified)")
    ap.add_argument("--out-npz", required=True,
                    help="Output transformed npz path. Refuses to overwrite unless --force.")
    ap.add_argument("--ref-frame", type=int, default=0,
                    help="Anchor npz frame to align CSV initial pose (default 0)")
    ap.add_argument("--num-frames", type=int, default=30,
                    help="Number of CSV frames to average for initial pose (default 30)")
    ap.add_argument("--anchor-body-idx", type=int, default=15,
                    help="Body index of torso_link in npz body axis (default 16 for G1)")
    ap.add_argument("--force", action="store_true",
                    help="Allow overwriting --out-npz if it exists")
    ap.add_argument("--full-rotation", action="store_true",
                    help=("Use full 6-DoF rotation alignment (DANGEROUS: bakes in any "
                          "torso roll/pitch from the start pose, e.g. a bent posture, "
                          "as a global frame tilt). Default = yaw-only + axis-flip, "
                          "which preserves gravity and only matches heading."))
    ap.add_argument("--torso-forward-axis", default="+x",
                    choices=["+x","-x","+y","-y"],
                    help=("Robot forward direction expressed in torso local frame "
                          "(default '+x' which matches G1 torso_link)."))
    args = ap.parse_args()

    out_path = Path(args.out_npz)
    if out_path.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite: {out_path}. Use --force or pick a different --out-npz."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load CSV mean poses ----
    with open(args.obs_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    torso_data = load_csv_pose(rows, "torso")
    obj_data   = load_csv_pose(rows, "obj")

    if len(torso_data) < 5:
        raise ValueError(f"Not enough valid torso frames in CSV (got {len(torso_data)})")
    n_torso = min(args.num_frames, len(torso_data))
    csv_torso_pos = np.mean([p for p, _ in torso_data[:n_torso]], axis=0)
    csv_torso_R   = average_R([q for _, q in torso_data[:n_torso]])

    n_obj = min(args.num_frames, len(obj_data)) if len(obj_data) >= 5 else 0
    if n_obj > 0:
        csv_obj_pos = np.mean([p for p, _ in obj_data[:n_obj]], axis=0)
        csv_obj_R   = average_R([q for _, q in obj_data[:n_obj]])
    else:
        csv_obj_pos = None
        csv_obj_R   = None

    # ---- Body-frame convention conversion (z-down body -> z-up body) ----
    # The CSV's torso_quat / obj_quat come from track_robot_and_box_multicam.py
    # which expresses orientation in the lab/origin-tag frame (z=DOWN). The
    # AprilTag-derived torso pose inherits the head-tag's body convention
    # (+x=forward, +y=right, +z=down), while mjlab's MuJoCo body frame is
    # (+x=forward, +y=left, +z=up). They differ by a 180° rotation about the
    # local +x axis = diag(1,-1,-1). Without this, post-alignment torso quat
    # collides with the npz reference by ~180°, which is the root cause of the
    # 178° rotation residual we observed for clean csv captures (qw std ~0.001).
    R_BODY_FLIP = np.diag([1.0, -1.0, -1.0])
    csv_torso_R = csv_torso_R @ R_BODY_FLIP
    # NB: box convention is checked empirically. The npz uses a body frame for
    # the box that already matches z-up. The CSV box pose comes from a tag-based
    # T_world_box where the box geom convention follows tag axis orientations.
    # We do NOT apply R_BODY_FLIP to obj because empirically it makes errors
    # worse (172° vs 69°), suggesting box is already in the correct convention.

    # ---- Load npz frame F anchor ----
    ref = np.load(args.ref_npz)
    f0    = args.ref_frame
    a_idx = args.anchor_body_idx

    npz_torso_pos = ref["body_pos_w"][f0, a_idx].astype(float)
    npz_torso_R   = quat_to_R(ref["body_quat_w"][f0, a_idx])
    npz_obj_pos   = ref["object_pos_w"][f0].astype(float)
    npz_obj_R     = quat_to_R(ref["object_quat_w"][f0])

    # ---- Compute single rigid transform ----
    # Robot's forward axis in torso local frame
    fwd_axis_map = {"+x": [1,0,0], "-x": [-1,0,0], "+y": [0,1,0], "-y": [0,-1,0]}
    fwd_local = np.array(fwd_axis_map[args.torso_forward_axis], dtype=float)

    if args.full_rotation:
        # Full 6-DoF: matches torso 3D orientation exactly (legacy behaviour)
        T_lab_torso_csv = np.eye(4)
        T_lab_torso_csv[:3, :3] = csv_torso_R
        T_lab_torso_csv[:3, 3]  = csv_torso_pos
        T_world_torso_npz = np.eye(4)
        T_world_torso_npz[:3, :3] = npz_torso_R
        T_world_torso_npz[:3, 3]  = npz_torso_pos
        T_lab_world = T_lab_torso_csv @ np.linalg.inv(T_world_torso_npz)
        R_lw = T_lab_world[:3, :3]
        t_lw = T_lab_world[:3, 3]
        align_mode = "full-rotation"
    else:
        # Yaw-only + axis-flip (DEFAULT, preserves gravity)
        # Convention: lab uses pupil_apriltags floor-tag frame -> +z = down.
        #             npz/sim uses +z = up.
        # R_flip = diag(1, -1, -1) maps sim x→lab x, sim y→lab -y, sim z→lab -z,
        # i.e. sim's vertical-up (+z) maps to lab's vertical-up (-z). Right-handed preserved.
        R_flip = np.diag([1.0, -1.0, -1.0])

        # Forward direction (in torso local frame) projected onto each frame's
        # horizontal plane (xy) gives the yaw heading.
        fwd_npz_in_lab = R_flip @ npz_torso_R @ fwd_local  # post-flip, in lab frame
        fwd_csv_in_lab = csv_torso_R @ fwd_local
        yaw_npz = float(np.arctan2(fwd_npz_in_lab[1], fwd_npz_in_lab[0]))
        yaw_csv = float(np.arctan2(fwd_csv_in_lab[1], fwd_csv_in_lab[0]))
        delta_yaw = yaw_csv - yaw_npz
        c, s = np.cos(delta_yaw), np.sin(delta_yaw)
        R_yaw = np.array([[c, -s, 0.0],
                          [s,  c, 0.0],
                          [0.0, 0.0, 1.0]])

        R_lw = R_yaw @ R_flip                         # gravity-preserving
        t_lw = csv_torso_pos - R_lw @ npz_torso_pos   # place torso at csv mean
        T_lab_world = np.eye(4)
        T_lab_world[:3, :3] = R_lw
        T_lab_world[:3, 3]  = t_lw
        align_mode = (f"yaw-only (delta_yaw={np.degrees(delta_yaw):+.2f}°, "
                      f"yaw_npz={np.degrees(yaw_npz):+.2f}°, yaw_csv={np.degrees(yaw_csv):+.2f}°)")

    q_lw = R_to_quat(R_lw)

    # ---- Apply to every world-frame quantity ----
    new_data = {k: ref[k].copy() for k in ref.files}

    def apply_pos(arr):
        return arr @ R_lw.T + t_lw

    def apply_quat(arr):
        return quat_mul_left_const(q_lw, arr)

    def apply_vec(arr):
        return arr @ R_lw.T

    transformed = []
    for key, fn in [
        ("body_pos_w",     apply_pos),
        ("body_quat_w",    apply_quat),
        ("body_lin_vel_w", apply_vec),
        ("body_ang_vel_w", apply_vec),
        ("object_pos_w",     apply_pos),
        ("object_quat_w",    apply_quat),
        ("object_lin_vel_w", apply_vec),
        ("object_ang_vel_w", apply_vec),
    ]:
        if key in ref.files:
            new_data[key] = fn(ref[key].astype(float))
            transformed.append(key)

    # ---- Diagnostics ----
    new_torso_pos_f0 = new_data["body_pos_w"][f0, a_idx]
    new_torso_R_f0   = quat_to_R(new_data["body_quat_w"][f0, a_idx])
    err_torso_pos_mm = float(np.linalg.norm(new_torso_pos_f0 - csv_torso_pos) * 1000)
    err_torso_R_deg  = float(np.degrees(angle_between_R(new_torso_R_f0, csv_torso_R)))

    err_obj_pos_mm = None
    err_obj_R_deg  = None
    if csv_obj_pos is not None:
        new_obj_pos_f0 = new_data["object_pos_w"][f0]
        new_obj_R_f0   = quat_to_R(new_data["object_quat_w"][f0])
        err_obj_pos_mm = float(np.linalg.norm(new_obj_pos_f0 - csv_obj_pos) * 1000)
        err_obj_R_deg  = float(np.degrees(angle_between_R(new_obj_R_f0, csv_obj_R)))

    diag = {
        "obs_csv": str(Path(args.obs_csv).resolve()),
        "ref_npz": str(Path(args.ref_npz).resolve()),
        "out_npz": str(out_path.resolve()),
        "align_mode": align_mode,
        "torso_forward_axis": args.torso_forward_axis,
        "ref_frame": int(f0),
        "anchor_body_idx": int(a_idx),
        "num_csv_torso_frames_used": int(n_torso),
        "num_csv_obj_frames_used":   int(n_obj),
        "transformed_keys": transformed,
        "T_lab_world": T_lab_world.tolist(),
        "R_lab_world": R_lw.tolist(),
        "t_lab_world": t_lw.tolist(),
        "q_lab_world_wxyz": q_lw.tolist(),
        "csv_mean_torso_pos": csv_torso_pos.tolist(),
        "csv_mean_obj_pos":   None if csv_obj_pos is None else csv_obj_pos.tolist(),
        "npz_anchor_torso_pos_orig": npz_torso_pos.tolist(),
        "npz_anchor_obj_pos_orig":   npz_obj_pos.tolist(),
        "post_align_torso_pos_err_mm": err_torso_pos_mm,
        "post_align_torso_R_err_deg":  err_torso_R_deg,
        "post_align_obj_pos_err_mm":   err_obj_pos_mm,
        "post_align_obj_R_err_deg":    err_obj_R_deg,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    np.savez(out_path, **new_data)
    diag_path = out_path.with_suffix(out_path.suffix + ".alignment.json")
    diag_path.write_text(json.dumps(diag, indent=2))

    # ---- Print summary ----
    print("=" * 64)
    print(f"Aligned npz : {out_path}")
    print(f"Diagnostics : {diag_path}")
    print(f"From CSV    : {args.obs_csv}  (using {n_torso} torso / {n_obj} obj frames)")
    print(f"Anchor      : npz frame {f0}, body {a_idx} (assumed torso_link)")
    print(f"Align mode  : {align_mode}")
    print()
    print("Transformed keys:")
    for k in transformed:
        print(f"   {k}  shape={new_data[k].shape}")
    print()
    # Health check thresholds (empirical):
    #   torso pos / rot err: ~0 by construction in full-rotation mode; in
    #     yaw-only mode the rotation error captures roll/pitch mismatch
    #     between csv and npz. >10 deg = csv torso quat probably bad
    #     (head/pelvis tag occlusion, or wrong head→torso transform).
    #   obj   pos err: distance between where the npz expects the box at f0
    #     and where the csv actually saw it. >100 mm = box not placed where
    #     the policy expects, OR npz f0 wasn't the very start of the motion.
    #   obj   rot err: >15 deg = box yaw/orientation mismatch. policy will
    #     see large ref_object_ori6_torso, which is in distribution but at
    #     the tail.
    POS_WARN_MM, POS_FAIL_MM = 50.0, 100.0
    R_WARN_DEG,   R_FAIL_DEG   = 5.0,  10.0
    OBJ_R_WARN, OBJ_R_FAIL     = 10.0, 15.0

    def _flag(val, warn, fail, unit):
        if val is None:
            return ""
        if val > fail: return f"  ❌ FAIL (>{fail:g}{unit})"
        if val > warn: return f"  ⚠ WARN (>{warn:g}{unit})"
        return "  ✓ OK"

    print("Post-alignment residuals at frame 0:")
    print(f"   torso pos err : {err_torso_pos_mm:7.2f} mm{_flag(err_torso_pos_mm, POS_WARN_MM, POS_FAIL_MM, ' mm')}")
    print(f"   torso rot err : {err_torso_R_deg:7.2f} deg{_flag(err_torso_R_deg, R_WARN_DEG, R_FAIL_DEG, ' deg')}")
    if err_obj_pos_mm is not None:
        print(f"   obj   pos err : {err_obj_pos_mm:7.2f} mm{_flag(err_obj_pos_mm, POS_WARN_MM, POS_FAIL_MM, ' mm')}")
        print(f"   obj   rot err : {err_obj_R_deg:7.2f} deg{_flag(err_obj_R_deg, OBJ_R_WARN, OBJ_R_FAIL, ' deg')}")
    print()

    any_fail = (
        err_torso_R_deg  > R_FAIL_DEG or
        (err_obj_pos_mm  is not None and err_obj_pos_mm  > POS_FAIL_MM) or
        (err_obj_R_deg   is not None and err_obj_R_deg   > OBJ_R_FAIL)
    )
    if any_fail:
        print("🚨 ALIGNMENT QUALITY FAILED. Do NOT deploy this npz —")
        print("   the policy will see step-0 obs far outside its training")
        print("   distribution and is likely to fall immediately.")
        print("   Likely causes:")
        print("     - head/pelvis tags occluded during csv capture")
        print("     - robot wasn't standing still during csv capture")
        print("     - box not in the expected starting position")
        print("     - wrong --ref-npz (frame 0 doesn't correspond to start pose)")
        print("   Recapture a clean csv (head_visible & pelvis_visible ~100%)")
        print("   and rerun, or use --full-rotation to also fit roll/pitch.")
        print()

    print("T_lab_world (4x4):")
    for row in T_lab_world:
        print(f"   [{row[0]:+.4f}, {row[1]:+.4f}, {row[2]:+.4f}, {row[3]:+.4f}]")
    print("=" * 64)
    if any_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
